/**
 * @fileoverview Express server for the multi-agent orchestration backend.
 *
 * Deployed to Google Cloud Run. Accepts POST /api/generate with a { prompt }
 * body, spins up a two-agent team (architect + developer) backed by the custom
 * VertexAdapter, runs them on the prompt, and returns the developer's final
 * text output.
 *
 * Authentication to Vertex AI is handled by VertexAdapter using the
 * vertex_api_api.json service-account key file in the same directory.
 *
 * CORS is permissive to allow requests from Cloudflare Pages.
 */

import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import express from 'express'
import cors from 'cors'

// Framework imports — the package is installed from npm
import {
  AgentRunner,
  ToolRegistry,
  ToolExecutor,
  defineTool,
  registerBuiltInTools,
} from '@jackchen_me/open-multi-agent'

// Custom Vertex AI adapter — reads vertex_api_api.json for credentials
import { VertexAdapter } from './VertexAdapter.js'

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url)
const __dirname  = dirname(__filename)

/** Absolute path to the service-account JSON in the same directory. */
const KEY_FILE_PATH = join(__dirname, 'vertex_api_api.json')

// ---------------------------------------------------------------------------
// Singleton adapter — created once at startup, reused per request.
// Re-using the adapter instance means credential initialisation only happens
// once and the underlying HTTP connections can be pooled.
// ---------------------------------------------------------------------------

let vertexAdapter
try {
  vertexAdapter = new VertexAdapter(KEY_FILE_PATH)
  console.log('[server] VertexAdapter initialised successfully.')
} catch (err) {
  console.error('[server] Failed to initialise VertexAdapter:', err.message)
  console.error('Ensure vertex_api_api.json exists in the same directory as server.js')
  process.exit(1)
}

// ---------------------------------------------------------------------------
// Express app
// ---------------------------------------------------------------------------

const app = express()

// Trust proxy headers (Cloud Run sits behind a load balancer)
app.set('trust proxy', 1)

// CORS — allow all origins; tighten in production by replacing '*' with your
// Cloudflare Pages domain, e.g. 'https://your-app.pages.dev'
app.use(cors({ origin: '*', methods: ['GET', 'POST', 'OPTIONS'] }))
app.options('*', cors())

// Parse JSON bodies
app.use(express.json({ limit: '1mb' }))

// ---------------------------------------------------------------------------
// Health check endpoint
// ---------------------------------------------------------------------------

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// ---------------------------------------------------------------------------
// POST /api/generate
// ---------------------------------------------------------------------------

/**
 * Main generation endpoint.
 *
 * Request body: { "prompt": "..." }
 * Response:     { "output": "...", "architect_output": "...", "meta": { ... } }
 *
 * Agent team:
 *   1. architect — uses gemini-2.5-pro, plans the solution with bash + file_write
 *   2. developer — uses gemini-1.5-pro, implements with bash, file_read, file_write, file_edit
 *
 * Execution flow:
 *   a. Run the architect agent on the user prompt.
 *   b. Feed the architect's output + user prompt to the developer agent.
 *   c. Return the developer's final text output.
 */
app.post('/api/generate', async (req, res) => {
  const { prompt } = req.body ?? {}

  if (!prompt || typeof prompt !== 'string' || prompt.trim() === '') {
    return res.status(400).json({
      error: 'Missing or empty "prompt" field in request body.',
    })
  }

  const requestStart = Date.now()
  console.log(`[/api/generate] Received prompt (${prompt.length} chars)`)

  try {
    // ------------------------------------------------------------------
    // Build independent tool registries for each agent.
    // Each agent gets its own registry so tool allow-lists are isolated.
    // ------------------------------------------------------------------

    /** Build a ToolRegistry pre-loaded with all built-in tools. */
    function makeRegistry() {
      const registry = new ToolRegistry()
      registerBuiltInTools(registry)
      return registry
    }

    // ------------------------------------------------------------------
    // Architect agent
    //   Model: gemini-2.5-pro
    //   Tools: bash, file_write
    //   Role: Analyse the request, produce a technical plan and skeleton code
    // ------------------------------------------------------------------

    const architectRegistry  = makeRegistry()
    const architectExecutor  = new ToolExecutor(architectRegistry)
    const architectRunner    = new AgentRunner(
      vertexAdapter,
      architectRegistry,
      architectExecutor,
      {
        model:        'gemini-2.5-pro',
        agentName:    'architect',
        agentRole:    'Senior software architect who plans solutions',
        systemPrompt: [
          'You are a senior software architect. Your job is to analyse the user\'s request,',
          'produce a crisp technical plan, and write any skeleton files or scaffolding needed.',
          'Focus on structure, interfaces, and overall design — leave deep implementation to the developer.',
          'Be concise and actionable.',
        ].join('\n'),
        allowedTools:  ['bash', 'file_write'],
        maxTurns:      8,
        maxTokens:     4096,
        temperature:   0.4,
      },
    )

    // Run the architect
    console.log('[/api/generate] Running architect agent...')
    const architectUserMsg = [
      { role: 'user', content: [{ type: 'text', text: prompt }] },
    ]
    const architectResult = await architectRunner.run(architectUserMsg)

    const architectOutput = architectResult.output ?? ''
    console.log(
      `[/api/generate] Architect done — ${architectResult.turns} turns, ` +
      `${architectResult.tokenUsage.output_tokens} output tokens`,
    )

    // ------------------------------------------------------------------
    // Developer agent
    //   Model: gemini-1.5-pro
    //   Tools: bash, file_read, file_write, file_edit
    //   Role: Implement based on architect's plan
    // ------------------------------------------------------------------

    const developerRegistry = makeRegistry()
    const developerExecutor = new ToolExecutor(developerRegistry)
    const developerRunner   = new AgentRunner(
      vertexAdapter,
      developerRegistry,
      developerExecutor,
      {
        model:        'gemini-1.5-pro',
        agentName:    'developer',
        agentRole:    'Senior full-stack developer who implements solutions',
        systemPrompt: [
          'You are a senior full-stack developer. You receive a user request and an architect\'s plan.',
          'Your job is to implement the solution completely and correctly.',
          'Write production-quality code, handle edge cases, and provide a clear summary of what you built.',
        ].join('\n'),
        allowedTools:  ['bash', 'file_read', 'file_write', 'file_edit'],
        maxTurns:      12,
        maxTokens:     8192,
        temperature:   0.3,
      },
    )

    // Compose the developer prompt from the original user prompt + architect output
    const developerPromptText = [
      '## User Request',
      prompt,
      '',
      '## Architect\'s Plan',
      architectOutput || '(The architect did not produce a plan — proceed with your own approach.)',
      '',
      '## Your Task',
      'Implement the solution described above. Write the code, run any required commands,',
      'and provide a concise summary of what you built and how to use it.',
    ].join('\n')

    console.log('[/api/generate] Running developer agent...')
    const developerUserMsg = [
      { role: 'user', content: [{ type: 'text', text: developerPromptText }] },
    ]
    const developerResult = await developerRunner.run(developerUserMsg)

    const developerOutput = developerResult.output ?? ''
    console.log(
      `[/api/generate] Developer done — ${developerResult.turns} turns, ` +
      `${developerResult.tokenUsage.output_tokens} output tokens`,
    )

    // ------------------------------------------------------------------
    // Compose and return the response
    // ------------------------------------------------------------------

    const elapsed = Date.now() - requestStart

    return res.json({
      output:           developerOutput,
      architect_output: architectOutput,
      meta: {
        elapsed_ms:    elapsed,
        architect: {
          model:        'gemini-2.5-pro',
          turns:        architectResult.turns,
          token_usage:  architectResult.tokenUsage,
          tool_calls:   architectResult.toolCalls.length,
        },
        developer: {
          model:        'gemini-1.5-pro',
          turns:        developerResult.turns,
          token_usage:  developerResult.tokenUsage,
          tool_calls:   developerResult.toolCalls.length,
        },
      },
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.error('[/api/generate] Error:', message)

    return res.status(500).json({
      error:   'An error occurred while running the agent team.',
      details: message,
    })
  }
})

// ---------------------------------------------------------------------------
// Start server
// ---------------------------------------------------------------------------

const PORT = parseInt(process.env['PORT'] ?? '8080', 10)

app.listen(PORT, () => {
  console.log(`[server] Listening on port ${PORT}`)
  console.log(`[server] Health check: http://localhost:${PORT}/health`)
  console.log(`[server] Generate endpoint: POST http://localhost:${PORT}/api/generate`)
})
