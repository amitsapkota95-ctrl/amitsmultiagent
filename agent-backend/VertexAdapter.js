/**
 * @fileoverview VertexAdapter — custom LLMAdapter for Google Vertex AI.
 *
 * Implements the LLMAdapter interface required by @jackchen_me/open-multi-agent,
 * authenticating via a Google Cloud Service Account JSON file (vertex_api_api.json).
 *
 * The adapter uses @google-cloud/vertexai to call the Gemini family of models.
 * Both chat() (non-streaming) and stream() (streaming) paths are implemented.
 *
 * Authentication flow:
 *   1. The JSON key file path is passed to the constructor.
 *   2. google-auth-library reads the file and issues short-lived OAuth2 tokens.
 *   3. Those tokens are attached as Bearer headers on every Vertex AI API call.
 */

import { readFileSync } from 'node:fs'
import { GoogleAuth } from 'google-auth-library'
import { VertexAI, HarmCategory, HarmBlockThreshold } from '@google-cloud/vertexai'
import { randomUUID } from 'node:crypto'

// ---------------------------------------------------------------------------
// Message conversion helpers
// ---------------------------------------------------------------------------

/**
 * Convert a framework ContentBlock[] into the Vertex AI "parts" array.
 * Vertex AI doesn't distinguish tool_result at the Part level — it uses
 * FunctionResponse parts for tool results, FunctionCall parts for tool use.
 */
function frameworkBlocksToVertexParts(blocks) {
  const parts = []
  for (const block of blocks) {
    switch (block.type) {
      case 'text':
        parts.push({ text: block.text })
        break
      case 'tool_use':
        parts.push({
          functionCall: {
            name: block.name,
            args: block.input,
          },
        })
        break
      case 'tool_result':
        parts.push({
          functionResponse: {
            name: block.tool_use_id, // Vertex matches by name — we use the id as name here
            response: { output: block.content, is_error: block.is_error ?? false },
          },
        })
        break
      case 'image':
        parts.push({
          inlineData: {
            mimeType: block.source.media_type,
            data: block.source.data,
          },
        })
        break
      default:
        // Graceful fallback for unknown block types
        parts.push({ text: `[unsupported block: ${block.type}]` })
    }
  }
  return parts
}

/**
 * Convert framework LLMMessage[] into Vertex AI Content[] (the "history" format).
 *
 * Vertex AI requires role to be "user" or "model" (not "assistant").
 */
function frameworkMessagesToVertexContents(messages) {
  return messages.map((msg) => ({
    role: msg.role === 'assistant' ? 'model' : 'user',
    parts: frameworkBlocksToVertexParts(msg.content),
  }))
}

/**
 * Convert Vertex AI FunctionDeclaration-compatible objects from framework LLMToolDefs.
 */
function frameworkToolsToVertexTools(tools) {
  if (!tools || tools.length === 0) return undefined
  return [
    {
      functionDeclarations: tools.map((t) => ({
        name: t.name,
        description: t.description,
        parameters: t.inputSchema,
      })),
    },
  ]
}

/**
 * Map Vertex AI candidate parts back to framework ContentBlock[].
 */
function vertexPartsToFrameworkBlocks(parts) {
  const blocks = []
  if (!parts) return blocks

  for (const part of parts) {
    if (part.text !== undefined && part.text !== null) {
      blocks.push({ type: 'text', text: part.text })
    } else if (part.functionCall) {
      blocks.push({
        type: 'tool_use',
        id: `fc_${randomUUID()}`,
        name: part.functionCall.name,
        input: part.functionCall.args ?? {},
      })
    }
    // FunctionResponse parts are model-outgoing — we don't expect them in responses
  }
  return blocks
}

/**
 * Map Vertex AI finishReason to a framework stop_reason string.
 */
function mapFinishReason(finishReason) {
  switch (finishReason) {
    case 'STOP': return 'end_turn'
    case 'MAX_TOKENS': return 'max_tokens'
    case 'SAFETY': return 'safety'
    case 'TOOL_CALLS':
    case 'FUNCTION_CALL': return 'tool_use'
    default: return finishReason ?? 'end_turn'
  }
}

// ---------------------------------------------------------------------------
// Safety settings (permissive defaults for a dev/orchestration backend)
// ---------------------------------------------------------------------------

const SAFETY_SETTINGS = [
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,       threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT,        threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH },
]

// ---------------------------------------------------------------------------
// VertexAdapter
// ---------------------------------------------------------------------------

/**
 * LLM adapter backed by Google Vertex AI (Gemini models).
 *
 * Authenticates using an explicit Service Account JSON key file.
 *
 * Thread-safe — one instance may serve multiple concurrent agent runs.
 *
 * @example
 * ```js
 * import { VertexAdapter } from './VertexAdapter.js'
 * const adapter = new VertexAdapter('./vertex_api_api.json')
 * const response = await adapter.chat(messages, { model: 'gemini-1.5-pro' })
 * ```
 */
export class VertexAdapter {
  /** Human-readable provider name expected by the framework. */
  name = 'vertex'

  #vertexAI
  #projectId
  #location

  /**
   * @param {string} keyFilePath - Absolute or relative path to the service account JSON.
   * @param {string} [location='us-central1'] - Vertex AI region.
   */
  constructor(keyFilePath, location = 'us-central1') {
    // Parse the key file to extract project info and set GOOGLE_APPLICATION_CREDENTIALS
    const keyData = JSON.parse(readFileSync(keyFilePath, 'utf8'))
    this.#projectId = keyData.project_id

    if (!this.#projectId) {
      throw new Error(`VertexAdapter: could not read project_id from ${keyFilePath}`)
    }

    this.#location = location

    // Initialise the Vertex AI client with explicit credentials
    this.#vertexAI = new VertexAI({
      project: this.#projectId,
      location: this.#location,
      googleAuthOptions: {
        keyFilename: keyFilePath,
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
      },
    })

    console.log(
      `[VertexAdapter] Initialised — project=${this.#projectId} location=${this.#location}`,
    )
  }

  // -------------------------------------------------------------------------
  // chat() — non-streaming completion
  // -------------------------------------------------------------------------

  /**
   * Send a chat request to Vertex AI and return the complete LLMResponse.
   *
   * @param {import('@jackchen_me/open-multi-agent').LLMMessage[]} messages
   * @param {import('@jackchen_me/open-multi-agent').LLMChatOptions} options
   * @returns {Promise<import('@jackchen_me/open-multi-agent').LLMResponse>}
   */
  async chat(messages, options) {
    const model = this.#getGenerativeModel(options)

    // Split messages into history (all but last) + final user message
    const { history, lastUserParts } = this.#splitMessages(messages)

    const chat = model.startChat({
      history,
      systemInstruction: options.systemPrompt
        ? { parts: [{ text: options.systemPrompt }] }
        : undefined,
    })

    const result = await chat.sendMessage(lastUserParts)
    const candidate = result.response.candidates?.[0]

    if (!candidate) {
      throw new Error('[VertexAdapter] API returned no candidates')
    }

    const content = vertexPartsToFrameworkBlocks(candidate.content?.parts)
    const usage = result.response.usageMetadata ?? {}

    return {
      id: `vertex-${randomUUID()}`,
      content,
      model: options.model,
      stop_reason: mapFinishReason(candidate.finishReason),
      usage: {
        input_tokens: usage.promptTokenCount ?? 0,
        output_tokens: usage.candidatesTokenCount ?? 0,
      },
    }
  }

  // -------------------------------------------------------------------------
  // stream() — streaming completion via AsyncIterable<StreamEvent>
  // -------------------------------------------------------------------------

  /**
   * Stream a response from Vertex AI, yielding StreamEvents incrementally.
   *
   * @param {import('@jackchen_me/open-multi-agent').LLMMessage[]} messages
   * @param {import('@jackchen_me/open-multi-agent').LLMStreamOptions} options
   * @returns {AsyncIterable<import('@jackchen_me/open-multi-agent').StreamEvent>}
   */
  async *stream(messages, options) {
    try {
      const model = this.#getGenerativeModel(options)
      const { history, lastUserParts } = this.#splitMessages(messages)

      const chat = model.startChat({
        history,
        systemInstruction: options.systemPrompt
          ? { parts: [{ text: options.systemPrompt }] }
          : undefined,
      })

      const streamResult = await chat.sendMessageStream(lastUserParts)

      // Accumulate tool calls to emit as tool_use events once fully assembled
      const toolCallAccumulator = new Map()
      let finalResponse = null
      let totalInputTokens = 0
      let totalOutputTokens = 0

      for await (const chunk of streamResult.stream) {
        const candidate = chunk.candidates?.[0]
        if (!candidate?.content?.parts) continue

        for (const part of candidate.content.parts) {
          if (part.text) {
            yield { type: 'text', data: part.text }
          } else if (part.functionCall) {
            // Buffer tool calls — Vertex may split them across chunks
            const key = part.functionCall.name
            const existing = toolCallAccumulator.get(key)
            if (!existing) {
              toolCallAccumulator.set(key, {
                id: `fc_${randomUUID()}`,
                name: part.functionCall.name,
                args: part.functionCall.args ?? {},
              })
            } else {
              // Merge args if split across chunks
              existing.args = { ...existing.args, ...(part.functionCall.args ?? {}) }
            }
          }
        }

        // Accumulate token usage across chunks
        const usage = chunk.usageMetadata ?? {}
        totalInputTokens = Math.max(totalInputTokens, usage.promptTokenCount ?? 0)
        totalOutputTokens += usage.candidatesTokenCount ?? 0
      }

      // Emit buffered tool_use events
      for (const [, tc] of toolCallAccumulator) {
        yield {
          type: 'tool_use',
          data: {
            type: 'tool_use',
            id: tc.id,
            name: tc.name,
            input: tc.args,
          },
        }
      }

      // Retrieve the final aggregated response for the done event
      const aggregated = await streamResult.response
      const finalCandidate = aggregated.candidates?.[0]
      const content = vertexPartsToFrameworkBlocks(finalCandidate?.content?.parts)
      const aggUsage = aggregated.usageMetadata ?? {}

      /** @type {import('@jackchen_me/open-multi-agent').LLMResponse} */
      finalResponse = {
        id: `vertex-${randomUUID()}`,
        content,
        model: options.model,
        stop_reason: mapFinishReason(finalCandidate?.finishReason),
        usage: {
          input_tokens: aggUsage.promptTokenCount ?? totalInputTokens,
          output_tokens: aggUsage.candidatesTokenCount ?? totalOutputTokens,
        },
      }

      yield { type: 'done', data: finalResponse }
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      yield { type: 'error', data: error }
    }
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  /**
   * Build a GenerativeModel instance with the correct config for this call.
   * @private
   */
  #getGenerativeModel(options) {
    return this.#vertexAI.getGenerativeModel({
      model: options.model,
      generationConfig: {
        maxOutputTokens: options.maxTokens ?? 8192,
        temperature: options.temperature ?? 0.7,
      },
      safetySettings: SAFETY_SETTINGS,
      tools: frameworkToolsToVertexTools(options.tools),
    })
  }

  /**
   * Split the messages array into Vertex AI "history" (all but the last user turn)
   * and the final user message parts (what we send to sendMessage / sendMessageStream).
   *
   * Vertex AI's chat API takes prior history separately from the current turn.
   * @private
   */
  #splitMessages(messages) {
    if (messages.length === 0) {
      throw new Error('[VertexAdapter] Cannot call chat/stream with an empty messages array')
    }

    const allAsVertex = frameworkMessagesToVertexContents(messages)

    // The last message should be a user turn (the current prompt)
    const lastContent = allAsVertex[allAsVertex.length - 1]
    const history = allAsVertex.slice(0, -1)

    return {
      history,
      lastUserParts: lastContent.parts,
    }
  }
}
