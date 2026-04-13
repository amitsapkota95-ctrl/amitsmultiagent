# ─────────────────────────────────────────────────────────────
# Stage 1: dependency installation
# Uses a full node image so native modules (if any) can compile.
# ─────────────────────────────────────────────────────────────
FROM node:20-slim AS deps

WORKDIR /app

# Copy package files first for better layer caching
COPY package.json package-lock.json* ./

# Install production dependencies only
RUN npm ci --omit=dev

# ─────────────────────────────────────────────────────────────
# Stage 2: runtime image
# Copies only the installed node_modules and source files.
# ─────────────────────────────────────────────────────────────
FROM node:20-slim AS runtime

# Install bash (required by the bash built-in tool) and other CLI utilities
# that agents may shell out to
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies from build stage
COPY --from=deps /app/node_modules ./node_modules

# Copy application source
COPY server.js       ./
COPY VertexAdapter.js ./
COPY package.json    ./

# ⚠️  IMPORTANT: The vertex_api_api.json service-account key file must be
# present in this directory before building the image OR passed in at runtime
# via a Cloud Run Secret Mount (recommended for production).
#
# To use Secret Manager:
#   1. Upload vertex_api_api.json as a secret in GCP Secret Manager.
#   2. Mount it at /app/vertex_api_api.json via Cloud Run's --secrets flag.
#
# For development/testing you can COPY it directly (NOT recommended for production):
COPY vertex_api_api.json ./

# Cloud Run injects PORT; default to 8080
ENV PORT=8080
ENV NODE_ENV=production

# Expose the port
EXPOSE 8080

# Non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nodeuser && \
    chown -R nodeuser:nodejs /app

USER nodeuser

# Health check — Cloud Run will poll this before routing traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["node", "server.js"]
