# syntax=docker/dockerfile:1.6
# Optimized multi-stage build for React frontend
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy only package manifests first to maximize layer caching
COPY webapp/frontend/package.json ./
COPY webapp/frontend/package-lock.json ./

# Configure npm for reliability and speed
RUN npm config set fetch-retries 5 \
 && npm config set fetch-retry-factor 2 \
 && npm config set fetch-retry-mintimeout 10000 \
 && npm config set fetch-retry-maxtimeout 60000 \
 && npm config set maxsockets 5

# Install dependencies with BuildKit cache; prefer npm ci if lockfile exists
RUN --mount=type=cache,target=/root/.npm \
    echo "📦 Installing dependencies..." && \
    echo "📋 Files in current directory:" && \
    ls -la && \
    /bin/sh -c 'if [ -f package-lock.json ]; then \
      echo "📦 Using npm ci (lockfile found)" && \
      npm ci --prefer-offline --no-audit --progress=false; \
    else \
      echo "📦 Using npm install (no lockfile found)" && \
      npm install --prefer-offline --no-audit --progress=false; \
    fi' && \
    echo "📦 Installing Chart.js date adapter dependencies..." && \
    npm install chartjs-adapter-date-fns@^3.0.0 date-fns@^3.6.0 --save && \
    echo "📋 Verifying Chart.js dependencies installation:" && \
    npm list chartjs-adapter-date-fns date-fns || echo "⚠️ Chart.js dependencies not found" && \
    echo "📋 Verifying @tailwindcss/forms installation:" && \
    npm list @tailwindcss/forms || echo "⚠️ @tailwindcss/forms not found in node_modules"

# Copy application sources (only what is needed)
COPY webapp/frontend/public ./public
COPY webapp/frontend/src ./src
COPY webapp/frontend/postcss.config.js webapp/frontend/tailwind.config.js ./


# Accept CRA env build args (allow override from build-args)
ARG REACT_APP_API_URL
ENV REACT_APP_API_URL=$REACT_APP_API_URL
ARG REACT_APP_WS_URL
ENV REACT_APP_WS_URL=$REACT_APP_WS_URL

# Optimize CRA build: disable eslint plugin and sourcemaps for speed
ARG DISABLE_ESLINT_PLUGIN=true
ENV DISABLE_ESLINT_PLUGIN=$DISABLE_ESLINT_PLUGIN
ARG GENERATE_SOURCEMAP=false
ENV GENERATE_SOURCEMAP=$GENERATE_SOURCEMAP

# Build the application using persistent webpack cache
RUN --mount=type=cache,target=/app/node_modules/.cache \
    npm run build

# Production stage
FROM nginx:alpine

# Install bash for the injection script
RUN apk add --no-cache bash

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY webapp/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copy the environment injection script
COPY webapp/frontend/inject-env.sh /usr/local/bin/inject-env.sh
RUN chmod +x /usr/local/bin/inject-env.sh

# Create startup script that injects env vars and starts nginx
RUN echo '#!/bin/bash' > /usr/local/bin/start.sh && \
    echo '/usr/local/bin/inject-env.sh' >> /usr/local/bin/start.sh && \
    echo 'nginx -g "daemon off;"' >> /usr/local/bin/start.sh && \
    chmod +x /usr/local/bin/start.sh

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:80/ || exit 1

# Start with our custom script
CMD ["/usr/local/bin/start.sh"]
