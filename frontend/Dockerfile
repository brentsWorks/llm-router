FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . ./

# Set environment variable for production build
ENV VITE_API_BASE_URL=https://llm-router-production-bce4.up.railway.app

# Build the application
RUN npm run build

# Debug: List build output
RUN ls -la /app/dist

# Production stage
FROM nginx:alpine

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Debug: Verify files copied
RUN ls -la /usr/share/nginx/html

# Create minimal nginx config for React SPA
RUN echo 'server { \
    listen 80; \
    server_name _; \
    root /usr/share/nginx/html; \
    index index.html; \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf.default 2>/dev/null || true

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]