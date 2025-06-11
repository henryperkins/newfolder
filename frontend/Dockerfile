# Build stage – install all deps & build static assets
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
#   1. Copy `package.json` and `package-lock.json` to leverage Docker layer
#      caching so that subsequent builds are faster.
#   2. Run `npm ci` **without** the `--only=production` flag so that the build
#      tool-chain (Vite, TypeScript, Tailwind, …) is available.

COPY package*.json ./
RUN npm ci --silent --progress=false

# Copy application source and build
COPY . .
RUN npm run build

# ------------------------------------------------------------
# **Production stage** – serve static assets with Nginx
# ------------------------------------------------------------
FROM nginx:alpine AS production

# Copy built static files into Nginx html directory
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]