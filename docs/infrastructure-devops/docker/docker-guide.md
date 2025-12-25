# Docker Guide

## What is Docker?

Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring it runs consistently across different environments.

## Prerequisites

- Docker Desktop installed (includes Docker Engine and Docker CLI)
- Basic understanding of command line
- Understanding of applications and dependencies

## Installation

### Linux
```bash
# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Verify installation
docker --version

# Add user to docker group (to run without sudo)
sudo usermod -aG docker $USER
```

### macOS/Windows
Download and install Docker Desktop from: https://www.docker.com/products/docker-desktop

## Core Concepts

### 1. **Images**
Read-only templates used to create containers. Think of them as blueprints.

### 2. **Containers**
Running instances of images. Lightweight and isolated environments.

### 3. **Dockerfile**
A text file containing instructions to build a Docker image.

### 4. **Docker Hub**
A registry for storing and sharing Docker images.

### 5. **Volumes**
Persistent data storage for containers.

### 6. **Networks**
Enable communication between containers.

## Basic Docker Commands

### Working with Images

```bash
# Pull an image from Docker Hub
docker pull nginx:latest

# List all images
docker images

# Build an image from Dockerfile
docker build -t myapp:1.0 .

# Build with custom Dockerfile name
docker build -t myapp:1.0 -f Dockerfile.prod .

# Remove an image
docker rmi <image-id>

# Remove all unused images
docker image prune -a

# Tag an image
docker tag myapp:1.0 myusername/myapp:1.0

# Push image to Docker Hub
docker push myusername/myapp:1.0

# View image history
docker history <image-name>

# Inspect image details
docker inspect <image-name>
```

### Working with Containers

```bash
# Run a container
docker run nginx

# Run container in detached mode
docker run -d nginx

# Run with custom name
docker run -d --name my-nginx nginx

# Run with port mapping
docker run -d -p 8080:80 nginx

# Run with environment variables
docker run -d -e MY_VAR=value nginx

# Run with volume mount
docker run -d -v /host/path:/container/path nginx

# Run interactively
docker run -it ubuntu bash

# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Stop a container
docker stop <container-id>

# Start a stopped container
docker start <container-id>

# Restart a container
docker restart <container-id>

# Remove a container
docker rm <container-id>

# Remove all stopped containers
docker container prune

# Force remove a running container
docker rm -f <container-id>

# View container logs
docker logs <container-id>

# Follow logs in real-time
docker logs -f <container-id>

# Execute command in running container
docker exec -it <container-id> bash

# View container resource usage
docker stats

# Inspect container details
docker inspect <container-id>

# Copy files from container to host
docker cp <container-id>:/path/to/file /host/path

# Copy files from host to container
docker cp /host/path <container-id>:/path/to/file
```

## Dockerfile

### Basic Dockerfile Structure

```dockerfile
# Base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Expose port
EXPOSE 3000

# Set environment variable
ENV NODE_ENV=production

# Define command to run
CMD ["node", "app.js"]
```

### Dockerfile Instructions

```dockerfile
# FROM - Set base image
FROM ubuntu:22.04

# LABEL - Add metadata
LABEL maintainer="your-email@example.com"
LABEL version="1.0"

# RUN - Execute commands during build
RUN apt-get update && apt-get install -y python3

# COPY - Copy files from host to image
COPY ./src /app/src

# ADD - Similar to COPY but can extract archives and download URLs
ADD archive.tar.gz /app/

# WORKDIR - Set working directory
WORKDIR /app

# ENV - Set environment variables
ENV API_URL=https://api.example.com
ENV PORT=3000

# EXPOSE - Document which ports the container listens on
EXPOSE 3000

# USER - Set user for subsequent commands
USER appuser

# VOLUME - Create mount point for volumes
VOLUME ["/data"]

# CMD - Default command (can be overridden)
CMD ["python3", "app.py"]

# ENTRYPOINT - Command that always runs (harder to override)
ENTRYPOINT ["python3"]
CMD ["app.py"]

# ARG - Build-time variables
ARG VERSION=1.0
RUN echo "Building version $VERSION"

# HEALTHCHECK - Check container health
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost/ || exit 1
```

### Multi-Stage Builds

```dockerfile
# Build stage
FROM node:18 AS builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
EXPOSE 3000
CMD ["node", "dist/app.js"]
```

## Docker Compose

Docker Compose is a tool for defining and running multi-container applications.

### docker-compose.yml Example

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://db:5432/myapp
    depends_on:
      - db
      - redis
    volumes:
      - ./src:/app/src
    networks:
      - app-network
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - app-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - app-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - web
    networks:
      - app-network

volumes:
  postgres-data:

networks:
  app-network:
    driver: bridge
```

### Docker Compose Commands

```bash
# Start services
docker-compose up

# Start in detached mode
docker-compose up -d

# Build and start
docker-compose up --build

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f web

# List running services
docker-compose ps

# Execute command in service
docker-compose exec web bash

# Scale a service
docker-compose up -d --scale web=3

# Restart services
docker-compose restart

# Build services
docker-compose build

# Pull images
docker-compose pull

# Validate compose file
docker-compose config
```

## Docker Volumes

### Volume Commands

```bash
# Create a volume
docker volume create my-volume

# List volumes
docker volume ls

# Inspect volume
docker volume inspect my-volume

# Remove volume
docker volume rm my-volume

# Remove all unused volumes
docker volume prune

# Use volume in container
docker run -d -v my-volume:/data nginx
```

### Volume Types

```bash
# Named volume
docker run -d -v my-data:/app/data nginx

# Bind mount (host directory)
docker run -d -v /host/path:/container/path nginx

# Anonymous volume
docker run -d -v /container/path nginx

# Read-only volume
docker run -d -v my-data:/app/data:ro nginx
```

## Docker Networks

### Network Commands

```bash
# List networks
docker network ls

# Create network
docker network create my-network

# Create network with custom driver
docker network create --driver bridge my-bridge

# Inspect network
docker network inspect my-network

# Connect container to network
docker network connect my-network my-container

# Disconnect container from network
docker network disconnect my-network my-container

# Remove network
docker network rm my-network

# Remove all unused networks
docker network prune
```

### Network Types

```bash
# Bridge (default) - containers on same host
docker network create --driver bridge my-bridge

# Host - use host's network
docker run --network host nginx

# None - no networking
docker run --network none nginx

# Custom bridge with subnet
docker network create --subnet=172.18.0.0/16 my-custom-network
```

## Common Use Cases

### Node.js Application

```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3000

USER node

CMD ["node", "server.js"]
```

### Python Application

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

### Go Application

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder
WORKDIR /app
COPY go.* ./
RUN go mod download
COPY . .
RUN go build -o main .

# Run stage
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/main .
EXPOSE 8080
CMD ["./main"]
```

### Static Website with Nginx

```dockerfile
FROM nginx:alpine
COPY ./html /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

## Best Practices

### 1. Use Official Base Images
```dockerfile
FROM node:18-alpine  # Official and minimal
```

### 2. Minimize Layers
```dockerfile
# Bad
RUN apt-get update
RUN apt-get install -y python3
RUN apt-get install -y curl

# Good
RUN apt-get update && apt-get install -y \
    python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*
```

### 3. Use .dockerignore
```
node_modules
.git
.env
*.log
.DS_Store
```

### 4. Don't Run as Root
```dockerfile
RUN addgroup -g 1001 -S appuser && \
    adduser -u 1001 -S appuser -G appuser
USER appuser
```

### 5. Use Multi-Stage Builds
Keep final images small by using multi-stage builds.

### 6. Order Dockerfile Instructions
Put instructions that change less frequently first to leverage caching.

```dockerfile
# Good order
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./    # Changes less frequently
RUN npm install
COPY . .                  # Changes more frequently
```

### 7. Use Specific Tags
```dockerfile
FROM node:18.17.1-alpine  # Specific version
# Not FROM node:latest
```

### 8. Scan for Vulnerabilities
```bash
docker scan myapp:latest
```

## Debugging

### View Container Logs
```bash
docker logs <container-id>
docker logs -f --tail 100 <container-id>
```

### Execute Shell in Container
```bash
docker exec -it <container-id> /bin/sh
docker exec -it <container-id> bash
```

### Inspect Container
```bash
docker inspect <container-id>
```

### View Running Processes
```bash
docker top <container-id>
```

### View Port Mappings
```bash
docker port <container-id>
```

### Export Container Filesystem
```bash
docker export <container-id> > container.tar
```

### Check Container Resource Usage
```bash
docker stats
docker stats <container-id>
```

## Registry Operations

### Docker Hub

```bash
# Login
docker login

# Tag image
docker tag myapp:latest username/myapp:latest

# Push image
docker push username/myapp:latest

# Pull image
docker pull username/myapp:latest

# Logout
docker logout
```

### Private Registry

```bash
# Login to private registry
docker login registry.example.com

# Tag for private registry
docker tag myapp:latest registry.example.com/myapp:latest

# Push to private registry
docker push registry.example.com/myapp:latest
```

## Cleanup Commands

```bash
# Remove all stopped containers
docker container prune

# Remove all unused images
docker image prune -a

# Remove all unused volumes
docker volume prune

# Remove all unused networks
docker network prune

# Remove all unused data
docker system prune -a --volumes

# Show disk usage
docker system df
```

## Health Checks

### In Dockerfile
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

### In Docker Compose
```yaml
services:
  web:
    image: myapp
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 5s
```

## Security

### Scan Images
```bash
docker scan myapp:latest
```

### Run as Non-Root
```dockerfile
USER node
```

### Use Secrets
```bash
# Create secret
echo "my_secret_password" | docker secret create db_password -

# Use in service
docker service create --secret db_password myapp
```

### Limit Resources
```bash
docker run -d \
  --memory="512m" \
  --cpus="1.0" \
  nginx
```

## Real-World Examples

### Example 1: Full-Stack MERN Application

**Directory Structure:**
```
mern-app/
├── docker-compose.yml
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
├── backend/
│   ├── Dockerfile
│   ├── package.json
│   └── src/
└── nginx/
    └── nginx.conf
```

**frontend/Dockerfile:**
```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**backend/Dockerfile:**
```dockerfile
FROM node:18-alpine

# Create app directory
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy app source
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001 && \
    chown -R nodejs:nodejs /app

USER nodejs

EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD node healthcheck.js || exit 1

CMD ["node", "server.js"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: mern-frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - mern-network
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: mern-backend
    ports:
      - "5000:5000"
    environment:
      - NODE_ENV=production
      - MONGO_URI=mongodb://mongo:27017/merndb
      - JWT_SECRET=${JWT_SECRET}
      - REDIS_URL=redis://redis:6379
    depends_on:
      mongo:
        condition: service_healthy
      redis:
        condition: service_started
    networks:
      - mern-network
    restart: unless-stopped
    volumes:
      - ./backend/logs:/app/logs

  mongo:
    image: mongo:6
    container_name: mern-mongo
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: ${MONGO_PASSWORD}
      MONGO_INITDB_DATABASE: merndb
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
      - ./mongo-init.js:/docker-entrypoint-initdb.d/mongo-init.js:ro
    networks:
      - mern-network
    restart: unless-stopped
    healthcheck:
      test: echo 'db.runCommand("ping").ok' | mongosh localhost:27017/test --quiet
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: mern-redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - mern-network
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: mern-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - frontend
      - backend
    networks:
      - mern-network
    restart: unless-stopped

volumes:
  mongo-data:
  redis-data:

networks:
  mern-network:
    driver: bridge
```

**.env:**
```bash
MONGO_PASSWORD=securepassword123
REDIS_PASSWORD=redispassword456
JWT_SECRET=your-jwt-secret-key
```

**Running:**
```bash
# Build and start all services
docker-compose up --build -d

# View logs
docker-compose logs -f

# Scale backend
docker-compose up -d --scale backend=3

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Example 2: Microservices with API Gateway

**docker-compose.microservices.yml:**
```yaml
version: '3.8'

services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "8080:8080"
    environment:
      - USER_SERVICE_URL=http://user-service:3001
      - ORDER_SERVICE_URL=http://order-service:3002
      - PRODUCT_SERVICE_URL=http://product-service:3003
    depends_on:
      - user-service
      - order-service
      - product-service
    networks:
      - microservices

  user-service:
    build: ./services/user-service
    expose:
      - "3001"
    environment:
      - DATABASE_URL=postgresql://postgres:5432/users
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - microservices
    deploy:
      replicas: 2

  order-service:
    build: ./services/order-service
    expose:
      - "3002"
    environment:
      - DATABASE_URL=postgresql://postgres:5432/orders
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - postgres
      - rabbitmq
    networks:
      - microservices
    deploy:
      replicas: 2

  product-service:
    build: ./services/product-service
    expose:
      - "3003"
    environment:
      - DATABASE_URL=mongodb://mongo:27017/products
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - mongo
      - elasticsearch
    networks:
      - microservices

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: microservices
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    networks:
      - microservices

  mongo:
    image: mongo:6
    volumes:
      - mongo-data:/data/db
    networks:
      - microservices

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis-data:/data
    networks:
      - microservices

  rabbitmq:
    image: rabbitmq:3-management-alpine
    ports:
      - "15672:15672"  # Management UI
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD}
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq
    networks:
      - microservices

  elasticsearch:
    image: elasticsearch:8.10.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
      - xpack.security.enabled=false
    volumes:
      - elasticsearch-data:/usr/share/elasticsearch/data
    networks:
      - microservices

  kibana:
    image: kibana:8.10.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_URL: http://elasticsearch:9200
    depends_on:
      - elasticsearch
    networks:
      - microservices

volumes:
  postgres-data:
  mongo-data:
  redis-data:
  rabbitmq-data:
  elasticsearch-data:

networks:
  microservices:
    driver: bridge
```

### Example 3: Development Environment with Hot Reload

**docker-compose.dev.yml:**
```yaml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.dev
    command: npm run dev
    volumes:
      - .:/app
      - /app/node_modules  # Anonymous volume for node_modules
    ports:
      - "3000:3000"
      - "9229:9229"  # Node.js debugger
    environment:
      - NODE_ENV=development
      - DATABASE_URL=postgresql://postgres:password@db:5432/devdb
      - REDIS_URL=redis://redis:6379
      - CHOKIDAR_USEPOLLING=true  # For hot reload on Windows/Mac
    depends_on:
      - db
      - redis
    networks:
      - dev-network
    stdin_open: true
    tty: true

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: devdb
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - dev-db-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - dev-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - dev-network

  mailhog:  # Email testing
    image: mailhog/mailhog:latest
    ports:
      - "1025:1025"  # SMTP
      - "8025:8025"  # Web UI
    networks:
      - dev-network

  adminer:  # Database admin UI
    image: adminer:latest
    ports:
      - "8080:8080"
    networks:
      - dev-network

volumes:
  dev-db-data:

networks:
  dev-network:
    driver: bridge
```

**Dockerfile.dev:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install nodemon for hot reload
RUN npm install -g nodemon

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev)
RUN npm install

# Copy app
COPY . .

EXPOSE 3000 9229

CMD ["npm", "run", "dev"]
```

### Example 4: Multi-Architecture Build

**Dockerfile.multiarch:**
```dockerfile
# Syntax for BuildKit
# syntax=docker/dockerfile:1

FROM --platform=$BUILDPLATFORM node:18-alpine AS builder

# Build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG TARGETOS
ARG TARGETARCH

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Production image
FROM node:18-alpine

WORKDIR /app

# Install only production dependencies
COPY package*.json ./
RUN npm ci --only=production

# Copy built files from builder
COPY --from=builder /app/dist ./dist

EXPOSE 3000

USER node

CMD ["node", "dist/server.js"]
```

**Build for multiple platforms:**
```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Create builder instance
docker buildx create --name multiarch-builder --use

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -t username/myapp:latest \
  --push \
  .

# Build and load for current platform
docker buildx build \
  --platform linux/amd64 \
  -t myapp:latest \
  --load \
  .
```

### Example 5: CI/CD Pipeline with Docker

**.gitlab-ci.yml:**
```yaml
stages:
  - build
  - test
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $IMAGE_TAG .
    - docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest
    - docker push $IMAGE_TAG
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

test:
  stage: test
  image: $IMAGE_TAG
  script:
    - npm run test
    - npm run lint
  coverage: '/Statements\s*:\s*([^%]+)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage/cobertura-coverage.xml

deploy:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - |
      curl -X POST $WEBHOOK_URL \
        -H "Content-Type: application/json" \
        -d "{\"image\":\"$IMAGE_TAG\"}"
  only:
    - main
  when: manual
```

### Example 6: Docker with Secrets Management

**docker-compose.secrets.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    secrets:
      - db_password
      - api_key
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
      - API_KEY_FILE=/run/secrets/api_key
    depends_on:
      - db

  db:
    image: postgres:15-alpine
    secrets:
      - db_password
    environment:
      - POSTGRES_PASSWORD_FILE=/run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

**Using secrets in application (Node.js):**
```javascript
const fs = require('fs');

function getSecret(secretName) {
  const secretPath = process.env[`${secretName.toUpperCase()}_FILE`];
  if (secretPath && fs.existsSync(secretPath)) {
    return fs.readFileSync(secretPath, 'utf8').trim();
  }
  return process.env[secretName.toUpperCase()];
}

const dbPassword = getSecret('db_password');
const apiKey = getSecret('api_key');
```

### Example 7: Docker Health Checks

**Dockerfile with comprehensive health check:**
```dockerfile
FROM node:18-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY . .

# Install curl for health check
RUN apk add --no-cache curl

EXPOSE 3000

# Comprehensive health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node healthcheck.js || exit 1

CMD ["node", "server.js"]
```

**healthcheck.js:**
```javascript
const http = require('http');

const options = {
  host: 'localhost',
  port: 3000,
  path: '/health',
  timeout: 2000
};

const request = http.request(options, (res) => {
  console.log(`STATUS: ${res.statusCode}`);
  if (res.statusCode === 200) {
    process.exit(0);
  } else {
    process.exit(1);
  }
});

request.on('error', (err) => {
  console.error('ERROR:', err.message);
  process.exit(1);
});

request.end();
```

### Example 8: Docker Network Isolation

**docker-compose.network.yml:**
```yaml
version: '3.8'

services:
  # Public-facing services
  nginx:
    image: nginx:alpine
    networks:
      - frontend
    ports:
      - "80:80"

  web:
    build: ./web
    networks:
      - frontend
      - backend
    depends_on:
      - api

  # Internal services (not exposed)
  api:
    build: ./api
    networks:
      - backend
      - database
    depends_on:
      - db
      - cache

  db:
    image: postgres:15-alpine
    networks:
      - database
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  cache:
    image: redis:7-alpine
    networks:
      - backend

networks:
  frontend:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
  
  backend:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.21.0.0/16
  
  database:
    driver: bridge
    internal: true  # No external access
    ipam:
      config:
        - subnet: 172.22.0.0/16
```

### Example 9: Resource Limits and Monitoring

**docker-compose.resources.yml:**
```yaml
version: '3.8'

services:
  app:
    build: .
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
        reservations:
          cpus: '0.25'
          memory: 256M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  db:
    image: postgres:15-alpine
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:
```

### Example 10: Database Backup and Restore

**Backup script (backup.sh):**
```bash
#!/bin/bash

# Configuration
CONTAINER_NAME="postgres"
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="backup_${TIMESTAMP}.sql.gz"

# Create backup
docker exec -t $CONTAINER_NAME pg_dumpall -c -U postgres | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete

echo "Backup completed: ${BACKUP_FILE}"
```

**Restore script (restore.sh):**
```bash
#!/bin/bash

CONTAINER_NAME="postgres"
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
  echo "Usage: $0 <backup_file>"
  exit 1
fi

# Restore backup
gunzip < $BACKUP_FILE | docker exec -i $CONTAINER_NAME psql -U postgres

echo "Restore completed"
```

**docker-compose.backup.yml:**
```yaml
version: '3.8'

services:
  db:
    image: postgres:15-alpine
    volumes:
      - db-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}

  backup:
    image: postgres:15-alpine
    depends_on:
      - db
    volumes:
      - ./backups:/backups
      - ./backup.sh:/backup.sh
    environment:
      POSTGRES_HOST: db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    entrypoint: /bin/sh -c "while true; do /backup.sh && sleep 86400; done"

volumes:
  db-data:
```

## Production Deployment Patterns

### Blue-Green Deployment

```bash
# Start green environment
docker-compose -f docker-compose.green.yml up -d

# Test green environment
curl http://green.example.com/health

# Switch traffic (update load balancer)
# Once verified, stop blue environment
docker-compose -f docker-compose.blue.yml down

# Green becomes the new blue
```

### Rolling Update

```bash
# Scale up new version
docker-compose up -d --scale web=4

# Wait for health checks
docker-compose ps

# Scale down old version
docker-compose up -d --scale web=2

# Complete rollout
docker-compose up -d --scale web=3
```

### Canary Deployment

**docker-compose.canary.yml:**
```yaml
version: '3.8'

services:
  app-stable:
    build: .
    image: myapp:stable
    deploy:
      replicas: 9
    labels:
      - "traefik.enable=true"
      - "traefik.weight=90"

  app-canary:
    build: .
    image: myapp:canary
    deploy:
      replicas: 1
    labels:
      - "traefik.enable=true"
      - "traefik.weight=10"

  traefik:
    image: traefik:latest
    ports:
      - "80:80"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
```

## Troubleshooting Guide

### Common Issues

1. **Container exits immediately:**
```bash
# Check logs
docker logs <container-id>

# Check exit code
docker ps -a

# Run interactively to debug
docker run -it <image> /bin/sh
```

2. **Port already in use:**
```bash
# Find process using port
lsof -i :8080

# Kill process or change port mapping
docker run -p 8081:80 nginx
```

3. **Permission denied:**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .

# Or run as current user
docker run --user $(id -u):$(id -g) <image>
```

4. **Out of disk space:**
```bash
# Check disk usage
docker system df

# Cleanup
docker system prune -a --volumes

# Remove specific items
docker container prune
docker image prune -a
docker volume prune
```

5. **Networking issues:**
```bash
# Inspect network
docker network inspect bridge

# Check DNS
docker exec <container> nslookup google.com

# Recreate network
docker-compose down
docker-compose up -d
```

6. **Performance issues:**
```bash
# Check resource usage
docker stats

# Limit resources
docker run --memory="512m" --cpus="0.5" <image>

# Check for multiple layers
docker history <image>
```

7. **Build cache issues:**
```bash
# Build without cache
docker build --no-cache -t myapp .

# Or for compose
docker-compose build --no-cache
```

## Useful Resources

- Official Documentation: https://docs.docker.com/
- Docker Hub: https://hub.docker.com/
- Docker Compose Documentation: https://docs.docker.com/compose/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/
- Docker Security: https://docs.docker.com/engine/security/
- BuildKit: https://docs.docker.com/develop/develop-images/build_enhancements/

## Quick Reference

| Command | Description |
|---------|-------------|
| `docker run <image>` | Create and start container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker stop <id>` | Stop container |
| `docker rm <id>` | Remove container |
| `docker images` | List images |
| `docker rmi <id>` | Remove image |
| `docker build -t <name> .` | Build image |
| `docker logs <id>` | View logs |
| `docker exec -it <id> bash` | Execute command in container |
| `docker-compose up -d` | Start services |
| `docker-compose down` | Stop services |

---

*This guide covers Docker fundamentals. For production deployments, consider additional topics like orchestration (Kubernetes, Swarm), security scanning, and monitoring solutions.*
