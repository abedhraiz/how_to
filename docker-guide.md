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

## Useful Resources

- Official Documentation: https://docs.docker.com/
- Docker Hub: https://hub.docker.com/
- Docker Compose Documentation: https://docs.docker.com/compose/
- Best Practices: https://docs.docker.com/develop/dev-best-practices/

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
