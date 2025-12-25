# Kubernetes with Docker in Production Guide

## Overview

This guide covers best practices for building, deploying, and managing Docker containers in production Kubernetes environments. It focuses on real-world production scenarios, security hardening, CI/CD integration, and operational excellence.

## Prerequisites

- Docker installed and configured
- Kubernetes cluster (EKS, GKE, AKS, or self-managed)
- kubectl configured
- Basic understanding of Docker and Kubernetes
- Container registry access (Docker Hub, ECR, GCR, ACR)

## Production-Ready Dockerfile

### Multi-Stage Build Example

```dockerfile
# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Production stage
FROM node:18-alpine AS production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

# Set working directory
WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/package*.json ./

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node healthcheck.js || exit 1

# Start application
CMD ["node", "dist/server.js"]
```

### Python Production Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser . .

# Update PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]
```

### Go Production Dockerfile

```dockerfile
# Build stage
FROM golang:1.21-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main .

# Production stage
FROM alpine:latest

# Install CA certificates
RUN apk --no-cache add ca-certificates

# Create non-root user
RUN addgroup -g 1001 -S appuser && \
    adduser -S appuser -u 1001 -G appuser

WORKDIR /app

# Copy binary from builder
COPY --from=builder --chown=appuser:appuser /app/main .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Run application
CMD ["./main"]
```

### Dockerfile Best Practices

```dockerfile
# Best practices summary

# 1. Use specific tags, not 'latest'
FROM node:18.17.0-alpine

# 2. Use multi-stage builds to reduce image size
FROM builder AS production

# 3. Minimize layers
RUN apt-get update && \
    apt-get install -y package1 package2 && \
    rm -rf /var/lib/apt/lists/*

# 4. Don't run as root
USER nodejs

# 5. Use .dockerignore
# Create .dockerignore with:
# node_modules
# .git
# .env
# *.md

# 6. Copy only what's needed
COPY package*.json ./
RUN npm ci --only=production
COPY dist ./dist

# 7. Set proper labels
LABEL maintainer="team@example.com"
LABEL version="1.0.0"
LABEL description="Production application"

# 8. Use HEALTHCHECK
HEALTHCHECK CMD curl --fail http://localhost:8080/health || exit 1

# 9. Optimize caching
# Copy dependency files first, then source code

# 10. Use specific WORKDIR
WORKDIR /app
```

## Building and Tagging Docker Images

### Build Strategy

```bash
# Set variables
APP_NAME="myapp"
VERSION="1.0.0"
REGISTRY="docker.io/username"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD)

# Build with multiple tags
docker build \
  --build-arg BUILD_DATE=${BUILD_DATE} \
  --build-arg GIT_COMMIT=${GIT_COMMIT} \
  -t ${REGISTRY}/${APP_NAME}:${VERSION} \
  -t ${REGISTRY}/${APP_NAME}:${GIT_COMMIT} \
  -t ${REGISTRY}/${APP_NAME}:latest \
  .

# Build for multiple platforms (ARM/AMD)
docker buildx create --use
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ${REGISTRY}/${APP_NAME}:${VERSION} \
  --push \
  .

# Scan for vulnerabilities
docker scan ${REGISTRY}/${APP_NAME}:${VERSION}

# Or use Trivy
trivy image ${REGISTRY}/${APP_NAME}:${VERSION}
```

### Build Script

```bash
#!/bin/bash
# build-and-push.sh

set -e

# Configuration
REGISTRY="${DOCKER_REGISTRY:-docker.io/username}"
APP_NAME="${1:-myapp}"
VERSION="${2:-$(git describe --tags --always)}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building ${APP_NAME}:${VERSION}${NC}"

# Build image
docker build \
  --build-arg VERSION=${VERSION} \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
  -t ${REGISTRY}/${APP_NAME}:${VERSION} \
  -t ${REGISTRY}/${APP_NAME}:latest \
  .

echo -e "${GREEN}Build completed successfully${NC}"

# Run security scan
echo -e "${YELLOW}Running security scan...${NC}"
if ! trivy image --severity HIGH,CRITICAL ${REGISTRY}/${APP_NAME}:${VERSION}; then
  echo -e "${RED}Security vulnerabilities found!${NC}"
  exit 1
fi

echo -e "${GREEN}Security scan passed${NC}"

# Push to registry
echo -e "${YELLOW}Pushing to registry...${NC}"
docker push ${REGISTRY}/${APP_NAME}:${VERSION}
docker push ${REGISTRY}/${APP_NAME}:latest

echo -e "${GREEN}Image pushed successfully${NC}"
echo "Image: ${REGISTRY}/${APP_NAME}:${VERSION}"
```

## Container Registry Setup

### AWS ECR

```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name myapp \
  --image-scanning-configuration scanOnPush=true \
  --encryption-configuration encryptionType=AES256

# Get login password
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-west-2.amazonaws.com

# Tag and push
docker tag myapp:1.0.0 123456789012.dkr.ecr.us-west-2.amazonaws.com/myapp:1.0.0
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/myapp:1.0.0

# Create lifecycle policy
cat > lifecycle-policy.json <<EOF
{
  "rules": [
    {
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {
        "tagStatus": "any",
        "countType": "imageCountMoreThan",
        "countNumber": 10
      },
      "action": {
        "type": "expire"
      }
    }
  ]
}
EOF

aws ecr put-lifecycle-policy \
  --repository-name myapp \
  --lifecycle-policy-text file://lifecycle-policy.json
```

### Google Container Registry (GCR)

```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Tag and push
docker tag myapp:1.0.0 gcr.io/project-id/myapp:1.0.0
docker push gcr.io/project-id/myapp:1.0.0

# Enable vulnerability scanning
gcloud container images describe gcr.io/project-id/myapp:1.0.0 \
  --show-package-vulnerability
```

### Azure Container Registry (ACR)

```bash
# Create ACR
az acr create \
  --resource-group myResourceGroup \
  --name myregistry \
  --sku Premium

# Login
az acr login --name myregistry

# Tag and push
docker tag myapp:1.0.0 myregistry.azurecr.io/myapp:1.0.0
docker push myregistry.azurecr.io/myapp:1.0.0

# Enable vulnerability scanning
az acr task create \
  --name quickscan \
  --registry myregistry \
  --image myapp:1.0.0 \
  --cmd 'mcr.microsoft.com/azure-cli az acr task run'
```

### Private Registry with Harbor

```bash
# Deploy Harbor with Helm
helm repo add harbor https://helm.goharbor.io
helm install harbor harbor/harbor \
  --set expose.type=ingress \
  --set expose.ingress.hosts.core=harbor.example.com \
  --set persistence.enabled=true \
  --set harborAdminPassword=admin123

# Login to Harbor
docker login harbor.example.com

# Push image
docker tag myapp:1.0.0 harbor.example.com/library/myapp:1.0.0
docker push harbor.example.com/library/myapp:1.0.0
```

## Kubernetes Image Pull Secrets

### Create Docker Registry Secret

```bash
# For Docker Hub
kubectl create secret docker-registry regcred \
  --docker-server=docker.io \
  --docker-username=myusername \
  --docker-password=mypassword \
  --docker-email=my@email.com

# For ECR
kubectl create secret docker-registry ecr-secret \
  --docker-server=123456789012.dkr.ecr.us-west-2.amazonaws.com \
  --docker-username=AWS \
  --docker-password=$(aws ecr get-login-password --region us-west-2)

# For GCR
kubectl create secret docker-registry gcr-secret \
  --docker-server=gcr.io \
  --docker-username=_json_key \
  --docker-password="$(cat key.json)"

# For ACR
kubectl create secret docker-registry acr-secret \
  --docker-server=myregistry.azurecr.io \
  --docker-username=myusername \
  --docker-password=mypassword
```

### Use in Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      imagePullSecrets:
        - name: regcred
      containers:
        - name: myapp
          image: docker.io/username/myapp:1.0.0
          imagePullPolicy: IfNotPresent
```

## Production Deployment Manifests

### Complete Application Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
  namespace: production
  labels:
    app: myapp
    version: v1.0.0
    environment: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      # Security context
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault
      
      # Image pull secrets
      imagePullSecrets:
        - name: regcred
      
      # Init container
      initContainers:
        - name: init-db
          image: busybox:1.35
          command: ['sh', '-c', 'until nc -z postgres 5432; do echo waiting for db; sleep 2; done']
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
      
      # Main container
      containers:
        - name: myapp
          image: myregistry.azurecr.io/myapp:1.0.0
          imagePullPolicy: IfNotPresent
          
          # Security context
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            runAsNonRoot: true
            runAsUser: 1001
            capabilities:
              drop:
                - ALL
          
          # Ports
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: metrics
              containerPort: 9090
              protocol: TCP
          
          # Environment variables
          env:
            - name: PORT
              value: "8080"
            - name: NODE_ENV
              value: "production"
            - name: DB_HOST
              valueFrom:
                configMapKeyRef:
                  name: myapp-config
                  key: db_host
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: myapp-secrets
                  key: db_password
          
          # Resource limits
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          
          # Liveness probe
          livenessProbe:
            httpGet:
              path: /health/live
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          
          # Readiness probe
          readinessProbe:
            httpGet:
              path: /health/ready
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          
          # Startup probe
          startupProbe:
            httpGet:
              path: /health/startup
              port: http
            initialDelaySeconds: 0
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 30
          
          # Volume mounts
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: cache
              mountPath: /app/cache
            - name: config
              mountPath: /app/config
              readOnly: true
      
      # Volumes
      volumes:
        - name: tmp
          emptyDir: {}
        - name: cache
          emptyDir: {}
        - name: config
          configMap:
            name: myapp-config
      
      # Affinity rules
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - myapp
                topologyKey: kubernetes.io/hostname
      
      # Topology spread
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: myapp
      
      # Termination grace period
      terminationGracePeriodSeconds: 30

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: myapp
  namespace: production
  labels:
    app: myapp
spec:
  type: ClusterIP
  selector:
    app: myapp
  ports:
    - name: http
      port: 80
      targetPort: http
      protocol: TCP
    - name: metrics
      port: 9090
      targetPort: metrics
      protocol: TCP
  sessionAffinity: ClientIP

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
  namespace: production
data:
  db_host: "postgres.production.svc.cluster.local"
  db_port: "5432"
  db_name: "myapp"
  log_level: "info"
  feature_flag_enabled: "true"

---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secrets
  namespace: production
type: Opaque
stringData:
  db_password: "changeme"
  api_key: "secret-api-key"
  jwt_secret: "jwt-secret-key"

---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 2
          periodSeconds: 15
      selectPolicy: Max

---
# pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp
  namespace: production
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: myapp

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp
  namespace: production
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - myapp.example.com
      secretName: myapp-tls
  rules:
    - host: myapp.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: myapp
                port:
                  number: 80
```

## Deployment Strategies

### Blue-Green Deployment

```yaml
# Blue deployment (current)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-blue
  labels:
    app: myapp
    version: blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: blue
  template:
    metadata:
      labels:
        app: myapp
        version: blue
    spec:
      containers:
        - name: myapp
          image: myapp:1.0.0
          ports:
            - containerPort: 8080

---
# Green deployment (new)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-green
  labels:
    app: myapp
    version: green
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
      version: green
  template:
    metadata:
      labels:
        app: myapp
        version: green
    spec:
      containers:
        - name: myapp
          image: myapp:2.0.0
          ports:
            - containerPort: 8080

---
# Service (switch by changing selector)
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp
    version: blue  # Change to 'green' to switch
  ports:
    - port: 80
      targetPort: 8080
```

```bash
# Deploy green
kubectl apply -f deployment-green.yaml

# Test green deployment
kubectl port-forward deployment/myapp-green 8080:8080

# Switch traffic to green
kubectl patch service myapp -p '{"spec":{"selector":{"version":"green"}}}'

# Delete blue deployment
kubectl delete deployment myapp-blue
```

### Canary Deployment

```yaml
# Stable deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-stable
spec:
  replicas: 9
  selector:
    matchLabels:
      app: myapp
      track: stable
  template:
    metadata:
      labels:
        app: myapp
        track: stable
        version: "1.0.0"
    spec:
      containers:
        - name: myapp
          image: myapp:1.0.0

---
# Canary deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp-canary
spec:
  replicas: 1  # 10% traffic
  selector:
    matchLabels:
      app: myapp
      track: canary
  template:
    metadata:
      labels:
        app: myapp
        track: canary
        version: "2.0.0"
    spec:
      containers:
        - name: myapp
          image: myapp:2.0.0

---
# Service (routes to both)
apiVersion: v1
kind: Service
metadata:
  name: myapp
spec:
  selector:
    app: myapp  # Matches both stable and canary
  ports:
    - port: 80
      targetPort: 8080
```

### Argo Rollouts (Progressive Delivery)

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
spec:
  replicas: 10
  strategy:
    canary:
      steps:
        - setWeight: 10
        - pause: {duration: 5m}
        - setWeight: 20
        - pause: {duration: 5m}
        - setWeight: 50
        - pause: {duration: 5m}
        - setWeight: 80
        - pause: {duration: 5m}
      analysis:
        templates:
          - templateName: success-rate
        args:
          - name: service-name
            value: myapp
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:2.0.0
          ports:
            - containerPort: 8080

---
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result >= 0.95
      provider:
        prometheus:
          address: http://prometheus.monitoring:9090
          query: |
            sum(rate(
              http_requests_total{service="{{args.service-name}}",status=~"2.."}[1m]
            )) /
            sum(rate(
              http_requests_total{service="{{args.service-name}}"}[1m]
            ))
```

## Health Checks and Monitoring

### Health Check Endpoints

```python
# Python Flask example
from flask import Flask, jsonify
import psycopg2
import redis

app = Flask(__name__)

@app.route('/health/live')
def liveness():
    """Liveness probe - is the app running?"""
    return jsonify({"status": "ok"}), 200

@app.route('/health/ready')
def readiness():
    """Readiness probe - can the app serve traffic?"""
    try:
        # Check database connection
        conn = psycopg2.connect("dbname=mydb user=postgres")
        conn.close()
        
        # Check Redis connection
        r = redis.Redis(host='redis', port=6379)
        r.ping()
        
        return jsonify({"status": "ready"}), 200
    except Exception as e:
        return jsonify({"status": "not ready", "error": str(e)}), 503

@app.route('/health/startup')
def startup():
    """Startup probe - has the app finished starting?"""
    # Check if initialization is complete
    if app_initialized:
        return jsonify({"status": "started"}), 200
    return jsonify({"status": "starting"}), 503

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest
    return generate_latest()
```

### ServiceMonitor for Prometheus

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp
  namespace: production
  labels:
    app: myapp
spec:
  selector:
    matchLabels:
      app: myapp
  endpoints:
    - port: metrics
      interval: 30s
      path: /metrics
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy to Production

on:
  push:
    branches:
      - main
    tags:
      - 'v*'

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy results to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
      
      - name: Update deployment image
        run: |
          kubectl set image deployment/myapp \
            myapp=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n production
      
      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/myapp -n production --timeout=5m
      
      - name: Verify deployment
        run: |
          kubectl get pods -n production -l app=myapp
          kubectl get deployment myapp -n production
      
      - name: Rollback on failure
        if: failure()
        run: |
          kubectl rollout undo deployment/myapp -n production
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - security
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

before_script:
  - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $IMAGE_TAG .
    - docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest
    - docker push $IMAGE_TAG
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main
    - tags

security-scan:
  stage: security
  image: aquasec/trivy:latest
  script:
    - trivy image --severity HIGH,CRITICAL --exit-code 1 $IMAGE_TAG
  allow_failure: false
  only:
    - main

deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl config use-context production
    - kubectl set image deployment/myapp myapp=$IMAGE_TAG -n production
    - kubectl rollout status deployment/myapp -n production
  environment:
    name: production
    url: https://myapp.example.com
  only:
    - main
  when: manual
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        REGISTRY = 'docker.io'
        IMAGE_NAME = 'username/myapp'
        DOCKER_CREDENTIALS = credentials('docker-hub-credentials')
        KUBECONFIG = credentials('kubeconfig-production')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    def imageTag = "${BUILD_NUMBER}"
                    docker.build("${REGISTRY}/${IMAGE_NAME}:${imageTag}")
                    docker.build("${REGISTRY}/${IMAGE_NAME}:latest")
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                script {
                    sh """
                        trivy image --severity HIGH,CRITICAL \
                        ${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                    """
                }
            }
        }
        
        stage('Push to Registry') {
            steps {
                script {
                    docker.withRegistry("https://${REGISTRY}", 'docker-hub-credentials') {
                        docker.image("${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}").push()
                        docker.image("${REGISTRY}/${IMAGE_NAME}:latest").push()
                    }
                }
            }
        }
        
        stage('Deploy to Kubernetes') {
            steps {
                script {
                    sh """
                        kubectl set image deployment/myapp \
                        myapp=${REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} \
                        -n production
                        
                        kubectl rollout status deployment/myapp -n production
                    """
                }
            }
        }
    }
    
    post {
        failure {
            script {
                sh 'kubectl rollout undo deployment/myapp -n production'
            }
        }
    }
}
```

## Security Best Practices

### Network Policies

```yaml
# Default deny all ingress
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress

---
# Allow ingress from specific namespaces
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-from-ingress-controller
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080

---
# Allow egress to database
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-database-egress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    - to:
        - namespaceSelector:
            matchLabels:
              name: kube-system
        - podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
```

### Pod Security Standards

```yaml
# Pod Security Admission
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted

---
# Security Context at pod level
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1001
    fsGroup: 1001
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: myapp:1.0.0
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        runAsNonRoot: true
        runAsUser: 1001
        capabilities:
          drop:
            - ALL
```

### Secrets Management with External Secrets Operator

```yaml
# Install External Secrets Operator
kubectl apply -f https://raw.githubusercontent.com/external-secrets/external-secrets/main/deploy/crds/bundle.yaml

---
# SecretStore for AWS Secrets Manager
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: aws-secrets
  namespace: production
spec:
  provider:
    aws:
      service: SecretsManager
      region: us-west-2
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets

---
# ExternalSecret
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: myapp-secrets
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets
    kind: SecretStore
  target:
    name: myapp-secrets
    creationPolicy: Owner
  data:
    - secretKey: db_password
      remoteRef:
        key: production/myapp/db_password
    - secretKey: api_key
      remoteRef:
        key: production/myapp/api_key
```

## Logging and Monitoring

### Structured Logging

```python
# Python structured logging
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Use logger
logger.info("Application started", extra={
    "version": "1.0.0",
    "environment": "production"
})
```

### Fluentd DaemonSet

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: kube-system
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      serviceAccountName: fluentd
      containers:
        - name: fluentd
          image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
          env:
            - name: FLUENT_ELASTICSEARCH_HOST
              value: "elasticsearch.logging"
            - name: FLUENT_ELASTICSEARCH_PORT
              value: "9200"
          volumeMounts:
            - name: varlog
              mountPath: /var/log
            - name: varlibdockercontainers
              mountPath: /var/lib/docker/containers
              readOnly: true
      volumes:
        - name: varlog
          hostPath:
            path: /var/log
        - name: varlibdockercontainers
          hostPath:
            path: /var/lib/docker/containers
```

## Troubleshooting Production Issues

### Debug Container

```bash
# Create debug container
kubectl debug -it myapp-pod-xyz --image=busybox --target=myapp

# Debug with different image
kubectl debug -it myapp-pod-xyz --image=ubuntu --share-processes

# Copy pod for debugging
kubectl debug myapp-pod-xyz -it --copy-to=myapp-debug --container=debug
```

### Common Debugging Commands

```bash
# Check pod status
kubectl get pods -n production
kubectl describe pod myapp-pod-xyz -n production

# View logs
kubectl logs myapp-pod-xyz -n production
kubectl logs myapp-pod-xyz -c container-name -n production --previous
kubectl logs -f myapp-pod-xyz -n production

# Execute commands in container
kubectl exec -it myapp-pod-xyz -n production -- /bin/sh
kubectl exec myapp-pod-xyz -n production -- env

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n production
kubectl top nodes

# Check rollout status
kubectl rollout status deployment/myapp -n production
kubectl rollout history deployment/myapp -n production

# Port forward for local testing
kubectl port-forward deployment/myapp 8080:8080 -n production

# Check service endpoints
kubectl get endpoints myapp -n production
kubectl describe service myapp -n production
```

### Image Pull Issues

```bash
# Check image pull secrets
kubectl get secrets -n production
kubectl describe secret regcred -n production

# Test image pull manually
docker pull myregistry.azurecr.io/myapp:1.0.0

# Check pod events for image pull errors
kubectl describe pod myapp-pod-xyz -n production | grep -A 10 Events

# Refresh ECR credentials (expires every 12 hours)
kubectl create secret docker-registry ecr-secret \
  --docker-server=123456789012.dkr.ecr.us-west-2.amazonaws.com \
  --docker-username=AWS \
  --docker-password=$(aws ecr get-login-password --region us-west-2) \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Cost Optimization

### Resource Requests and Limits

```yaml
# Right-size resources
resources:
  requests:
    memory: "128Mi"  # Start small
    cpu: "100m"      # 0.1 CPU
  limits:
    memory: "256Mi"  # Allow burst
    cpu: "200m"      # Prevent CPU throttling
```

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: myapp-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
      - containerName: myapp
        minAllowed:
          cpu: 100m
          memory: 128Mi
        maxAllowed:
          cpu: 1
          memory: 1Gi
```

### Cluster Autoscaler

```bash
# AWS
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Configure
kubectl -n kube-system annotate deployment.apps/cluster-autoscaler \
  cluster-autoscaler.kubernetes.io/safe-to-evict="false"
```

## Resources

- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **Docker Documentation**: https://docs.docker.com/
- **CNCF Landscape**: https://landscape.cncf.io/
- **Kubernetes Best Practices**: https://kubernetes.io/docs/concepts/configuration/overview/

## Quick Reference

```bash
# Build and push
docker build -t myapp:1.0.0 .
docker tag myapp:1.0.0 registry/myapp:1.0.0
docker push registry/myapp:1.0.0

# Deploy
kubectl apply -f deployment.yaml
kubectl rollout status deployment/myapp

# Scale
kubectl scale deployment myapp --replicas=5

# Update image
kubectl set image deployment/myapp myapp=myapp:2.0.0

# Rollback
kubectl rollout undo deployment/myapp

# Debug
kubectl logs -f deployment/myapp
kubectl exec -it myapp-pod -- /bin/sh
kubectl describe pod myapp-pod
```

---

*This guide covers production-ready Docker and Kubernetes practices for building, deploying, and operating containerized applications at scale.*
