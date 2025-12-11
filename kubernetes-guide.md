# Kubernetes Guide

## What is Kubernetes?

Kubernetes (K8s) is an open-source container orchestration platform that automates the deployment, scaling, and management of containerized applications.

## Prerequisites

- Docker installed
- kubectl CLI tool installed
- Access to a Kubernetes cluster (minikube, Docker Desktop, cloud provider, etc.)

## Core Concepts

### 1. **Pods**
The smallest deployable unit in Kubernetes. A pod can contain one or more containers.

### 2. **Deployments**
Manages the deployment and scaling of a set of pods.

### 3. **Services**
Exposes an application running on a set of pods as a network service.

### 4. **ConfigMaps & Secrets**
Store configuration data and sensitive information respectively.

### 5. **Namespaces**
Virtual clusters within a physical cluster for resource isolation.

## Basic kubectl Commands

### Cluster Information
```bash
# Check cluster status
kubectl cluster-info

# View cluster nodes
kubectl get nodes

# Get cluster version
kubectl version
```

### Working with Pods
```bash
# List all pods
kubectl get pods

# List pods in all namespaces
kubectl get pods --all-namespaces

# Get detailed information about a pod
kubectl describe pod <pod-name>

# View pod logs
kubectl logs <pod-name>

# Execute command in a pod
kubectl exec -it <pod-name> -- /bin/bash
```

### Working with Deployments
```bash
# List deployments
kubectl get deployments

# Create a deployment
kubectl create deployment <name> --image=<image-name>

# Scale a deployment
kubectl scale deployment <name> --replicas=3

# Update deployment image
kubectl set image deployment/<name> <container>=<new-image>

# Delete a deployment
kubectl delete deployment <name>
```

### Working with Services
```bash
# List services
kubectl get services

# Expose a deployment as a service
kubectl expose deployment <name> --port=80 --type=LoadBalancer

# Delete a service
kubectl delete service <name>
```

### Working with Namespaces
```bash
# List namespaces
kubectl get namespaces

# Create a namespace
kubectl create namespace <name>

# Set default namespace
kubectl config set-context --current --namespace=<name>
```

## Creating Resources with YAML

### Example: Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.14.2
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Example: Service YAML
```yaml
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  selector:
    app: nginx
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
  type: LoadBalancer
```

### Example: Complete Application Stack
```yaml
---
# Namespace
apiVersion: v1
kind: Namespace
metadata:
  name: my-app

---
# ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: my-app
data:
  database_url: "postgresql://db:5432/myapp"
  log_level: "info"

---
# Secret
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: my-app
type: Opaque
data:
  db-password: cGFzc3dvcmQxMjM=  # base64 encoded
  api-key: YWJjZGVmZ2hpamtsbW5vcA==

---
# PersistentVolumeClaim
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: my-app
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi

---
# Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web
        image: myapp:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database_url
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: db-password
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: log_level
        volumeMounts:
        - name: data
          mountPath: /data
        resources:
          requests:
            memory: "256Mi"
            cpu: "500m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: data-pvc

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: web-service
  namespace: my-app
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP

---
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
  namespace: my-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
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
```

### Apply YAML Configuration
```bash
# Apply a configuration file
kubectl apply -f deployment.yaml

# Apply all files in a directory
kubectl apply -f ./configs/

# Delete resources from a file
kubectl delete -f deployment.yaml

# Apply with recording (for rollback)
kubectl apply -f deployment.yaml --record

# Dry run to see what would be applied
kubectl apply -f deployment.yaml --dry-run=client

# Show diff before applying
kubectl diff -f deployment.yaml
```

## Persistent Storage

### PersistentVolume (PV)
```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-data
spec:
  capacity:
    storage: 10Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: /mnt/data
```

### PersistentVolumeClaim (PVC)
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard
```

### Using PVC in Pod
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: myapp:1.0
    volumeMounts:
    - name: data-volume
      mountPath: /data
  volumes:
  - name: data-volume
    persistentVolumeClaim:
      claimName: pvc-data
```

## StatefulSets

### StatefulSet for Database
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres-service
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 10Gi
```

## Ingress

### Ingress Resource
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: web-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
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
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8080
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 80
```

### Install Ingress Controller
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Verify installation
kubectl get pods -n ingress-nginx

# Check ingress
kubectl get ingress
kubectl describe ingress web-ingress
```

## Jobs and CronJobs

### Job
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-migration
spec:
  template:
    spec:
      containers:
      - name: migrator
        image: migration-tool:1.0
        command: ["./migrate.sh"]
      restartPolicy: Never
  backoffLimit: 4
```

### CronJob
```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-job
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:1.0
            command: ["./backup.sh"]
            volumeMounts:
            - name: data
              mountPath: /data
          restartPolicy: OnFailure
          volumes:
          - name: data
            persistentVolumeClaim:
              claimName: data-pvc
```

## Network Policies

### Allow Specific Traffic
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: database
    ports:
    - protocol: TCP
      port: 5432
```

### Deny All Traffic
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
```

## RBAC (Role-Based Access Control)

### ServiceAccount
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: production
```

### Role
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "watch", "list"]
```

### RoleBinding
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: production
subjects:
- kind: ServiceAccount
  name: app-service-account
  namespace: production
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### ClusterRole
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-admin-custom
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
```

## Debugging and Troubleshooting

### Check Resource Status
```bash
# Get detailed information
kubectl describe <resource-type> <resource-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check events in specific namespace
kubectl get events -n production --sort-by='.lastTimestamp'

# View resource usage
kubectl top nodes
kubectl top pods
kubectl top pods -n production --sort-by=memory
```

### Logs and Diagnostics
```bash
# View logs
kubectl logs <pod-name>

# Follow logs in real-time
kubectl logs -f <pod-name>

# View logs from a specific container in a pod
kubectl logs <pod-name> -c <container-name>

# View previous container logs (if crashed)
kubectl logs <pod-name> --previous
```

### Port Forwarding
```bash
# Forward a local port to a pod port
kubectl port-forward <pod-name> 8080:80

# Forward to a service
kubectl port-forward service/<service-name> 8080:80
```

## ConfigMaps and Secrets

### ConfigMaps
```bash
# Create from literal values
kubectl create configmap <name> --from-literal=key1=value1

# Create from file
kubectl create configmap <name> --from-file=config.txt

# View configmap
kubectl get configmap <name> -o yaml
```

### Secrets
```bash
# Create a generic secret
kubectl create secret generic <name> --from-literal=password=mypass

# Create from file
kubectl create secret generic <name> --from-file=./secret.txt

# View secret (base64 encoded)
kubectl get secret <name> -o yaml
```

## Resource Management

### Labels and Selectors
```bash
# Add a label
kubectl label pods <pod-name> env=production

# Show labels
kubectl get pods --show-labels

# Filter by label
kubectl get pods -l env=production

# Remove a label
kubectl label pods <pod-name> env-
```

### Resource Quotas
```bash
# View resource quotas
kubectl get resourcequota

# Describe quota
kubectl describe resourcequota <name>
```

## Advanced Operations

### Rolling Updates
```bash
# Update deployment image
kubectl set image deployment/<name> <container>=<new-image>

# Check rollout status
kubectl rollout status deployment/<name>

# View rollout history
kubectl rollout history deployment/<name>

# Rollback to previous version
kubectl rollout undo deployment/<name>

# Rollback to specific revision
kubectl rollout undo deployment/<name> --to-revision=2
```

### Scaling
```bash
# Manual scaling
kubectl scale deployment <name> --replicas=5

# Autoscaling
kubectl autoscale deployment <name> --min=2 --max=10 --cpu-percent=80
```

### Resource Export and Backup
```bash
# Export resource to YAML
kubectl get deployment <name> -o yaml > deployment-backup.yaml

# Export all resources in namespace
kubectl get all -o yaml > namespace-backup.yaml
```

## Multi-Container Pods

### Sidecar Pattern
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-sidecar
spec:
  containers:
  # Main application container
  - name: app
    image: myapp:1.0
    ports:
    - containerPort: 8080
    volumeMounts:
    - name: shared-logs
      mountPath: /var/log
  # Sidecar container for log processing
  - name: log-processor
    image: fluent-bit:1.9
    volumeMounts:
    - name: shared-logs
      mountPath: /var/log
  volumes:
  - name: shared-logs
    emptyDir: {}
```

### Init Containers
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: app-with-init
spec:
  initContainers:
  # Run database migrations before app starts
  - name: migration
    image: migration-tool:1.0
    command: ['sh', '-c', 'migrate.sh']
  # Wait for database to be ready
  - name: wait-for-db
    image: busybox:1.35
    command: ['sh', '-c', 'until nc -z db-service 5432; do sleep 2; done']
  containers:
  - name: app
    image: myapp:1.0
    ports:
    - containerPort: 8080
```

## Helm Package Manager

### Install Helm
```bash
# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installation
helm version
```

### Using Helm Charts
```bash
# Add repository
helm repo add bitnami https://charts.bitnami.com/bitnami

# Update repositories
helm repo update

# Search for charts
helm search repo nginx

# Install a chart
helm install my-nginx bitnami/nginx

# Install with custom values
helm install my-nginx bitnami/nginx --values custom-values.yaml

# List installed releases
helm list

# Upgrade release
helm upgrade my-nginx bitnami/nginx --version 14.0.0

# Rollback release
helm rollback my-nginx 1

# Uninstall release
helm uninstall my-nginx
```

### Create Custom Chart
```bash
# Create new chart
helm create my-app

# Chart structure
my-app/
├── Chart.yaml          # Chart metadata
├── values.yaml         # Default values
├── charts/             # Dependencies
└── templates/          # Kubernetes manifests
    ├── deployment.yaml
    ├── service.yaml
    └── ingress.yaml

# Install local chart
helm install my-release ./my-app

# Package chart
helm package my-app

# Lint chart
helm lint my-app
```

## Monitoring and Observability

### Metrics Server
```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# View resource usage
kubectl top nodes
kubectl top pods -A
```

### Prometheus and Grafana
```bash
# Add Prometheus Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

# Install Prometheus and Grafana
helm install monitoring prometheus-community/kube-prometheus-stack

# Access Grafana
kubectl port-forward svc/monitoring-grafana 3000:80

# Default credentials: admin / prom-operator
```

## Best Practices

### 1. Resource Management
```yaml
# Always set resource requests and limits
resources:
  requests:
    memory: "256Mi"
    cpu: "500m"
  limits:
    memory: "512Mi"
    cpu: "1000m"
```

### 2. Health Checks
```yaml
# Implement both liveness and readiness probes
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 10
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3
```

### 3. Configuration Management
```yaml
# Use ConfigMaps for non-sensitive config
# Use Secrets for sensitive data (base64 encoded)
# Use external secret managers for production
```

### 4. High Availability
```yaml
# Use multiple replicas
replicas: 3

# Use pod anti-affinity to spread across nodes
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
    - labelSelector:
        matchExpressions:
        - key: app
          operator: In
          values:
          - myapp
      topologyKey: kubernetes.io/hostname
```

### 5. Security
```yaml
# Run as non-root user
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
```

### 6. Resource Isolation
- **Use Namespaces** - Organize resources and apply resource quotas
- **Set Resource Limits** - Define CPU and memory limits for containers
- **Use Labels** - Organize and select resources effectively
- **Use Network Policies** - Control traffic between pods

### 7. Operations
- **Version Control** - Store YAML configurations in Git
- **Use Helm** - Package manager for Kubernetes applications
- **Monitor Resources** - Use tools like Prometheus and Grafana
- **Backup Regularly** - Export critical configurations
- **RBAC** - Implement proper role-based access control
- **Auto-scaling** - Use HPA for automatic scaling
- **Blue-Green Deployments** - Zero-downtime deployments

## Deployment Strategies

### Rolling Update
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rolling-update-app
spec:
  replicas: 10
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2        # Max pods above desired count
      maxUnavailable: 1   # Max pods unavailable during update
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: app
        image: myapp:2.0.0
```

```bash
# Perform rolling update
kubectl set image deployment/rolling-update-app app=myapp:2.0.0

# Monitor rollout
kubectl rollout status deployment/rolling-update-app

# Pause rollout
kubectl rollout pause deployment/rolling-update-app

# Resume rollout
kubectl rollout resume deployment/rolling-update-app

# Rollback
kubectl rollout undo deployment/rolling-update-app
```

### Blue-Green Deployment
```yaml
---
# Blue deployment (v1)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-blue
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
      - name: app
        image: myapp:1.0.0

---
# Green deployment (v2)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-green
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
      - name: app
        image: myapp:2.0.0

---
# Service (switch between blue and green)
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp
    version: blue  # Change to 'green' to switch traffic
  ports:
  - port: 80
    targetPort: 8080
```

### Canary Deployment
```yaml
---
# Stable deployment (90% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-stable
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
    spec:
      containers:
      - name: app
        image: myapp:1.0.0

---
# Canary deployment (10% traffic)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app-canary
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
      track: canary
  template:
    metadata:
      labels:
        app: myapp
        track: canary
    spec:
      containers:
      - name: app
        image: myapp:2.0.0

---
# Service (routes to both)
apiVersion: v1
kind: Service
metadata:
  name: app-service
spec:
  selector:
    app: myapp  # Selects both stable and canary
  ports:
  - port: 80
    targetPort: 8080
```

## Complete Microservices Example

### Frontend Service
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: microservices
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: frontend:1.0.0
        ports:
        - containerPort: 80
        env:
        - name: API_URL
          value: "http://api-service:8080"
        resources:
          requests:
            memory: "128Mi"
            cpu: "250m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: frontend-service
  namespace: microservices
spec:
  type: LoadBalancer
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
```

### Backend API Service
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: microservices
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: api:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: microservices
spec:
  selector:
    app: api
  ports:
  - port: 8080
    targetPort: 8080
```

### Database (StatefulSet)
```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: microservices
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: appdb
        - name: POSTGRES_USER
          value: appuser
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: microservices
spec:
  clusterIP: None  # Headless service for StatefulSet
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
```

### Redis Cache
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: microservices
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: microservices
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

## Common Issues and Solutions

### Pod Not Starting
```bash
# Check pod status
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>

# Check previous container logs (if crashed)
kubectl logs <pod-name> --previous

# Check events
kubectl get events --field-selector involvedObject.name=<pod-name>

# Common causes:
# - Image pull errors (wrong image name, auth issues)
# - Resource limits (not enough CPU/memory)
# - Failed health checks
# - Missing secrets/configmaps
```

### Service Not Accessible
```bash
# Check service endpoints
kubectl get endpoints <service-name>
kubectl describe service <service-name>

# Test connectivity from within cluster
kubectl run debug --image=busybox -it --rm -- wget -O- http://<service-name>:80

# Check if pods are ready
kubectl get pods -l app=<app-label>

# Common causes:
# - Wrong selector labels
# - Pods not ready
# - Network policies blocking traffic
# - Wrong port configuration
```

### Resource Constraints
```bash
# Check node resources
kubectl describe nodes
kubectl top nodes
kubectl top pods

# Check pod resource usage
kubectl describe pod <pod-name> | grep -A 5 "Limits\|Requests"

# Check if pods are evicted
kubectl get pods --all-namespaces | grep Evicted

# Common causes:
# - Insufficient cluster resources
# - Resource limits too low
# - Memory leaks
# - No resource requests set
```

### ImagePullBackOff
```bash
# Check image pull status
kubectl describe pod <pod-name> | grep -A 5 "Events"

# Common causes:
# - Wrong image name or tag
# - Private registry without credentials
# - Network issues pulling image

# Solution: Add image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  --docker-email=<email>

# Reference in pod spec:
imagePullSecrets:
- name: regcred
```

### CrashLoopBackOff
```bash
# View logs
kubectl logs <pod-name>
kubectl logs <pod-name> --previous

# Check restart count
kubectl get pod <pod-name> -o jsonpath='{.status.containerStatuses[0].restartCount}'

# Common causes:
# - Application errors
# - Missing environment variables
# - Failed health checks
# - Resource limits too low

# Debug interactively
kubectl run debug --image=<your-image> -it --rm -- /bin/sh
```

### Pending Pods
```bash
# Check why pod is pending
kubectl describe pod <pod-name>

# Common causes:
# - Insufficient cluster resources
# - No nodes matching affinity rules
# - PVC not bound
# - Image pull secrets missing

# Check node capacity
kubectl describe nodes | grep -A 5 "Allocated resources"
```

## Useful Resources

- Official Documentation: https://kubernetes.io/docs/
- Kubernetes GitHub: https://github.com/kubernetes/kubernetes
- kubectl Cheat Sheet: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
- Interactive Tutorial: https://kubernetes.io/docs/tutorials/

## Production Readiness Checklist

### Security
- [ ] Enable RBAC and create appropriate roles
- [ ] Use Network Policies to restrict traffic
- [ ] Store sensitive data in Secrets (encrypted at rest)
- [ ] Run containers as non-root users
- [ ] Use read-only root filesystems where possible
- [ ] Scan images for vulnerabilities
- [ ] Enable pod security policies/admission controllers
- [ ] Use private registries for images

### High Availability
- [ ] Run multiple replicas of stateless applications
- [ ] Use anti-affinity rules to spread pods across nodes
- [ ] Implement proper health checks (liveness and readiness)
- [ ] Set up pod disruption budgets
- [ ] Use StatefulSets for stateful applications
- [ ] Configure horizontal pod autoscaling

### Resource Management
- [ ] Set resource requests and limits for all containers
- [ ] Configure resource quotas for namespaces
- [ ] Monitor resource usage
- [ ] Set up cluster autoscaling
- [ ] Use pod priority classes

### Monitoring & Logging
- [ ] Deploy metrics server
- [ ] Set up Prometheus and Grafana
- [ ] Configure centralized logging (ELK, Loki)
- [ ] Set up alerting rules
- [ ] Monitor cluster health
- [ ] Track application metrics

### Backup & Disaster Recovery
- [ ] Backup etcd regularly
- [ ] Export and version control all YAML files
- [ ] Test restore procedures
- [ ] Document disaster recovery plans
- [ ] Use GitOps for declarative deployments

### CI/CD Integration
- [ ] Automate deployments
- [ ] Implement blue-green or canary deployments
- [ ] Use Helm charts for packaging
- [ ] Set up automated testing
- [ ] Version all container images

## Advanced kubectl Tips

### Useful Aliases
```bash
# Add to ~/.bashrc or ~/.zshrc
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kgd='kubectl get deployments'
alias kd='kubectl describe'
alias kdp='kubectl describe pod'
alias kl='kubectl logs'
alias klf='kubectl logs -f'
alias kex='kubectl exec -it'
alias kaf='kubectl apply -f'
alias kdf='kubectl delete -f'

# With completion
source <(kubectl completion bash)
complete -F __start_kubectl k
```

### Advanced Commands
```bash
# Get pod by label
kubectl get pods -l app=nginx

# Get all resources
kubectl get all -n production

# Watch resources
kubectl get pods -w

# Output as JSON/YAML
kubectl get pod <pod> -o json
kubectl get pod <pod> -o yaml

# Use JSONPath
kubectl get pods -o jsonpath='{.items[*].metadata.name}'

# Sort by creation time
kubectl get pods --sort-by=.metadata.creationTimestamp

# Show labels
kubectl get pods --show-labels

# Multiple namespaces
kubectl get pods -A  # or --all-namespaces

# Explain resource
kubectl explain pod.spec.containers

# Dry run
kubectl run nginx --image=nginx --dry-run=client -o yaml

# Create from stdin
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test
    image: nginx
EOF

# Edit resource
kubectl edit deployment/my-deployment

# Patch resource
kubectl patch deployment my-deployment -p '{"spec":{"replicas":5}}'

# Replace resource
kubectl replace -f deployment.yaml

# Diff before apply
kubectl diff -f deployment.yaml

# Resource shortcuts
kubectl get po    # pods
kubectl get svc   # services
kubectl get deploy # deployments
kubectl get rs    # replicasets
kubectl get ns    # namespaces
kubectl get no    # nodes
kubectl get cm    # configmaps
kubectl get pv    # persistentvolumes
kubectl get pvc   # persistentvolumeclaims
```

### Debugging Techniques
```bash
# Run temporary pod for debugging
kubectl run debug --image=busybox -it --rm -- sh

# Debug with specific image
kubectl run debug --image=nicolaka/netshoot -it --rm -- bash

# Copy files from pod
kubectl cp <pod>:/path/to/file ./local-file

# Copy files to pod
kubectl cp ./local-file <pod>:/path/to/file

# Execute command without shell
kubectl exec <pod> -- ls /app

# Get shell in running pod
kubectl exec -it <pod> -- /bin/bash

# Debug node
kubectl debug node/<node-name> -it --image=ubuntu

# View API resources
kubectl api-resources

# View API versions
kubectl api-versions

# Validate YAML
kubectl apply -f deployment.yaml --dry-run=client --validate=true

# Check which resources use a specific image
kubectl get pods -A -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.containers[*].image}{"\n"}{end}' | grep nginx
```

### Context and Configuration
```bash
# View current context
kubectl config current-context

# List all contexts
kubectl config get-contexts

# Switch context
kubectl config use-context <context-name>

# Set default namespace
kubectl config set-context --current --namespace=production

# View config
kubectl config view

# Create new context
kubectl config set-context dev --namespace=development --cluster=my-cluster --user=dev-user

# Rename context
kubectl config rename-context old-name new-name

# Delete context
kubectl config delete-context <context-name>
```

## Quick Reference

### Essential Commands

| Command | Description |
|---------|-------------|
| `kubectl get pods` | List all pods |
| `kubectl get services` | List all services |
| `kubectl get deployments` | List all deployments |
| `kubectl apply -f file.yaml` | Apply configuration |
| `kubectl delete -f file.yaml` | Delete resources from file |
| `kubectl logs <pod>` | View pod logs |
| `kubectl logs -f <pod>` | Follow pod logs |
| `kubectl exec -it <pod> -- bash` | Shell into pod |
| `kubectl describe <resource> <name>` | Detailed resource info |
| `kubectl port-forward <pod> 8080:80` | Forward port |
| `kubectl scale deployment <name> --replicas=3` | Scale deployment |
| `kubectl rollout restart deployment/<name>` | Restart deployment |
| `kubectl top pods` | View resource usage |
| `kubectl get events` | View cluster events |

### Resource Types (Shortcuts)

| Full Name | Short | Example |
|-----------|-------|---------|
| pods | po | `kubectl get po` |
| services | svc | `kubectl get svc` |
| deployments | deploy | `kubectl get deploy` |
| replicasets | rs | `kubectl get rs` |
| namespaces | ns | `kubectl get ns` |
| nodes | no | `kubectl get no` |
| configmaps | cm | `kubectl get cm` |
| secrets | - | `kubectl get secrets` |
| persistentvolumes | pv | `kubectl get pv` |
| persistentvolumeclaims | pvc | `kubectl get pvc` |
| ingress | ing | `kubectl get ing` |
| statefulsets | sts | `kubectl get sts` |
| daemonsets | ds | `kubectl get ds` |
| jobs | - | `kubectl get jobs` |
| cronjobs | cj | `kubectl get cj` |

### Common Flags

| Flag | Description |
|------|-------------|
| `-n <namespace>` | Specify namespace |
| `-A` or `--all-namespaces` | All namespaces |
| `-o yaml` | Output as YAML |
| `-o json` | Output as JSON |
| `-o wide` | Additional info |
| `--show-labels` | Display labels |
| `-l key=value` | Filter by label |
| `-w` or `--watch` | Watch for changes |
| `--dry-run=client` | Test without applying |
| `-f <file>` | Specify file |
| `--force` | Force operation |
| `-it` | Interactive terminal |

---

*This comprehensive guide covers Kubernetes from basics to production-ready deployments. Practice with minikube or kind locally, then apply these patterns to production clusters. Always test changes in non-production environments first!*
