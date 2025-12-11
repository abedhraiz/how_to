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

### Apply YAML Configuration
```bash
# Apply a configuration file
kubectl apply -f deployment.yaml

# Apply all files in a directory
kubectl apply -f ./configs/

# Delete resources from a file
kubectl delete -f deployment.yaml
```

## Debugging and Troubleshooting

### Check Resource Status
```bash
# Get detailed information
kubectl describe <resource-type> <resource-name>

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp

# View resource usage
kubectl top nodes
kubectl top pods
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

## Best Practices

1. **Use Namespaces** - Organize resources and apply resource quotas
2. **Set Resource Limits** - Define CPU and memory limits for containers
3. **Use Labels** - Organize and select resources effectively
4. **Health Checks** - Implement liveness and readiness probes
5. **Use Secrets** - Never hardcode sensitive data
6. **Version Control** - Store YAML configurations in Git
7. **Use Helm** - Package manager for Kubernetes applications
8. **Monitor Resources** - Use tools like Prometheus and Grafana
9. **Backup Regularly** - Export critical configurations
10. **RBAC** - Implement proper role-based access control

## Common Issues and Solutions

### Pod Not Starting
```bash
# Check pod status
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Service Not Accessible
```bash
# Check service endpoints
kubectl get endpoints <service-name>
kubectl describe service <service-name>
```

### Resource Constraints
```bash
# Check node resources
kubectl describe nodes
kubectl top nodes
kubectl top pods
```

## Useful Resources

- Official Documentation: https://kubernetes.io/docs/
- Kubernetes GitHub: https://github.com/kubernetes/kubernetes
- kubectl Cheat Sheet: https://kubernetes.io/docs/reference/kubectl/cheatsheet/
- Interactive Tutorial: https://kubernetes.io/docs/tutorials/

## Quick Reference

| Command | Description |
|---------|-------------|
| `kubectl get pods` | List all pods |
| `kubectl get services` | List all services |
| `kubectl get deployments` | List all deployments |
| `kubectl apply -f file.yaml` | Apply configuration |
| `kubectl delete -f file.yaml` | Delete resources from file |
| `kubectl logs <pod>` | View pod logs |
| `kubectl exec -it <pod> -- bash` | Shell into pod |
| `kubectl describe <resource> <name>` | Detailed resource info |
| `kubectl port-forward <pod> 8080:80` | Forward port |
| `kubectl scale deployment <name> --replicas=3` | Scale deployment |

---

*This guide covers the fundamentals of Kubernetes. For production deployments, consider additional topics like RBAC, network policies, persistent volumes, and monitoring solutions.*
