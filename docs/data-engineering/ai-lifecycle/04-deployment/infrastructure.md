# Infrastructure

## Purpose

Design and provision cloud infrastructure to reliably serve models at scale with appropriate compute, storage, and networking resources.

## 1. Infrastructure Architecture

### Typical Architecture

```
┌─────────────────────────────────────────────────┐
│            Load Balancer / API Gateway          │
└─────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐   ┌────▼────┐   ┌────▼────┐
    │ Service │   │ Service │   │ Service │
    │ Pod 1   │   │ Pod 2   │   │ Pod 3   │
    └─────────┘   └─────────┘   └─────────┘
        │              │              │
        └──────────────┼──────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
    ┌───▼────┐   ┌────▼────┐   ┌────▼────┐
    │ Model  │   │ Feature  │   │ Logging  │
    │ Store  │   │ Store    │   │ / Monitor│
    └────────┘   └──────────┘   └──────────┘
```

## 2. Compute Resources

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model-service
  template:
    metadata:
      labels:
        app: ml-model-service
    spec:
      containers:
      - name: model-service
        image: my-registry/ml-model:v1.0.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "2Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        env:
        - name: MODEL_PATH
          value: /models/model_v1.pkl
```

### Auto-Scaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-service
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

## 3. Storage

### Model Storage

```
Cloud Storage (S3, GCS, etc.)
└── model-storage/
    ├── models/
    │   ├── v1.0.0/
    │   │   ├── model.pkl
    │   │   └── metadata.json
    │   └── v1.0.1/
    │       ├── model.pkl
    │       └── metadata.json
    ├── features/
    │   └── feature_definitions.yaml
    └── artifacts/
        └── training_config.json
```

### Feature Store

```python
# Using Feast feature store
import feast
from feast import FeatureStore

store = FeatureStore(repo_path=".")

# Online retrieval
features = store.get_online_features(
    features=[
        'customer_features:age',
        'customer_features:income',
        'customer_features:credit_score'
    ],
    entity_rows=[{'customer_id': 12345}]
).to_dict()
```

## 4. Networking

### API Gateway

```
┌─────────────────────────────┐
│      API Gateway            │
│  (Authentication, Rate      │
│   Limiting, Routing)        │
└──────────────┬──────────────┘
               │
        ┌──────┴──────┐
        │             │
  ┌─────▼──────┐  ┌──▼──────┐
  │  /predict  │  │ /health  │
  │            │  │          │
  │ Port 8000  │  │ Port 8001│
  └────────────┘  └──────────┘
```

## 5. Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Track predictions
prediction_counter = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model_version', 'outcome']
)

# Track latency
prediction_latency = Histogram(
    'model_prediction_duration_seconds',
    'Prediction latency',
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0)
)

# Track errors
errors_counter = Counter(
    'model_errors_total',
    'Total errors',
    ['error_type']
)

@app.route('/predict', methods=['POST'])
def predict():
    with prediction_latency.time():
        try:
            result = model.predict(features)
            prediction_counter.labels(
                model_version='v1.0.0',
                outcome='success'
            ).inc()
            return result
        except Exception as e:
            errors_counter.labels(error_type=type(e).__name__).inc()
            raise
```

### Structured Logging

```python
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_prediction(request_id, features, prediction, latency):
    logger.info(json.dumps({
        'event': 'prediction',
        'request_id': request_id,
        'features': features,
        'prediction': prediction,
        'latency_ms': latency,
        'timestamp': datetime.now().isoformat(),
    }))
```

## 6. Security

### Network Security

```yaml
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-service-policy
spec:
  podSelector:
    matchLabels:
      app: ml-model-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: api-gateway
    ports:
    - port: 8000
```

### Secrets Management

```python
# Using cloud secret manager
from google.cloud import secretmanager

def access_secret_version(secret_id, version_id="latest"):
    client = secretmanager.SecretManagerServiceClient()
    project_id = "my-project"
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

# Use in code
db_password = access_secret_version("db-password")
api_key = access_secret_version("model-api-key")
```

## 7. Disaster Recovery

### Backup Strategy

```
Daily incremental backup
Weekly full backup
Monthly archive backup

Retention:
- Daily: 7 days
- Weekly: 4 weeks
- Monthly: 1 year
```

### High Availability

```yaml
# Multi-region deployment
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  type: LoadBalancer
  selector:
    app: ml-model-service
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
```

## 8. Infrastructure as Code

### Terraform Example

```hcl
# main.tf
provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_container_cluster" "ml_cluster" {
  name     = "ml-model-cluster"
  location = var.region

  node_pool {
    initial_node_count = 3
    node_config {
      machine_type = "n1-standard-2"
      oauth_scopes = [
        "https://www.googleapis.com/auth/compute",
      ]
    }
  }
}

resource "google_storage_bucket" "model_bucket" {
  name          = "ml-models-${var.project_id}"
  location      = var.region
  force_destroy = false
}
```

## 9. Infrastructure Checklist

- [ ] Compute resources sized appropriately
- [ ] Auto-scaling configured
- [ ] Models stored securely
- [ ] Feature store set up
- [ ] API Gateway configured
- [ ] Monitoring and logging active
- [ ] Secrets management configured
- [ ] Network security configured
- [ ] Backup/recovery plan established
- [ ] Load testing completed
- [ ] Disaster recovery tested

## Best Practices

1. ✅ Use containers (Docker)
2. ✅ Orchestrate with Kubernetes
3. ✅ Infrastructure as Code
4. ✅ Multiple replicas
5. ✅ Auto-scaling enabled
6. ✅ Centralized logging
7. ✅ Health checks
8. ✅ Security hardened

---

## Related Documents

- [API Specification](./api-specification.md) - API design
- [Monitoring Plan](./monitoring-plan.md) - System monitoring
