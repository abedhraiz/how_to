# Prometheus & Grafana Guide

## What are Prometheus and Grafana?

**Prometheus** is an open-source monitoring and alerting toolkit designed for reliability and scalability. It collects and stores metrics as time series data.

**Grafana** is an open-source analytics and visualization platform that works with Prometheus and other data sources to create beautiful dashboards.

Together, they form a powerful monitoring stack used by organizations worldwide.

## Prerequisites

- Basic understanding of metrics and monitoring concepts
- Docker (recommended for quick setup)
- Linux/Unix system knowledge
- Understanding of HTTP and APIs

## Core Concepts

### Prometheus

- **Metrics** - Numerical measurements over time
- **Labels** - Key-value pairs for dimensional data
- **Time Series** - Stream of timestamped values
- **Scraping** - Pulling metrics from targets
- **PromQL** - Prometheus Query Language
- **Alerting** - Triggering notifications based on metrics

### Grafana

- **Data Sources** - Where data comes from (Prometheus, etc.)
- **Dashboards** - Collections of panels
- **Panels** - Individual visualizations
- **Variables** - Dynamic dashboard elements
- **Alerts** - Notifications based on query results

## Installation

### Prometheus

#### Using Docker

```bash
# Run Prometheus
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Access at http://localhost:9090
```

#### Using Binary (Linux)

```bash
# Download Prometheus
wget https://github.com/prometheus/prometheus/releases/download/v2.45.0/prometheus-2.45.0.linux-amd64.tar.gz

# Extract
tar xvfz prometheus-*.tar.gz
cd prometheus-*

# Run Prometheus
./prometheus --config.file=prometheus.yml
```

#### Configuration (prometheus.yml)

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'my-cluster'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load rules
rule_files:
  - "rules/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node exporter
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
        labels:
          group: 'production'

  # Application
  - job_name: 'my-app'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

### Grafana

#### Using Docker

```bash
# Run Grafana
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
  grafana/grafana

# Access at http://localhost:3000
# Default credentials: admin/admin
```

#### Using Binary (Linux)

```bash
# Add repository
sudo apt-get install -y software-properties-common
sudo add-apt-repository "deb https://packages.grafana.com/oss/deb stable main"

# Install
wget -q -O - https://packages.grafana.com/gpg.key | sudo apt-key add -
sudo apt-get update
sudo apt-get install grafana

# Start service
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./rules:/etc/prometheus/rules
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    command:
      - '--path.rootfs=/host'
    volumes:
      - '/:/host:ro,rslave'
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

```bash
# Start stack
docker-compose up -d

# View logs
docker-compose logs -f

# Stop stack
docker-compose down
```

## Prometheus Basics

### Metric Types

#### Counter
Cumulative metric that only increases.

```python
from prometheus_client import Counter

requests_total = Counter('requests_total', 'Total requests')
requests_total.inc()  # Increment by 1
requests_total.inc(5)  # Increment by 5
```

#### Gauge
Metric that can go up and down.

```python
from prometheus_client import Gauge

temperature = Gauge('temperature_celsius', 'Temperature in Celsius')
temperature.set(25.5)
temperature.inc(2)  # Increase by 2
temperature.dec(1)  # Decrease by 1
```

#### Histogram
Samples observations and counts them in configurable buckets.

```python
from prometheus_client import Histogram

request_duration = Histogram(
    'request_duration_seconds',
    'Request duration in seconds'
)

with request_duration.time():
    # Your code here
    process_request()
```

#### Summary
Similar to histogram but calculates quantiles.

```python
from prometheus_client import Summary

request_latency = Summary(
    'request_latency_seconds',
    'Request latency in seconds'
)

with request_latency.time():
    handle_request()
```

### Instrumenting Applications

#### Python Application

```python
from prometheus_client import start_http_server, Counter, Histogram
import random
import time

# Define metrics
REQUEST_COUNT = Counter(
    'app_requests_total',
    'Total app requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'app_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

def process_request(method, endpoint):
    # Simulate processing
    duration = random.random()
    time.sleep(duration)
    
    status = 200 if random.random() > 0.1 else 500
    
    # Record metrics
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    return status

if __name__ == '__main__':
    # Start metrics server
    start_http_server(8000)
    
    print("Metrics available at http://localhost:8000/metrics")
    
    # Simulate requests
    while True:
        process_request('GET', '/api/users')
        time.sleep(1)
```

#### Node.js Application

```javascript
const express = require('express');
const promClient = require('prom-client');

const app = express();

// Create a Registry
const register = new promClient.Registry();

// Add default metrics
promClient.collectDefaultMetrics({ register });

// Custom metrics
const httpRequestDuration = new promClient.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

const httpRequestTotal = new promClient.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

// Middleware to track metrics
app.use((req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    
    httpRequestDuration
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .observe(duration);
    
    httpRequestTotal
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .inc();
  });
  
  next();
});

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

// Your routes
app.get('/api/users', (req, res) => {
  res.json({ users: [] });
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
  console.log('Metrics available at http://localhost:3000/metrics');
});
```

### PromQL (Prometheus Query Language)

#### Basic Queries

```promql
# Instant vector - current value
http_requests_total

# Filter by label
http_requests_total{job="api-server", method="GET"}

# Range vector - values over time
http_requests_total[5m]

# Rate of increase over 5 minutes
rate(http_requests_total[5m])

# Sum across dimensions
sum(http_requests_total)

# Sum by label
sum by (job) (http_requests_total)

# Average
avg(http_request_duration_seconds)

# 95th percentile
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
```

#### Advanced Queries

```promql
# Request rate per second
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m])

# Requests per minute
rate(http_requests_total[1m]) * 60

# Top 5 endpoints by request count
topk(5, sum by (endpoint) (rate(http_requests_total[5m])))

# Memory usage percentage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) /
node_memory_MemTotal_bytes * 100

# CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Predict disk full in 4 hours
predict_linear(node_filesystem_free_bytes[1h], 4 * 3600) < 0

# Alert if value changes
abs(delta(some_metric[5m])) > 10
```

### Exporters

#### Node Exporter (System Metrics)

```bash
# Run node exporter
docker run -d \
  --name node-exporter \
  -p 9100:9100 \
  --net="host" \
  --pid="host" \
  -v "/:/host:ro,rslave" \
  prom/node-exporter \
  --path.rootfs=/host

# View metrics
curl http://localhost:9100/metrics
```

Common metrics:
- `node_cpu_seconds_total` - CPU usage
- `node_memory_MemAvailable_bytes` - Available memory
- `node_disk_io_time_seconds_total` - Disk I/O
- `node_network_receive_bytes_total` - Network traffic

#### Blackbox Exporter (Endpoint Monitoring)

```yaml
# blackbox.yml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: []  # Defaults to 2xx
      method: GET
      
  tcp_connect:
    prober: tcp
    timeout: 5s
```

```bash
# Run blackbox exporter
docker run -d \
  --name blackbox-exporter \
  -p 9115:9115 \
  -v $(pwd)/blackbox.yml:/etc/blackbox_exporter/config.yml \
  prom/blackbox-exporter \
  --config.file=/etc/blackbox_exporter/config.yml
```

Add to Prometheus config:
```yaml
scrape_configs:
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - https://example.com
          - https://api.example.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: localhost:9115
```

### Alerting Rules

Create `rules/alerts.yml`:

```yaml
groups:
  - name: instance_alerts
    interval: 30s
    rules:
      # Instance down
      - alert: InstanceDown
        expr: up == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Instance {{ $labels.instance }} down"
          description: "{{ $labels.instance }} has been down for more than 5 minutes"

      # High CPU usage
      - alert: HighCPUUsage
        expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.instance }}"
          description: "CPU usage is above 80% (current: {{ $value }}%)"

      # High memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.instance }}"
          description: "Memory usage is above 90% (current: {{ $value }}%)"

      # Disk space low
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100 < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low disk space on {{ $labels.instance }}"
          description: "Disk space is below 10% (current: {{ $value }}%)"

      # High error rate
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% (current: {{ $value }}%)"

      # Service latency high
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is above 1s (current: {{ $value }}s)"
```

### Alertmanager Configuration

Create `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@example.com'
  smtp_auth_username: 'alerts@example.com'
  smtp_auth_password: 'password'

# Templates
templates:
  - '/etc/alertmanager/templates/*.tmpl'

# Route tree
route:
  receiver: 'default-receiver'
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  
  routes:
    # Critical alerts
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true
    
    # Warning alerts
    - match:
        severity: warning
      receiver: 'slack'

# Receivers
receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'team@example.com'
        headers:
          Subject: 'Alert: {{ .GroupLabels.alertname }}'

  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#alerts'
        title: 'Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}: {{ .CommonAnnotations.summary }}'

# Inhibition rules
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

## Grafana Setup

### Add Prometheus Data Source

1. Login to Grafana (http://localhost:3000)
2. Go to Configuration → Data Sources
3. Click "Add data source"
4. Select "Prometheus"
5. Enter URL: `http://prometheus:9090` (or `http://localhost:9090`)
6. Click "Save & Test"

### Or via Configuration File

Create `grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

### Creating Dashboards

#### Simple Dashboard JSON

Create `grafana/provisioning/dashboards/system.json`:

```json
{
  "dashboard": {
    "title": "System Metrics",
    "panels": [
      {
        "id": 1,
        "type": "graph",
        "title": "CPU Usage",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{ instance }}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "type": "graph",
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100",
            "legendFormat": "{{ instance }}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      }
    ]
  }
}
```

### Popular Community Dashboards

Import from https://grafana.com/grafana/dashboards/

**Node Exporter Full**
- Dashboard ID: 1860
- Comprehensive system metrics

**Kubernetes Cluster Monitoring**
- Dashboard ID: 7249
- K8s cluster overview

**NGINX Ingress Controller**
- Dashboard ID: 9614
- Ingress metrics

**MySQL Overview**
- Dashboard ID: 7362
- Database performance

### Variables in Dashboards

```json
{
  "templating": {
    "list": [
      {
        "name": "instance",
        "type": "query",
        "datasource": "Prometheus",
        "query": "label_values(node_cpu_seconds_total, instance)",
        "multi": true,
        "includeAll": true
      },
      {
        "name": "interval",
        "type": "interval",
        "options": ["1m", "5m", "15m", "30m", "1h"],
        "auto_count": 30,
        "auto_min": "10s"
      }
    ]
  }
}
```

Use in queries:
```promql
rate(node_cpu_seconds_total{instance=~"$instance"}[$interval])
```

## Complete Monitoring Stack Example

### Application with Metrics

```python
# app.py
from flask import Flask, jsonify
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import time
import random

app = Flask(__name__)

# Metrics
REQUEST_COUNT = Counter(
    'app_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'app_request_duration_seconds',
    'Request duration',
    ['method', 'endpoint']
)

@app.route('/api/users')
def get_users():
    start_time = time.time()
    
    # Simulate processing
    time.sleep(random.uniform(0.1, 0.5))
    
    # Simulate occasional errors
    if random.random() < 0.1:
        status = 500
        response = jsonify({'error': 'Internal server error'}), 500
    else:
        status = 200
        response = jsonify({'users': ['Alice', 'Bob', 'Charlie']})
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels('GET', '/api/users', status).inc()
    REQUEST_DURATION.labels('GET', '/api/users').observe(duration)
    
    return response

@app.route('/metrics')
def metrics():
    return generate_latest(REGISTRY)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Full Stack Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8080:8080"
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    restart: unless-stopped

  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
    restart: unless-stopped
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'rules/*.yml'

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'app'
    static_configs:
      - targets: ['app:8080']
    metrics_path: '/metrics'
```

## Real-World Examples

### Example 1: Complete Monitoring Stack

**docker-compose-complete.yml:**
```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:latest
    container_name: alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager:/etc/alertmanager
      - alertmanager-data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    restart: unless-stopped
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    ports:
      - "9100:9100"
    command:
      - '--path.rootfs=/host'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    volumes:
      - '/:/host:ro,rslave'
    restart: unless-stopped
    networks:
      - monitoring

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    devices:
      - /dev/kmsg
    restart: unless-stopped
    networks:
      - monitoring

  blackbox-exporter:
    image: prom/blackbox-exporter:latest
    container_name: blackbox-exporter
    ports:
      - "9115:9115"
    volumes:
      - ./blackbox:/etc/blackbox_exporter
    command:
      - '--config.file=/etc/blackbox_exporter/blackbox.yml'
    restart: unless-stopped
    networks:
      - monitoring

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    container_name: postgres-exporter
    ports:
      - "9187:9187"
    environment:
      DATA_SOURCE_NAME: "postgresql://user:password@postgres:5432/database?sslmode=disable"
    restart: unless-stopped
    networks:
      - monitoring

  redis-exporter:
    image: oliver006/redis_exporter:latest
    container_name: redis-exporter
    ports:
      - "9121:9121"
    environment:
      REDIS_ADDR: "redis:6379"
      REDIS_PASSWORD: "your_redis_password"
    restart: unless-stopped
    networks:
      - monitoring

  nginx-exporter:
    image: nginx/nginx-prometheus-exporter:latest
    container_name: nginx-exporter
    ports:
      - "9113:9113"
    command:
      - '-nginx.scrape-uri=http://nginx:80/stub_status'
    restart: unless-stopped
    networks:
      - monitoring

  pushgateway:
    image: prom/pushgateway:latest
    container_name: pushgateway
    ports:
      - "9091:9091"
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:
  alertmanager-data:

networks:
  monitoring:
    driver: bridge
```

**prometheus/prometheus.yml:**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'production'
    environment: 'prod'

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - 'alerts/*.yml'
  - 'rules/*.yml'

scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
        labels:
          service: 'prometheus'

  # Node Exporter - System metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
        labels:
          instance: 'server-1'
          datacenter: 'us-east-1'

  # cAdvisor - Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Blackbox Exporter - Endpoint monitoring
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - https://example.com
          - https://api.example.com
          - https://www.google.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115

  # PostgreSQL Exporter
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
        labels:
          database: 'production'

  # Redis Exporter
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
        labels:
          instance: 'redis-cache'

  # Nginx Exporter
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']

  # Application metrics
  - job_name: 'application'
    static_configs:
      - targets: ['app:8000']
        labels:
          service: 'api'
          version: 'v1.0.0'

  # Pushgateway - Batch jobs
  - job_name: 'pushgateway'
    static_configs:
      - targets: ['pushgateway:9091']
    honor_labels: true

  # Service discovery (Kubernetes example)
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
```

**prometheus/alerts/application.yml:**
```yaml
groups:
  - name: application_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
            /
            sum(rate(http_requests_total[5m])) by (service)
          ) * 100 > 5
        for: 5m
        labels:
          severity: critical
          team: backend
        annotations:
          summary: "High error rate detected"
          description: "{{ $labels.service }} has {{ $value | humanize }}% error rate"

      # High response time
      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 1
        for: 5m
        labels:
          severity: warning
          team: backend
        annotations:
          summary: "High response time"
          description: "{{ $labels.service }} 95th percentile response time is {{ $value }}s"

      # Service down
      - alert: ServiceDown
        expr: up{job="application"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Service is down"
          description: "{{ $labels.instance }} is down"

      # High memory usage
      - alert: HighMemoryUsage
        expr: |
          (
            process_resident_memory_bytes{job="application"}
            /
            node_memory_MemTotal_bytes
          ) * 100 > 80
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High memory usage"
          description: "{{ $labels.instance }} memory usage is {{ $value | humanize }}%"

      # Database connection pool exhausted
      - alert: DBConnectionPoolExhausted
        expr: |
          (
            database_connections_active
            /
            database_connections_max
          ) * 100 > 90
        for: 5m
        labels:
          severity: critical
          team: database
        annotations:
          summary: "Database connection pool nearly exhausted"
          description: "{{ $labels.instance }} connection pool is {{ $value }}% full"

      # Slow database queries
      - alert: SlowDatabaseQueries
        expr: |
          histogram_quantile(0.95,
            rate(database_query_duration_seconds_bucket[5m])
          ) > 2
        for: 10m
        labels:
          severity: warning
          team: database
        annotations:
          summary: "Slow database queries detected"
          description: "95th percentile query time is {{ $value }}s"
```

**prometheus/alerts/infrastructure.yml:**
```yaml
groups:
  - name: infrastructure_alerts
    interval: 30s
    rules:
      # High CPU usage
      - alert: HighCPUUsage
        expr: |
          100 - (
            avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100
          ) > 80
        for: 10m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High CPU usage"
          description: "{{ $labels.instance }} CPU usage is {{ $value | humanize }}%"

      # High disk usage
      - alert: HighDiskUsage
        expr: |
          (
            (
              node_filesystem_size_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}
              -
              node_filesystem_free_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}
            )
            /
            node_filesystem_size_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}
          ) * 100 > 85
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High disk usage"
          description: "{{ $labels.instance }} disk {{ $labels.mountpoint }} usage is {{ $value | humanize }}%"

      # Disk will fill in 4 hours
      - alert: DiskWillFillSoon
        expr: |
          predict_linear(
            node_filesystem_free_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}[1h],
            4 * 3600
          ) < 0
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Disk will fill soon"
          description: "{{ $labels.instance }} disk {{ $labels.mountpoint }} will fill in approximately 4 hours"

      # High memory usage
      - alert: HighMemoryUsage
        expr: |
          (
            (
              node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
            )
            /
            node_memory_MemTotal_bytes
          ) * 100 > 90
        for: 10m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High memory usage"
          description: "{{ $labels.instance }} memory usage is {{ $value | humanize }}%"

      # High load average
      - alert: HighLoadAverage
        expr: node_load15 / count(node_cpu_seconds_total{mode="idle"}) by (instance) > 2
        for: 15m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High load average"
          description: "{{ $labels.instance }} load average is {{ $value }}"

      # Network errors
      - alert: NetworkErrors
        expr: rate(node_network_receive_errs_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
          team: network
        annotations:
          summary: "Network interface errors"
          description: "{{ $labels.instance }} interface {{ $labels.device }} has {{ $value }} errors/s"

      # Container restarts
      - alert: ContainerRestarting
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "Container restarting"
          description: "Pod {{ $labels.namespace }}/{{ $labels.pod }} container {{ $labels.container }} is restarting"
```

**alertmanager/alertmanager.yml:**
```yaml
global:
  resolve_timeout: 5m
  slack_api_url: 'YOUR_SLACK_WEBHOOK_URL'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    # Critical alerts go to PagerDuty
    - match:
        severity: critical
      receiver: 'pagerduty'
      continue: true
    
    # All alerts go to Slack
    - match_re:
        severity: ^(warning|critical)$
      receiver: 'slack'
    
    # Database team alerts
    - match:
        team: database
      receiver: 'database-team'
    
    # Platform team alerts
    - match:
        team: platform
      receiver: 'platform-team'

receivers:
  - name: 'default'
    email_configs:
      - to: 'alerts@example.com'
        from: 'prometheus@example.com'
        smarthost: smtp.example.com:587
        auth_username: 'prometheus@example.com'
        auth_password: 'password'

  - name: 'slack'
    slack_configs:
      - channel: '#alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
        send_resolved: true

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
        description: '{{ .GroupLabels.alertname }}'

  - name: 'database-team'
    slack_configs:
      - channel: '#database-alerts'
        send_resolved: true
    email_configs:
      - to: 'database-team@example.com'

  - name: 'platform-team'
    slack_configs:
      - channel: '#platform-alerts'
        send_resolved: true
    email_configs:
      - to: 'platform-team@example.com'

inhibit_rules:
  # Inhibit warning alerts if critical alert is firing
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

**blackbox/blackbox.yml:**
```yaml
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: []  # Defaults to 2xx
      method: GET
      preferred_ip_protocol: "ip4"

  http_post_2xx:
    prober: http
    http:
      method: POST
      headers:
        Content-Type: application/json
      body: '{"test": true}'

  tcp_connect:
    prober: tcp
    timeout: 5s

  icmp:
    prober: icmp
    timeout: 5s
    icmp:
      preferred_ip_protocol: "ip4"

  dns_query:
    prober: dns
    dns:
      query_name: "example.com"
      query_type: "A"
```

### Example 2: Grafana Dashboard (JSON)

**grafana/dashboards/application-overview.json:**
```json
{
  "dashboard": {
    "title": "Application Overview",
    "tags": ["application", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "yaxes": [
          {"format": "reqps", "label": "Requests/sec"},
          {"format": "short"}
        ]
      },
      {
        "id": 2,
        "title": "Error Rate",
        "type": "graph",
        "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service) * 100",
            "legendFormat": "{{service}}"
          }
        ],
        "yaxes": [
          {"format": "percent", "label": "Error Rate %"},
          {"format": "short"}
        ]
      },
      {
        "id": 3,
        "title": "Response Time (p95)",
        "type": "graph",
        "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))",
            "legendFormat": "{{service}} p95"
          }
        ],
        "yaxes": [
          {"format": "s", "label": "Latency"},
          {"format": "short"}
        ]
      },
      {
        "id": 4,
        "title": "Active Connections",
        "type": "graph",
        "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
        "targets": [
          {
            "expr": "sum(database_connections_active) by (instance)",
            "legendFormat": "{{instance}}"
          }
        ]
      }
    ]
  }
}
```

**grafana/provisioning/dashboards/dashboard.yml:**
```yaml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

**grafana/provisioning/datasources/prometheus.yml:**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

### Example 3: Custom Application Metrics

**Python Flask Application:**
```python
from flask import Flask, request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
import time
import random
import psutil

app = Flask(__name__)

# Define metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_inprogress',
    'Number of requests in progress',
    ['method', 'endpoint']
)

DATABASE_CONNECTIONS = Gauge(
    'database_connections_active',
    'Active database connections'
)

CACHE_HITS = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

QUEUE_SIZE = Gauge(
    'queue_size',
    'Current queue size',
    ['queue_name']
)

CPU_USAGE = Gauge(
    'cpu_usage_percent',
    'CPU usage percentage'
)

MEMORY_USAGE = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes'
)

# Middleware to track metrics
@app.before_request
def before_request():
    request.start_time = time.time()
    ACTIVE_REQUESTS.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown'
    ).inc()

@app.after_request
def after_request(response):
    latency = time.time() - request.start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown',
        status=response.status_code
    ).inc()
    
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown'
    ).observe(latency)
    
    ACTIVE_REQUESTS.labels(
        method=request.method,
        endpoint=request.endpoint or 'unknown'
    ).dec()
    
    return response

# Background task to update system metrics
def update_system_metrics():
    while True:
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().used)
        time.sleep(5)

@app.route('/api/users', methods=['GET'])
def get_users():
    # Simulate cache check
    if random.random() > 0.3:
        CACHE_HITS.labels(cache_type='redis').inc()
    else:
        CACHE_MISSES.labels(cache_type='redis').inc()
    
    # Simulate database connections
    DATABASE_CONNECTIONS.set(random.randint(10, 50))
    
    # Simulate processing time
    time.sleep(random.uniform(0.01, 0.1))
    
    return {'users': []}, 200

@app.route('/api/orders', methods=['POST'])
def create_order():
    # Simulate queue size
    QUEUE_SIZE.labels(queue_name='orders').set(random.randint(0, 100))
    
    time.sleep(random.uniform(0.05, 0.2))
    
    if random.random() > 0.95:
        return {'error': 'Server error'}, 500
    
    return {'order_id': 123}, 201

@app.route('/metrics')
def metrics():
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

@app.route('/health')
def health():
    return {'status': 'healthy'}, 200

if __name__ == '__main__':
    import threading
    
    # Start background metrics updater
    metrics_thread = threading.Thread(target=update_system_metrics, daemon=True)
    metrics_thread.start()
    
    app.run(host='0.0.0.0', port=8000)
```

### Example 4: Recording Rules for Performance

**prometheus/rules/recording.yml:**
```yaml
groups:
  - name: api_performance
    interval: 30s
    rules:
      # Request rates
      - record: job:http_requests:rate5m
        expr: sum(rate(http_requests_total[5m])) by (job, service)

      - record: job:http_requests:rate1h
        expr: sum(rate(http_requests_total[1h])) by (job, service)

      # Error rates
      - record: job:http_errors:rate5m
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (job, service)

      - record: job:http_error_ratio:rate5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (job, service)
          /
          sum(rate(http_requests_total[5m])) by (job, service)

      # Latency percentiles
      - record: job:http_latency:p50
        expr: histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job, service))

      - record: job:http_latency:p95
        expr: histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job, service))

      - record: job:http_latency:p99
        expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job, service))

  - name: infrastructure_utilization
    interval: 30s
    rules:
      # CPU utilization
      - record: instance:cpu_utilization:rate5m
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

      # Memory utilization
      - record: instance:memory_utilization:ratio
        expr: |
          (
            node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
          ) / node_memory_MemTotal_bytes

      # Disk utilization
      - record: instance:disk_utilization:ratio
        expr: |
          (
            node_filesystem_size_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}
            -
            node_filesystem_free_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}
          ) / node_filesystem_size_bytes{fstype!~"tmpfs|fuse.lxcfs|squashfs|vfat"}

      # Network throughput
      - record: instance:network_receive:rate5m
        expr: sum(rate(node_network_receive_bytes_total[5m])) by (instance)

      - record: instance:network_transmit:rate5m
        expr: sum(rate(node_network_transmit_bytes_total[5m])) by (instance)
```

### Example 5: Service Level Objectives (SLOs)

**prometheus/rules/slo.yml:**
```yaml
groups:
  - name: slo_rules
    interval: 30s
    rules:
      # Availability SLO (99.9%)
      - record: slo:availability:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{status!~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)

      # Error budget remaining (monthly)
      - record: slo:error_budget:remaining
        expr: |
          1 - (
            (1 - slo:availability:ratio_rate5m)
            /
            (1 - 0.999)
          )

      # Latency SLO (95% < 200ms)
      - record: slo:latency:p95_ratio
        expr: |
          (
            histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)) < 0.2
          ) * 100

  - name: slo_alerts
    interval: 30s
    rules:
      - alert: SLOAvailabilityBreach
        expr: slo:availability:ratio_rate5m < 0.999
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "SLO availability breach"
          description: "{{ $labels.service }} availability is {{ $value | humanizePercentage }}"

      - alert: SLOErrorBudgetLow
        expr: slo:error_budget:remaining < 0.1
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "SLO error budget running low"
          description: "{{ $labels.service }} error budget is {{ $value | humanizePercentage }} remaining"
```

## Best Practices

### 1. Metric Naming Conventions

```
# Good naming
http_requests_total
http_request_duration_seconds
database_queries_total
cache_hits_total

# Bad naming
HttpRequestsCount
request_time
db_query_count
```

### 2. Label Usage

```promql
# Good - Use labels for dimensions
http_requests_total{method="GET", endpoint="/api/users", status="200"}

# Bad - Don't create separate metrics
http_requests_get_api_users_200_total
```

### 3. Recording Rules

Create `rules/recording.yml`:

```yaml
groups:
  - name: recording_rules
    interval: 30s
    rules:
      # Pre-calculate expensive queries
      - record: job:http_requests:rate5m
        expr: rate(http_requests_total[5m])

      - record: job:http_errors:rate5m
        expr: rate(http_requests_total{status=~"5.."}[5m])

      - record: job:http_error_rate:ratio
        expr: job:http_errors:rate5m / job:http_requests:rate5m
```

### 4. Retention and Storage

```bash
# Set retention period
prometheus \
  --storage.tsdb.retention.time=30d \
  --storage.tsdb.retention.size=50GB
```

### 5. High Availability

```yaml
# Run multiple Prometheus instances
# Use federation or Thanos for long-term storage
```

## Troubleshooting

### Check Prometheus Targets

Visit: http://localhost:9090/targets

### Check Alertmanager Status

Visit: http://localhost:9093/#/status

### Common Issues

**Metrics not showing up:**
```bash
# Check if target is up
curl http://localhost:8080/metrics

# Check Prometheus logs
docker logs prometheus

# Verify scrape configuration
```

**High memory usage:**
```bash
# Reduce retention time
# Reduce scrape frequency
# Use recording rules for expensive queries
```

**Alert not firing:**
```bash
# Check rules syntax
promtool check rules rules/*.yml

# Check Alertmanager config
promtool check config alertmanager.yml
```

## Resources

- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **PromQL**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Exporters**: https://prometheus.io/docs/instrumenting/exporters/
- **Dashboards**: https://grafana.com/grafana/dashboards/

## Quick Reference

### PromQL Functions

| Function | Description | Example |
|----------|-------------|---------|
| `rate()` | Per-second rate | `rate(requests[5m])` |
| `increase()` | Total increase | `increase(requests[1h])` |
| `sum()` | Sum values | `sum(requests)` |
| `avg()` | Average values | `avg(temperature)` |
| `max()` | Maximum value | `max(cpu_usage)` |
| `min()` | Minimum value | `min(memory_free)` |
| `count()` | Count values | `count(up == 1)` |
| `topk()` | Top K values | `topk(5, requests)` |
| `bottomk()` | Bottom K values | `bottomk(3, latency)` |

### Common Metrics Patterns

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) /
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_duration_seconds_bucket[5m]))

# CPU usage percentage
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage percentage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) /
node_memory_MemTotal_bytes * 100

# Disk usage percentage
(node_filesystem_size_bytes - node_filesystem_free_bytes) /
node_filesystem_size_bytes * 100
```

---

*This guide covers Prometheus and Grafana fundamentals for monitoring and observability. Start with basic metrics and gradually build comprehensive monitoring for your infrastructure and applications.*
