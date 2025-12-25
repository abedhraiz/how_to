# Monitoring & Observability

## Purpose

Comprehensive guide to monitoring infrastructure, applications, and services using industry-standard tools. Learn to implement observability, set up dashboards, configure alerts, and troubleshoot issues in production systems.

## Technologies Covered

### Monitoring Stack
- **[Prometheus](./prometheus-grafana/prometheus-grafana-guide.md)** - Time-series database and monitoring system
- **[Grafana](./prometheus-grafana/prometheus-grafana-guide.md)** - Visualization and analytics platform

## Prerequisites

### Basic Requirements
- Understanding of system metrics (CPU, memory, disk, network)
- Basic networking knowledge
- Command-line proficiency
- Understanding of HTTP and APIs

### Recommended Knowledge
- Time-series data concepts
- Query languages (PromQL)
- Docker and container orchestration
- Alert management and on-call practices
- Infrastructure monitoring concepts

## Common Use Cases

### Infrastructure Monitoring
- ✅ Track server health (CPU, memory, disk usage)
- ✅ Monitor network performance
- ✅ Detect resource exhaustion
- ✅ Capacity planning
- ✅ Cost optimization

### Application Monitoring
- ✅ Track application performance (response time, throughput)
- ✅ Monitor error rates and exceptions
- ✅ Analyze user experience metrics
- ✅ Identify bottlenecks
- ✅ Debug production issues

### Service Reliability
- ✅ Measure SLA/SLO compliance
- ✅ Track service uptime
- ✅ Monitor dependencies
- ✅ Implement alerting
- ✅ On-call incident response

### Business Metrics
- ✅ Track key business KPIs
- ✅ Monitor user behavior
- ✅ Revenue and conversion tracking
- ✅ Custom business dashboards

## Learning Path

### Beginner (1-2 months)
1. **Monitoring Fundamentals**
   - Understand the four golden signals (latency, traffic, errors, saturation)
   - Learn about metrics, logs, and traces
   - Basic system monitoring concepts
   - Introduction to time-series data

2. **Prometheus Basics**
   - Install and configure Prometheus
   - Understand metric types (counter, gauge, histogram, summary)
   - Write basic PromQL queries
   - Set up service discovery
   - Expose application metrics

3. **Grafana Essentials**
   - Create dashboards
   - Add data sources
   - Build visualizations
   - Set up basic alerts

### Intermediate (2-3 months)
4. **Advanced PromQL**
   - Complex queries with aggregations
   - Rate and increase functions
   - Recording rules
   - Label manipulation
   - Query optimization

5. **Grafana Dashboards**
   - Design effective dashboards
   - Use variables and templating
   - Create alert rules
   - Implement annotations
   - Share and export dashboards

6. **Alerting**
   - Configure Alertmanager
   - Design alert rules
   - Set up notification channels
   - Implement alert routing
   - Reduce alert fatigue

### Advanced (3+ months)
7. **Production Observability**
   - Implement distributed tracing
   - Log aggregation and analysis
   - Correlate metrics, logs, and traces
   - Design SLO-based alerts
   - Build runbooks

8. **Enterprise Monitoring**
   - High availability setup
   - Long-term storage solutions
   - Multi-cluster monitoring
   - Security and access control
   - Compliance and auditing

## Observability Pillars

### The Three Pillars
```
Metrics (What happened?)
    ↓
Prometheus → Grafana
    - System metrics
    - Application metrics
    - Business metrics

Logs (Why it happened?)
    ↓
Log Aggregation → Analysis
    - Application logs
    - System logs
    - Audit logs

Traces (Where it happened?)
    ↓
Distributed Tracing
    - Request flow
    - Service dependencies
    - Performance bottlenecks
```

## Four Golden Signals

```
1. Latency
   - How long requests take
   - P50, P95, P99 percentiles

2. Traffic
   - Number of requests
   - Throughput (req/sec)

3. Errors
   - Failed requests
   - Error rate percentage

4. Saturation
   - Resource utilization
   - Capacity limits
```

## Related Categories

- 🏗️ **[Infrastructure & DevOps](../infrastructure-devops/README.md)** - Monitor infrastructure health
- ☁️ **[Cloud Platforms](../cloud-platforms/README.md)** - Cloud-native monitoring
- 🔧 **[Data Engineering](../data-engineering/README.md)** - Monitor data pipelines
- 🔄 **[CI/CD Automation](../cicd-automation/README.md)** - Monitor deployment pipelines
- 💾 **[Databases](../databases/README.md)** - Database performance monitoring

## Quick Start Examples

### Prometheus: Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
  
  - job_name: 'application'
    static_configs:
      - targets: ['app:8080']
```

### Prometheus: PromQL Queries
```promql
# CPU usage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# Memory usage percentage
(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes * 100

# HTTP request rate
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

### Application: Expose Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_REQUESTS = Gauge('http_requests_in_progress', 'Active HTTP requests')

# Instrument application
@REQUEST_DURATION.time()
def process_request(method, endpoint):
    ACTIVE_REQUESTS.inc()
    try:
        # Process request
        result = handle_request()
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status='200').inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status='500').inc()
        raise
    finally:
        ACTIVE_REQUESTS.dec()

# Start metrics server
start_http_server(8000)
```

### Grafana: Dashboard JSON
```json
{
  "dashboard": {
    "title": "Application Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Errors"
          }
        ]
      }
    ]
  }
}
```

### Alertmanager: Configuration
```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-notifications'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty'
    - match:
        severity: warning
      receiver: 'slack'

receivers:
  - name: 'team-notifications'
    email_configs:
      - to: 'team@example.com'
  
  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#alerts'
  
  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'your-service-key'
```

### Alert Rules
```yaml
# alert_rules.yml
groups:
  - name: application_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} for {{ $labels.instance }}"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"
      
      - alert: InstanceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Instance is down"
          description: "{{ $labels.instance }} is unreachable"
```

## Best Practices

### Metrics Collection
1. ✅ **Consistent Naming** - Use clear, standard metric names
2. ✅ **Appropriate Labels** - Add context without explosion
3. ✅ **Rate Over Count** - Use rate() for counters
4. ✅ **Histogram for Latency** - Track distributions, not averages
5. ✅ **Sample Rate** - Balance detail vs. overhead

### Dashboard Design
1. ✅ **Start with Overview** - High-level health indicators
2. ✅ **Drill-Down Capability** - From general to specific
3. ✅ **Use RED Method** - Rate, Errors, Duration for services
4. ✅ **Include SLOs** - Track service level objectives
5. ✅ **Avoid Clutter** - Focus on actionable metrics

### Alerting
1. ✅ **Alert on Symptoms** - Not causes (what users experience)
2. ✅ **Actionable Alerts** - Every alert needs a response
3. ✅ **Reduce Noise** - Group related alerts
4. ✅ **Context in Alerts** - Include runbook links
5. ✅ **Test Alerts** - Verify notification delivery

### Performance
1. ✅ **Limit Cardinality** - Avoid high-cardinality labels
2. ✅ **Use Recording Rules** - Pre-compute expensive queries
3. ✅ **Retention Policy** - Balance storage vs. history
4. ✅ **Federation** - Scale across multiple Prometheus instances

## Common Monitoring Patterns

### USE Method (Resources)
- **Utilization**: % time resource is busy
- **Saturation**: Queue depth or wait time
- **Errors**: Error count or rate

### RED Method (Services)
- **Rate**: Requests per second
- **Errors**: Failed requests per second
- **Duration**: Request latency distribution

## Navigation

- [← Back to Main Documentation](../../README.md)
- [→ Next: Databases](../databases/README.md)
