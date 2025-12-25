# Monitoring Plan

## Purpose

Establish comprehensive monitoring to detect performance degradation, anomalies, and failures in production model systems.

## 1. Monitoring Metrics

### Model Performance Metrics

```
Accuracy: % of correct predictions
Precision: Of positive predictions, how many are correct
Recall: Of actual positives, how many did we catch
F1-Score: Harmonic mean of precision and recall
AUC-ROC: Model's ability to distinguish classes

Example thresholds:
- Alert if accuracy drops below 90%
- Alert if AUC-ROC drops below 0.85
```

### System Metrics

```
CPU Usage: % of CPU utilized
Memory Usage: % of memory utilized
Disk Usage: % of disk space used
Network I/O: Network traffic in/out

Example thresholds:
- Alert if CPU > 85%
- Alert if Memory > 90%
- Alert if Disk > 95%
```

### Application Metrics

```
Request Rate: Requests per second
Error Rate: % of requests that error
Latency P50: 50th percentile response time
Latency P99: 99th percentile response time

Example thresholds:
- Alert if error rate > 5%
- Alert if P99 latency > 1000ms
```

## 2. Monitoring Dashboard

### Key Dashboard Panels

```
┌─────────────────────────────────────────┐
│  Model Service Health Dashboard         │
├─────────────────────────────────────────┤
│                                         │
│  Model Accuracy: 0.92                   │
│  System Status: Healthy                 │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Request Latency (Last 24h)      │   │
│  │ P50: 45ms  P99: 180ms           │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Error Rate                      │   │
│  │ Current: 0.2%  (Target: <0.5%)  │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ Resource Usage                  │   │
│  │ CPU: 42%  Memory: 55%           │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

## 3. Alerting Rules

### Critical Alerts

```
Rule: Model Unavailable
  Condition: No successful predictions in 5 minutes
  Action: Page on-call engineer immediately
  Response Time: < 5 minutes

Rule: High Error Rate
  Condition: Error rate > 5%
  Action: Page on-call engineer
  Response Time: < 15 minutes

Rule: Accuracy Drop
  Condition: Accuracy < 85%
  Action: Create incident, investigate
  Response Time: < 30 minutes
```

### Warning Alerts

```
Rule: Degraded Performance
  Condition: P99 latency > 500ms
  Action: Send email to team
  Review: Daily

Rule: High Resource Usage
  Condition: Memory > 85%
  Action: Send email to ops team
  Review: Daily
```

## 4. Data Quality Monitoring

### Data Drift Detection

```python
def detect_data_drift(current_data, baseline_data):
    """Compare current data distribution to baseline"""
    
    drift_metrics = {}
    
    for feature in baseline_data.columns:
        # Kolmogorov-Smirnov test
        stat, p_value = stats.ks_2samp(
            baseline_data[feature],
            current_data[feature]
        )
        drift_metrics[feature] = {
            'ks_statistic': stat,
            'p_value': p_value,
            'is_drifted': p_value < 0.05  # Significant difference
        }
    
    return drift_metrics
```

### Missing Data Monitoring

```python
def monitor_missing_data(data):
    """Monitor missing values in incoming data"""
    
    missing_summary = {
        'timestamp': datetime.now(),
        'total_records': len(data),
        'missing_by_feature': {}
    }
    
    for column in data.columns:
        missing_pct = data[column].isnull().sum() / len(data) * 100
        missing_summary['missing_by_feature'][column] = missing_pct
        
        if missing_pct > 10:
            logger.warning(f"High missing rate in {column}: {missing_pct}%")
    
    return missing_summary
```

## 5. Logging Strategy

### Structured Logging

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'extra': {
                'request_id': getattr(record, 'request_id', None),
                'user_id': getattr(record, 'user_id', None),
                'model_version': getattr(record, 'model_version', None),
            }
        }
        return json.dumps(log_data)

# Usage
logger.info(
    "Prediction generated",
    extra={
        'request_id': 'req_123',
        'user_id': 'user_456',
        'model_version': 'v1.0.0'
    }
)
```

## 6. Incident Response

### Incident Severity Levels

```
CRITICAL (Severity 1)
- Model completely unavailable
- Production down
- Customer impact: Complete
- Response: Immediate (on-call page)
- Resolution time: < 30 minutes

MAJOR (Severity 2)
- Significant performance degradation
- Some users impacted
- Response: < 30 minutes
- Resolution time: < 2 hours

MINOR (Severity 3)
- Low impact issues
- Workaround exists
- Response: < 2 hours
- Resolution time: < 1 day
```

### Incident Response Process

```
1. DETECT: Alert fires
2. ALERT: Page on-call engineer
3. ACKNOWLEDGE: Engineer acknowledges
4. INVESTIGATE: Determine root cause
5. MITIGATE: Implement fix or rollback
6. NOTIFY: Update stakeholders
7. RESOLVE: Confirm fix works
8. POST-MORTEM: Review and improve
```

## 7. Metrics Export

### Prometheus Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-model-service'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Custom Metrics Export

```python
from prometheus_client import start_http_server, Counter, Histogram

# Start metrics server
start_http_server(8000)

# Define metrics
model_predictions = Counter(
    'model_predictions_total',
    'Total predictions',
    ['model', 'outcome']
)

prediction_latency = Histogram(
    'model_prediction_seconds',
    'Prediction latency'
)

# Export metrics
@app.route('/metrics')
def metrics():
    return generate_latest()
```

## 8. Monitoring Checklist

- [ ] Model performance metrics defined
- [ ] System metrics tracked
- [ ] Application metrics collected
- [ ] Alerts configured for critical issues
- [ ] Dashboard created for operations
- [ ] Data drift detection enabled
- [ ] Logging configured
- [ ] Metrics retention policy set
- [ ] On-call rotation established
- [ ] Incident response plan documented

## Best Practices

1. ✅ Monitor early and often
2. ✅ Alert on symptoms, not noise
3. ✅ Have clear severity levels
4. ✅ Automated incident routing
5. ✅ Structured logging
6. ✅ Retention policies in place
7. ✅ Regular drill exercises
8. ✅ Post-incident reviews

---

## Related Documents

- [API Specification](./api-specification.md) - Service design
- [Infrastructure](./infrastructure.md) - Production setup
- [Performance Metrics](../05-monitoring/performance-metrics.md) - Detailed metrics
