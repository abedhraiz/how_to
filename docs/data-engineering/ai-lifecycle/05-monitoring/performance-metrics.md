# Performance Metrics

## Purpose

Define, track, and analyze key performance indicators for models in production.

## 1. Classification Metrics

### Core Metrics

```
Accuracy: (TP + TN) / (TP + TN + FP + FN)
  - Proportion of correct predictions
  - Good for balanced classes
  - Target: > 0.90

Precision: TP / (TP + FP)
  - Of positive predictions, how many are correct
  - Minimize false positives
  - Target: > 0.85

Recall: TP / (TP + FN)
  - Of actual positives, how many we caught
  - Minimize false negatives
  - Target: > 0.85

F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
  - Harmonic mean of precision and recall
  - Good balance metric
  - Target: > 0.85

AUC-ROC: Area under the ROC curve
  - Probability that model ranks random positive higher than random negative
  - Good for imbalanced data
  - Target: > 0.85
```

## 2. Regression Metrics

```
MAE: Mean Absolute Error
  - Average absolute difference from actual
  - Robust to outliers
  - Unit: Same as target

RMSE: Root Mean Square Error
  - Penalizes larger errors more
  - Sensitive to outliers
  - Unit: Same as target

MAPE: Mean Absolute Percentage Error
  - Percentage-based error
  - Good for scale-independent comparison
  - Target: < 10%

R²: Coefficient of Determination
  - Proportion of variance explained
  - Range: -∞ to 1 (higher is better)
  - Target: > 0.85
```

## 3. Metric Tracking Implementation

```python
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score

class MetricsTracker:
    def __init__(self, metrics_file='metrics.csv'):
        self.metrics_file = metrics_file
        self.metrics = []
    
    def log_metrics(self, predictions, actuals, metadata=None):
        """Log metrics for a batch of predictions"""
        
        metrics = {
            'timestamp': datetime.now(),
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, average='weighted'),
            'recall': recall_score(actuals, predictions, average='weighted'),
            'sample_count': len(predictions),
        }
        
        if metadata:
            metrics.update(metadata)
        
        self.metrics.append(metrics)
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to file"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_file, index=False)
    
    def get_trend(self, metric_name, window=100):
        """Get recent trend for a metric"""
        df = pd.read_csv(self.metrics_file)
        recent = df[metric_name].tail(window)
        return {
            'mean': recent.mean(),
            'std': recent.std(),
            'min': recent.min(),
            'max': recent.max(),
            'trend': 'increasing' if recent.iloc[-1] > recent.iloc[0] else 'decreasing'
        }
```

## 4. Metric Dashboards

### Real-Time Dashboard

```
Model Performance Dashboard

┌────────────────────────────┐
│ Accuracy (Last 24h)        │
│ Current: 0.920             │
│ Target:  0.900             │
│ Status:  ✓ On Target       │
└────────────────────────────┘

┌────────────────────────────┐
│ Precision vs Recall        │
│ Precision: 0.85            │
│ Recall:    0.88            │
│ F1-Score:  0.86            │
└────────────────────────────┘

┌────────────────────────────┐
│ Confidence Distribution    │
│ Mean: 0.78                 │
│ Std:  0.12                 │
└────────────────────────────┘
```

## 5. Alerting Rules

### Performance Alerts

```yaml
alert_rules:
  - name: accuracy_below_threshold
    condition: accuracy < 0.85
    duration: 1h
    severity: critical
    action: page_on_call
  
  - name: precision_dropping
    condition: precision < 0.80
    duration: 30m
    severity: high
    action: create_incident
  
  - name: error_rate_high
    condition: error_rate > 0.05
    duration: 5m
    severity: critical
    action: page_on_call
```

## 6. Metric Aggregation

### Time-Based Aggregation

```python
def aggregate_metrics_daily(metrics_df):
    """Aggregate metrics to daily level"""
    metrics_df['date'] = pd.to_datetime(metrics_df['timestamp']).dt.date
    
    daily = metrics_df.groupby('date').agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'sample_count': 'sum',
        'error_count': 'sum',
    })
    
    daily['error_rate'] = daily['error_count'] / daily['sample_count']
    return daily

# Track trends
daily_metrics = aggregate_metrics_daily(metrics_df)
accuracy_trend = daily_metrics['accuracy'].tail(30)  # Last 30 days
```

### Cohort-Based Aggregation

```python
def aggregate_by_cohort(metrics_df, cohort_column):
    """Aggregate metrics by user/segment cohort"""
    
    cohorts = metrics_df.groupby(cohort_column).agg({
        'accuracy': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'sample_count': 'sum',
    })
    
    # Identify underperforming cohorts
    low_performers = cohorts[cohorts['accuracy'] < 0.80]
    return cohorts, low_performers
```

## 7. Comparative Analysis

### vs. Baseline

```python
def compare_to_baseline(current_metrics, baseline_metrics):
    """Compare current performance to baseline"""
    
    comparison = {}
    
    for metric in baseline_metrics.keys():
        current = current_metrics.get(metric)
        baseline = baseline_metrics[metric]
        diff = current - baseline
        diff_pct = (diff / baseline) * 100
        
        comparison[metric] = {
            'current': current,
            'baseline': baseline,
            'difference': diff,
            'difference_pct': diff_pct,
            'status': '✓' if diff > 0 else '✗'
        }
    
    return comparison
```

### vs. Previous Model Version

```python
def compare_model_versions(v1_metrics, v2_metrics):
    """Compare two model versions"""
    
    comparison = {}
    
    for metric in v1_metrics.keys():
        v1 = v1_metrics[metric]
        v2 = v2_metrics[metric]
        improvement = v2 - v1
        
        comparison[metric] = {
            'v1': v1,
            'v2': v2,
            'improvement': improvement,
            'winner': 'v2' if improvement > 0 else 'v1'
        }
    
    return comparison
```

## 8. Metric Visualization

### Accuracy Trend

```python
import matplotlib.pyplot as plt
import pandas as pd

daily_metrics = pd.read_csv('daily_metrics.csv')

plt.figure(figsize=(12, 6))
plt.plot(daily_metrics['date'], daily_metrics['accuracy'], marker='o', label='Accuracy')
plt.axhline(y=0.90, color='r', linestyle='--', label='Target')
plt.fill_between(daily_metrics['date'], 0.88, 0.92, alpha=0.2, label='Acceptable Range')
plt.xlabel('Date')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Trend')
plt.legend()
plt.xticks(rotation=45)
plt.show()
```

## 9. Metric Documentation

### Metrics Definition Table

| Metric | Definition | Formula | Target | Alert Threshold |
|--------|-----------|---------|--------|-----------------|
| Accuracy | Correct / Total | (TP+TN)/(TP+TN+FP+FN) | >0.90 | <0.85 |
| Precision | Correct Positives / Predicted Positives | TP/(TP+FP) | >0.85 | <0.80 |
| Recall | Correct Positives / Actual Positives | TP/(TP+FN) | >0.85 | <0.80 |
| F1-Score | Harmonic mean | 2*(P*R)/(P+R) | >0.85 | <0.80 |

## 10. Best Practices

1. ✅ Track multiple metrics
2. ✅ Compare against baselines
3. ✅ Visualize trends
4. ✅ Alert on anomalies
5. ✅ Document all metrics
6. ✅ Aggregate appropriately
7. ✅ Review regularly
8. ✅ Store historical data

---

## Related Documents

- [Drift Detection](./drift-detection.md) - Detecting distribution changes
- [Retraining Strategy](./retraining-strategy.md) - Model updates
- [Monitoring Plan](../04-deployment/monitoring-plan.md) - Infrastructure monitoring
