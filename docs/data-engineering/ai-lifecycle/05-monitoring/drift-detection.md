# Drift Detection

## Purpose

Identify when data distribution or model performance changes, triggering retraining or investigation.

## 1. Data Drift Detection

### Statistical Tests

```python
from scipy import stats
import numpy as np

class DataDriftDetector:
    def __init__(self, baseline_data, threshold=0.05):
        self.baseline_data = baseline_data
        self.threshold = threshold
    
    def detect_drift_ks_test(self, current_data):
        """Kolmogorov-Smirnov test for distribution change"""
        drift_results = {}
        
        for feature in self.baseline_data.columns:
            baseline = self.baseline_data[feature].dropna()
            current = current_data[feature].dropna()
            
            statistic, p_value = stats.ks_2samp(baseline, current)
            
            drift_results[feature] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_drifted': p_value < self.threshold,
                'severity': self._classify_severity(statistic)
            }
        
        return drift_results
    
    def detect_drift_chi_square(self, current_data):
        """Chi-square test for categorical features"""
        # For categorical features
        pass
    
    def _classify_severity(self, statistic):
        if statistic < 0.1:
            return 'low'
        elif statistic < 0.25:
            return 'medium'
        else:
            return 'high'
```

### Distribution Comparison

```python
def compare_distributions(baseline, current, feature):
    """Visual comparison of distributions"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    ax1.hist(baseline[feature], bins=30, alpha=0.5, label='Baseline')
    ax1.hist(current[feature], bins=30, alpha=0.5, label='Current')
    ax1.set_xlabel(feature)
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Statistical summary
    ax2.text(0.1, 0.9, f'Baseline Mean: {baseline[feature].mean():.3f}')
    ax2.text(0.1, 0.8, f'Current Mean: {current[feature].mean():.3f}')
    ax2.text(0.1, 0.7, f'Baseline Std: {baseline[feature].std():.3f}')
    ax2.text(0.1, 0.6, f'Current Std: {current[feature].std():.3f}')
    ax2.axis('off')
    
    plt.show()
```

## 2. Model Drift Detection

### Performance-Based Drift

```python
class ModelDriftDetector:
    def __init__(self, baseline_metrics, threshold=0.05):
        self.baseline_metrics = baseline_metrics
        self.threshold = threshold
    
    def detect_performance_drift(self, current_metrics):
        """Check if model performance has degraded"""
        drift_alerts = []
        
        for metric, baseline_value in self.baseline_metrics.items():
            current_value = current_metrics.get(metric)
            
            if current_value is None:
                continue
            
            # Calculate percentage change
            pct_change = (baseline_value - current_value) / baseline_value
            
            if pct_change > self.threshold:
                drift_alerts.append({
                    'metric': metric,
                    'baseline': baseline_value,
                    'current': current_value,
                    'change': pct_change,
                    'severity': 'high' if pct_change > 0.1 else 'medium'
                })
        
        return drift_alerts
```

## 3. Prediction Drift

### Output Distribution Change

```python
def detect_prediction_drift(baseline_predictions, current_predictions):
    """Detect if prediction distribution has changed"""
    
    # For classification: check class distribution
    baseline_dist = pd.Series(baseline_predictions).value_counts(normalize=True)
    current_dist = pd.Series(current_predictions).value_counts(normalize=True)
    
    # Chi-square test
    from scipy.stats import chisquare
    f_obs = current_dist.values
    f_exp = baseline_dist.values
    
    # Align indices
    all_classes = set(baseline_dist.index) | set(current_dist.index)
    f_exp = np.array([baseline_dist.get(c, 0) for c in all_classes])
    f_obs = np.array([current_dist.get(c, 0) for c in all_classes])
    
    stat, p_value = chisquare(f_obs, f_exp)
    
    return {
        'test': 'chi_square',
        'statistic': stat,
        'p_value': p_value,
        'is_drifted': p_value < 0.05,
        'baseline_dist': baseline_dist,
        'current_dist': current_dist
    }
```

## 4. Confidence Score Monitoring

```python
def analyze_confidence_distribution(predictions_with_confidence):
    """Monitor model confidence levels"""
    
    confidences = [pred['confidence'] for pred in predictions_with_confidence]
    
    analysis = {
        'mean_confidence': np.mean(confidences),
        'std_confidence': np.std(confidences),
        'low_confidence_count': sum(1 for c in confidences if c < 0.5),
        'low_confidence_pct': sum(1 for c in confidences if c < 0.5) / len(confidences),
    }
    
    # Alert if many low-confidence predictions
    if analysis['low_confidence_pct'] > 0.2:
        logger.warning(f"High proportion of low-confidence predictions: {analysis['low_confidence_pct']*100:.1f}%")
    
    return analysis
```

## 5. Drift Monitoring Dashboard

```
Data Drift Monitor

┌──────────────────────────────────────┐
│ Feature Drift Status                 │
├──────────────────────────────────────┤
│                                      │
│ age:           ✓ No drift            │
│ income:        ⚠ Low drift detected  │
│ credit_score:  ✓ No drift            │
│ tenure:        🔴 High drift detected│
│                                      │
└──────────────────────────────────────┘

Model Performance Drift

┌──────────────────────────────────────┐
│ Accuracy:  92.0% → 88.5% ↓ (3.5%)   │
│ Precision: 85.0% → 82.1% ↓ (2.9%)   │
│ Recall:    87.5% → 85.3% ↓ (2.2%)   │
│                                      │
│ Status: ⚠ Performance degradation    │
└──────────────────────────────────────┘
```

## 6. Drift Response

### Automated Response

```python
def respond_to_drift(drift_detection_result):
    """Automated response to detected drift"""
    
    if drift_detection_result['severity'] == 'high':
        # Critical drift - immediate action
        logger.critical("High drift detected, triggering retraining")
        trigger_retraining_job(priority='high')
        notify_team('critical_drift')
    
    elif drift_detection_result['severity'] == 'medium':
        # Medium drift - schedule investigation
        logger.warning("Medium drift detected, scheduling investigation")
        schedule_investigation_task()
        notify_team('medium_drift')
    
    else:
        # Low drift - monitor closely
        logger.info("Low drift detected, increasing monitoring frequency")
```

### Investigation Process

```
1. Detect Drift
   ↓
2. Classify Severity (Low/Medium/High)
   ↓
3. If Critical: Immediate Response
   - Notify on-call engineer
   - Prepare rollback
   - Investigate root cause
   ↓
4. If Medium: Scheduled Investigation
   - Analyze data characteristics
   - Compare feature distributions
   - Review recent data changes
   ↓
5. Decision Point
   - Root cause identified?
     → Retrain model
     → Deploy new version
   - No clear cause?
     → Continue monitoring
     → Increase data collection
```

## 7. Drift Documentation

### Drift Log Template

```python
drift_log = {
    'timestamp': '2024-01-15T10:30:00Z',
    'detection_method': 'ks_test',
    'drifted_features': ['income', 'tenure'],
    'severity': 'medium',
    'p_values': {
        'income': 0.02,
        'tenure': 0.04,
    },
    'action_taken': 'scheduled_retraining',
    'notes': 'Possible due to economic downturn affecting income distribution',
}
```

## Best Practices

1. ✅ Monitor multiple dimensions
2. ✅ Establish baselines early
3. ✅ Use statistical tests
4. ✅ Visualize distributions
5. ✅ Automated alerting
6. ✅ Document drift events
7. ✅ Regular review
8. ✅ Proactive response

---

## Related Documents

- [Performance Metrics](./performance-metrics.md) - Metric tracking
- [Retraining Strategy](./retraining-strategy.md) - Model updates
