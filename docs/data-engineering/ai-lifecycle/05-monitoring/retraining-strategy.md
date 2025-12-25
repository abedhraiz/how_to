# Retraining Strategy

## Purpose

Plan and execute systematic model retraining to maintain performance as data and environment evolve.

## 1. Retraining Triggers

### Automatic Triggers

```python
class RetrainingTrigger:
    def __init__(self, config):
        self.config = config
    
    def should_retrain(self, current_metrics, drift_analysis, data_freshness):
        """Determine if retraining should be triggered"""
        
        triggers = {
            'performance_degradation': self._check_performance(current_metrics),
            'high_data_drift': self._check_drift(drift_analysis),
            'scheduled_time': self._check_schedule(data_freshness),
            'data_volume': self._check_volume(data_freshness),
        }
        
        # Trigger if any condition met
        should_retrain = any(triggers.values())
        
        if should_retrain:
            logger.info(f"Retraining triggered: {triggers}")
        
        return should_retrain
    
    def _check_performance(self, metrics):
        """Check if performance below threshold"""
        return metrics['accuracy'] < self.config['min_accuracy']
    
    def _check_drift(self, drift_analysis):
        """Check if high data drift detected"""
        high_drift_features = sum(
            1 for f in drift_analysis.values() 
            if f['severity'] == 'high'
        )
        return high_drift_features > self.config['max_drift_features']
    
    def _check_schedule(self, data_freshness):
        """Check if scheduled retraining time"""
        days_since_training = data_freshness['days_since_training']
        return days_since_training > self.config['retrain_frequency_days']
    
    def _check_volume(self, data_freshness):
        """Check if sufficient new data available"""
        new_samples = data_freshness['new_samples']
        return new_samples > self.config['min_new_samples']
```

### Trigger Rules

```yaml
retraining_rules:
  performance_based:
    trigger: accuracy_drops_below_0.85
    condition: accuracy < 0.85
    action: retrain
  
  scheduled:
    trigger: weekly_retraining
    condition: days_since_training > 7
    action: retrain
  
  data_based:
    trigger: new_data_volume
    condition: new_samples > 10000
    action: retrain
  
  drift_based:
    trigger: high_data_drift
    condition: drift_count > 5
    action: investigate_and_retrain
```

## 2. Retraining Pipeline

### Automated Pipeline

```python
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'model_retraining_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
)

def extract_new_data():
    """Extract new training data"""
    # Fetch data since last training
    new_data = fetch_data(since='last_training_date')
    new_data.to_parquet('data/new_training_data.parquet')

def preprocess_data():
    """Preprocess new training data"""
    # Apply same preprocessing as original
    preprocessor.fit_transform('data/new_training_data.parquet')

def train_model():
    """Train new model"""
    new_model = train(X_train, y_train)
    joblib.dump(new_model, 'models/candidate_model.pkl')

def evaluate_model():
    """Evaluate new model"""
    metrics = evaluate(new_model, X_test, y_test)
    if metrics['accuracy'] >= 0.85:
        return 'deploy'
    else:
        return 'reject'

def deploy_model(result):
    """Deploy if performance acceptable"""
    if result == 'deploy':
        deploy_to_staging(new_model)

# Define task dependencies
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_new_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

extract_task >> preprocess_task >> train_task >> evaluate_task
```

## 3. Model Comparison

### A/B Testing

```python
def run_ab_test(current_model, candidate_model, test_data, duration_days=7):
    """Run A/B test comparing models"""
    
    results = {
        'current_model_accuracy': [],
        'candidate_model_accuracy': [],
        'current_model_latency': [],
        'candidate_model_latency': [],
    }
    
    start_time = datetime.now()
    
    while (datetime.now() - start_time).days < duration_days:
        # Route traffic
        for sample in test_data:
            # 50% to each model
            if random.random() < 0.5:
                pred = current_model.predict(sample)
                results['current_model_accuracy'].append(pred)
            else:
                pred = candidate_model.predict(sample)
                results['candidate_model_accuracy'].append(pred)
    
    # Analyze results
    current_acc = accuracy_score(
        test_data['labels'],
        results['current_model_accuracy']
    )
    candidate_acc = accuracy_score(
        test_data['labels'],
        results['candidate_model_accuracy']
    )
    
    return {
        'current_accuracy': current_acc,
        'candidate_accuracy': candidate_acc,
        'winner': 'candidate' if candidate_acc > current_acc else 'current',
        'significant': is_statistically_significant(current_acc, candidate_acc),
    }
```

## 4. Model Versioning & Rollback

### Version Management

```python
def create_model_version(model, metrics, config):
    """Create and register new model version"""
    
    version = {
        'version_id': generate_version_id(),
        'timestamp': datetime.now().isoformat(),
        'algorithm': config['algorithm'],
        'training_data_version': config['data_version'],
        'metrics': metrics,
        'features': config['features'],
        'status': 'staging',  # staging -> canary -> production
    }
    
    # Register in model registry
    register_model_version(version)
    
    return version

def rollback_model(previous_version):
    """Rollback to previous model version"""
    
    # Switch traffic
    switch_to_version(previous_version)
    
    # Log incident
    logger.critical(f"Rolled back to {previous_version}")
    
    # Notify team
    notify_incident_channel(f"Rolled back model to {previous_version}")
```

## 5. Retraining Frequency

### Adaptive Retraining

```
Low Data Change + Stable Performance
  → Monthly retraining

Moderate Data Change + Some Drift
  → Weekly retraining

High Data Change + Performance Degradation
  → Daily retraining

Critical Performance Drop
  → Immediate retraining
```

### Schedule-Based Retraining

```
Business Hours Retraining
- Weekly: Monday morning
- Duration: 2-3 hours
- Parallel testing

Off-Hours Retraining
- Monthly: Sunday evening
- Full model rebuilds
- Comprehensive testing
```

## 6. Retraining Checklist

- [ ] New data collected and validated
- [ ] Preprocessing applied consistently
- [ ] Model trained on full dataset
- [ ] Performance metrics calculated
- [ ] Compared against baseline
- [ ] Performance acceptable (> threshold)
- [ ] New model passes tests
- [ ] Model versioned and registered
- [ ] Deployment approved
- [ ] Canary deployment in place
- [ ] Monitoring alerts configured

## 7. Retraining Documentation

### Retraining Log

```python
retraining_log = {
    'retraining_id': 'retrain_20240115_001',
    'trigger': 'scheduled_weekly',
    'start_time': '2024-01-15T02:00:00Z',
    'end_time': '2024-01-15T04:30:00Z',
    'training_duration_minutes': 150,
    'data_version': '2024-01-15',
    'new_samples': 50000,
    'model_version': 'v1.2.0',
    'metrics': {
        'accuracy': 0.913,
        'precision': 0.892,
        'recall': 0.875,
    },
    'baseline_metrics': {
        'accuracy': 0.910,
        'precision': 0.890,
        'recall': 0.873,
    },
    'status': 'deployed',
    'notes': 'Small improvement in accuracy and precision',
}
```

## Best Practices

1. ✅ Automate retraining process
2. ✅ Maintain multiple versions
3. ✅ Systematic testing
4. ✅ Quick rollback capability
5. ✅ Version control code
6. ✅ Document all changes
7. ✅ Consistent preprocessing
8. ✅ Monitor retraining performance

---

## Related Documents

- [Performance Metrics](./performance-metrics.md) - Metric tracking
- [Drift Detection](./drift-detection.md) - Detecting changes
- [Deployment Strategy](../04-deployment/deployment-strategy.md) - Rollout process
