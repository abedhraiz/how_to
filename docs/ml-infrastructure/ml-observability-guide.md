# ML Observability & Debugging Guide

## Introduction

ML observability goes beyond traditional application monitoring to track model performance, data quality, and ML-specific metrics in production, enabling rapid detection and debugging of model issues.

**What You'll Learn:**
- Model performance monitoring
- Data quality monitoring
- Model explainability in production
- Incident response for ML systems
- Debugging techniques

---

## Table of Contents

1. [ML Observability Stack](#ml-observability-stack)
2. [Model Performance Monitoring](#model-performance-monitoring)
3. [Data Quality Monitoring](#data-quality-monitoring)
4. [Model Explainability](#model-explainability)
5. [Incident Response](#incident-response)
6. [Debugging Playbooks](#debugging-playbooks)

---

## ML Observability Stack

```python
# Complete ML Observability Architecture

┌─────────────────────────────────────────────────────┐
│                ML Service (Inference)               │
│  Logs: Predictions, Latency, Errors                │
│  Metrics: Request rate, P95 latency                 │
│  Traces: Request flow                               │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│   Metrics     │    │   Prediction   │
│  (Prometheus) │    │   Logging      │
│               │    │   (S3/BigQuery)│
│ - QPS         │    │                │
│ - Latency     │    │ - Features     │
│ - Accuracy    │    │ - Predictions  │
│ - Drift       │    │ - Ground truth │
└───────┬───────┘    └────────┬───────┘
        │                     │
        └──────────┬──────────┘
                   ▼
        ┌────────────────────┐
        │  Monitoring System │
        │   (Grafana, etc)   │
        │                    │
        │ - Dashboards       │
        │ - Alerts           │
        │ - Drift detection  │
        └────────────────────┘
```

### Key Metrics to Track

```python
# ML Observability Metrics (4 Categories)

1. Model Performance Metrics:
   - Accuracy, Precision, Recall, F1
   - AUC-ROC, AUC-PR
   - MAE, RMSE (regression)
   - Custom business metrics

2. Data Quality Metrics:
   - Feature drift (distribution shift)
   - Missing value rate
   - Out-of-range values
   - Schema violations

3. System Performance Metrics:
   - Prediction latency (p50, p95, p99)
   - Throughput (QPS)
   - Error rate
   - Resource utilization

4. Business Metrics:
   - Revenue impact
   - User engagement
   - Conversion rate
   - Cost per prediction
```

---

## Model Performance Monitoring

### Real-Time Performance Tracking

```python
# model_monitoring.py - Track Model Performance

from prometheus_client import Counter, Histogram, Gauge, Summary
from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Define Prometheus metrics
PREDICTIONS_TOTAL = Counter(
    'ml_predictions_total',
    'Total number of predictions',
    ['model_name', 'model_version', 'prediction_class']
)

PREDICTION_LATENCY = Histogram(
    'ml_prediction_latency_seconds',
    'Prediction latency',
    ['model_name', 'model_version'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'ml_prediction_confidence',
    'Prediction confidence score',
    ['model_name', 'model_version'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

MODEL_ACCURACY = Gauge(
    'ml_model_accuracy',
    'Model accuracy (when ground truth available)',
    ['model_name', 'model_version', 'time_window']
)

FEATURE_VALUES = Summary(
    'ml_feature_value',
    'Feature value distribution',
    ['model_name', 'feature_name']
)

@dataclass
class PredictionLog:
    """Structure for logging predictions"""
    prediction_id: str
    model_name: str
    model_version: str
    timestamp: datetime
    features: Dict
    prediction: any
    confidence: float
    latency_ms: float
    ground_truth: Optional[any] = None

class ModelPerformanceMonitor:
    """Monitor model performance in production"""
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        log_backend: str = "s3"
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.log_backend = log_backend
        self.prediction_buffer: List[PredictionLog] = []
    
    def log_prediction(
        self,
        prediction_log: PredictionLog
    ):
        """Log a single prediction with metrics"""
        
        # Update Prometheus metrics
        PREDICTIONS_TOTAL.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_class=str(prediction_log.prediction)
        ).inc()
        
        PREDICTION_LATENCY.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(prediction_log.latency_ms / 1000)
        
        PREDICTION_CONFIDENCE.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(prediction_log.confidence)
        
        # Log feature values
        for feature_name, value in prediction_log.features.items():
            if isinstance(value, (int, float)):
                FEATURE_VALUES.labels(
                    model_name=self.model_name,
                    feature_name=feature_name
                ).observe(value)
        
        # Buffer for batch logging
        self.prediction_buffer.append(prediction_log)
        
        # Flush periodically
        if len(self.prediction_buffer) >= 1000:
            self.flush_logs()
    
    def update_accuracy(
        self,
        predictions: List,
        ground_truth: List,
        time_window: str = "1h"
    ):
        """Update accuracy metric when ground truth is available"""
        
        accuracy = np.mean(np.array(predictions) == np.array(ground_truth))
        
        MODEL_ACCURACY.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            time_window=time_window
        ).set(accuracy)
        
        logger.info(
            f"Model {self.model_name} v{self.model_version} "
            f"accuracy ({time_window}): {accuracy:.3f}"
        )
    
    def flush_logs(self):
        """Flush prediction logs to storage"""
        if not self.prediction_buffer:
            return
        
        # Convert to DataFrame for storage
        import pandas as pd
        
        logs_df = pd.DataFrame([
            {
                'prediction_id': log.prediction_id,
                'timestamp': log.timestamp,
                'prediction': log.prediction,
                'confidence': log.confidence,
                'latency_ms': log.latency_ms,
                **{f'feature_{k}': v for k, v in log.features.items()},
                'ground_truth': log.ground_truth
            }
            for log in self.prediction_buffer
        ])
        
        # Save to S3/BigQuery/etc
        self._save_logs(logs_df)
        
        # Clear buffer
        self.prediction_buffer.clear()
    
    def _save_logs(self, logs_df):
        """Save logs to backend storage"""
        if self.log_backend == "s3":
            # Save to S3
            import boto3
            from io import BytesIO
            
            s3 = boto3.client('s3')
            buffer = BytesIO()
            logs_df.to_parquet(buffer, index=False)
            
            s3.put_object(
                Bucket='ml-predictions',
                Key=f'logs/{self.model_name}/{datetime.now().isoformat()}.parquet',
                Body=buffer.getvalue()
            )

# Usage
monitor = ModelPerformanceMonitor(
    model_name="fraud_detector",
    model_version="v1.2.0"
)

# Log prediction
prediction_log = PredictionLog(
    prediction_id="pred_12345",
    model_name="fraud_detector",
    model_version="v1.2.0",
    timestamp=datetime.now(),
    features={
        'amount': 125.50,
        'merchant_id': 'MERCH_001',
        'transaction_count_24h': 3
    },
    prediction=0,  # Not fraud
    confidence=0.95,
    latency_ms=45.3
)

monitor.log_prediction(prediction_log)
```

### Performance Degradation Detection

```python
# performance_degradation.py - Detect Model Degradation

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats

@dataclass
class PerformanceAlert:
    """Alert for performance degradation"""
    alert_type: str
    severity: str  # 'warning', 'critical'
    metric_name: str
    current_value: float
    baseline_value: float
    change_percent: float
    timestamp: datetime
    details: Dict

class PerformanceDegradationDetector:
    """Detect model performance degradation"""
    
    def __init__(
        self,
        baseline_window: int = 7,  # days
        alert_threshold: float = 0.05  # 5% degradation
    ):
        self.baseline_window = baseline_window
        self.alert_threshold = alert_threshold
        self.baseline_metrics: Dict = {}
    
    def compute_baseline(
        self,
        historical_metrics: pd.DataFrame
    ):
        """Compute baseline metrics from historical data"""
        
        cutoff_date = datetime.now() - timedelta(days=self.baseline_window)
        baseline_data = historical_metrics[
            historical_metrics['timestamp'] >= cutoff_date
        ]
        
        self.baseline_metrics = {
            'accuracy': baseline_data['accuracy'].mean(),
            'precision': baseline_data['precision'].mean(),
            'recall': baseline_data['recall'].mean(),
            'f1_score': baseline_data['f1_score'].mean(),
            'auc': baseline_data['auc'].mean(),
            'latency_p95': baseline_data['latency_ms'].quantile(0.95)
        }
    
    def check_for_degradation(
        self,
        current_metrics: Dict
    ) -> List[PerformanceAlert]:
        """Check if current metrics show degradation"""
        
        alerts = []
        
        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name not in current_metrics:
                continue
            
            current_value = current_metrics[metric_name]
            
            # Calculate change
            change = (current_value - baseline_value) / baseline_value
            
            # Check for degradation (lower is worse, except latency)
            is_degraded = False
            if metric_name == 'latency_p95':
                # Higher latency is worse
                is_degraded = change > self.alert_threshold
            else:
                # Lower performance is worse
                is_degraded = change < -self.alert_threshold
            
            if is_degraded:
                severity = 'critical' if abs(change) > 0.10 else 'warning'
                
                alerts.append(PerformanceAlert(
                    alert_type='performance_degradation',
                    severity=severity,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    change_percent=change * 100,
                    timestamp=datetime.now(),
                    details={
                        'baseline_window_days': self.baseline_window,
                        'threshold': self.alert_threshold
                    }
                ))
        
        return alerts
    
    def perform_statistical_test(
        self,
        baseline_data: List[float],
        current_data: List[float]
    ) -> Tuple[bool, float]:
        """Statistical test for significant change"""
        
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(
            baseline_data,
            current_data,
            alternative='two-sided'
        )
        
        is_significant = p_value < 0.05
        
        return is_significant, p_value

# Usage
detector = PerformanceDegradationDetector(
    baseline_window=7,
    alert_threshold=0.05
)

# Compute baseline from historical data
historical_df = pd.read_csv('metrics/historical.csv')
detector.compute_baseline(historical_df)

# Check current performance
current_metrics = {
    'accuracy': 0.82,  # Was 0.87 (5% drop)
    'precision': 0.80,
    'recall': 0.75,
    'latency_p95': 150  # ms
}

alerts = detector.check_for_degradation(current_metrics)

for alert in alerts:
    print(f"⚠️ {alert.severity.upper()}: {alert.metric_name}")
    print(f"   Current: {alert.current_value:.3f}")
    print(f"   Baseline: {alert.baseline_value:.3f}")
    print(f"   Change: {alert.change_percent:.1f}%")
```

---

## Data Quality Monitoring

### Feature Drift Detection

```python
# drift_detection.py - Detect Feature Drift

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DriftAlert:
    """Alert for detected drift"""
    feature_name: str
    drift_score: float
    p_value: float
    drift_type: str  # 'mean', 'distribution', 'categorical'
    timestamp: datetime
    baseline_stats: Dict
    current_stats: Dict

class FeatureDriftDetector:
    """Detect distribution drift in features"""
    
    def __init__(self, sensitivity: float = 0.05):
        self.sensitivity = sensitivity  # p-value threshold
        self.baseline_distributions: Dict = {}
    
    def fit(self, baseline_data: pd.DataFrame):
        """Fit on baseline/training data"""
        
        for column in baseline_data.columns:
            if pd.api.types.is_numeric_dtype(baseline_data[column]):
                self.baseline_distributions[column] = {
                    'type': 'numeric',
                    'mean': baseline_data[column].mean(),
                    'std': baseline_data[column].std(),
                    'quantiles': baseline_data[column].quantile([0.25, 0.5, 0.75]).to_dict(),
                    'data': baseline_data[column].dropna().values
                }
            else:
                self.baseline_distributions[column] = {
                    'type': 'categorical',
                    'value_counts': baseline_data[column].value_counts(normalize=True).to_dict(),
                    'data': baseline_data[column].dropna().values
                }
    
    def detect_drift(
        self,
        current_data: pd.DataFrame
    ) -> List[DriftAlert]:
        """Detect drift in current data"""
        
        alerts = []
        
        for column, baseline_dist in self.baseline_distributions.items():
            if column not in current_data.columns:
                continue
            
            current_values = current_data[column].dropna()
            
            if baseline_dist['type'] == 'numeric':
                alert = self._check_numeric_drift(
                    column,
                    baseline_dist,
                    current_values
                )
            else:
                alert = self._check_categorical_drift(
                    column,
                    baseline_dist,
                    current_values
                )
            
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _check_numeric_drift(
        self,
        feature_name: str,
        baseline_dist: Dict,
        current_values: pd.Series
    ) -> Optional[DriftAlert]:
        """Check drift in numeric feature"""
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(
            baseline_dist['data'],
            current_values.values
        )
        
        if p_value < self.sensitivity:
            current_stats = {
                'mean': current_values.mean(),
                'std': current_values.std(),
                'quantiles': current_values.quantile([0.25, 0.5, 0.75]).to_dict()
            }
            
            return DriftAlert(
                feature_name=feature_name,
                drift_score=ks_statistic,
                p_value=p_value,
                drift_type='distribution',
                timestamp=datetime.now(),
                baseline_stats={
                    'mean': baseline_dist['mean'],
                    'std': baseline_dist['std']
                },
                current_stats=current_stats
            )
        
        return None
    
    def _check_categorical_drift(
        self,
        feature_name: str,
        baseline_dist: Dict,
        current_values: pd.Series
    ) -> Optional[DriftAlert]:
        """Check drift in categorical feature"""
        
        # Chi-square test
        baseline_counts = baseline_dist['value_counts']
        current_counts = current_values.value_counts(normalize=True).to_dict()
        
        # Get all categories
        all_categories = set(baseline_counts.keys()) | set(current_counts.keys())
        
        # Build frequency arrays
        baseline_freq = [baseline_counts.get(cat, 0) for cat in all_categories]
        current_freq = [current_counts.get(cat, 0) for cat in all_categories]
        
        # Chi-square test
        chi2_statistic, p_value = stats.chisquare(
            f_obs=[f * len(current_values) for f in current_freq],
            f_exp=[f * len(current_values) for f in baseline_freq]
        )
        
        if p_value < self.sensitivity:
            return DriftAlert(
                feature_name=feature_name,
                drift_score=chi2_statistic,
                p_value=p_value,
                drift_type='categorical',
                timestamp=datetime.now(),
                baseline_stats={'distribution': baseline_counts},
                current_stats={'distribution': current_counts}
            )
        
        return None

# Usage
detector = FeatureDriftDetector(sensitivity=0.05)

# Fit on training data
training_data = pd.read_csv('data/training.csv')
detector.fit(training_data)

# Check production data
production_data = pd.read_csv('data/production_last_hour.csv')
drift_alerts = detector.detect_drift(production_data)

for alert in drift_alerts:
    print(f"🚨 Drift detected in '{alert.feature_name}'")
    print(f"   Type: {alert.drift_type}")
    print(f"   Drift score: {alert.drift_score:.4f}")
    print(f"   P-value: {alert.p_value:.4f}")
    
    if alert.drift_type == 'distribution':
        baseline_mean = alert.baseline_stats['mean']
        current_mean = alert.current_stats['mean']
        print(f"   Mean shift: {baseline_mean:.2f} → {current_mean:.2f}")
```

---

## Model Explainability

### SHAP Values in Production

```python
# explainability.py - Model Explainability

import shap
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import io
import base64

class ModelExplainer:
    """Explain model predictions using SHAP"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def fit(self, background_data: np.ndarray, method: str = "kernel"):
        """Initialize explainer with background data"""
        
        if method == "kernel":
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background_data
            )
        elif method == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def explain_prediction(
        self,
        input_data: np.ndarray
    ) -> Dict:
        """Explain a single prediction"""
        
        # Get SHAP values
        shap_values = self.explainer.shap_values(input_data)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Get feature contributions
        contributions = []
        for i, feature_name in enumerate(self.feature_names):
            contributions.append({
                'feature': feature_name,
                'value': float(input_data[0][i]),
                'contribution': float(shap_values[0][i])
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return {
            'top_features': contributions[:5],
            'all_contributions': contributions,
            'base_value': float(self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value)
        }
    
    def generate_explanation_plot(
        self,
        input_data: np.ndarray,
        prediction: float
    ) -> str:
        """Generate waterfall plot as base64 image"""
        
        shap_values = self.explainer.shap_values(input_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Create waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value[1],
                data=input_data[0],
                feature_names=self.feature_names
            ),
            show=False
        )
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        return img_base64
    
    def compute_global_importance(
        self,
        data: np.ndarray
    ) -> Dict[str, float]:
        """Compute global feature importance"""
        
        shap_values = self.explainer.shap_values(data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Mean absolute SHAP value per feature
        importance = np.abs(shap_values).mean(axis=0)
        
        return dict(zip(self.feature_names, importance))

# Usage
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize explainer
explainer = ModelExplainer(
    model=model,
    feature_names=['age', 'income', 'credit_score', 'num_transactions']
)

# Fit on background data
explainer.fit(X_train[:100], method="tree")

# Explain prediction
explanation = explainer.explain_prediction(X_test[:1])

print("Top contributing features:")
for contrib in explanation['top_features']:
    print(f"  {contrib['feature']}: {contrib['contribution']:.3f}")
    print(f"    (value: {contrib['value']:.2f})")
```

---

## Incident Response

### ML Incident Playbook

```python
# incident_response.py - ML Incident Response

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

class IncidentSeverity(Enum):
    SEV1 = "critical"  # Model down, major degradation
    SEV2 = "high"      # Significant performance drop
    SEV3 = "medium"    # Minor degradation
    SEV4 = "low"       # Potential issue

@dataclass
class MLIncident:
    """ML system incident"""
    incident_id: str
    severity: IncidentSeverity
    title: str
    description: str
    detected_at: datetime
    affected_model: str
    symptoms: List[str]
    metrics: Dict
    resolved_at: Optional[datetime] = None
    root_cause: Optional[str] = None
    resolution: Optional[str] = None

class MLIncidentPlaybook:
    """Incident response playbook for ML systems"""
    
    @staticmethod
    def diagnose_performance_drop(
        model_name: str,
        current_accuracy: float,
        baseline_accuracy: float,
        recent_predictions: pd.DataFrame
    ) -> Dict:
        """Diagnose performance drop"""
        
        diagnostics = {
            'accuracy_drop': baseline_accuracy - current_accuracy,
            'predictions_analyzed': len(recent_predictions),
            'potential_causes': []
        }
        
        # Check 1: Data drift
        # (Simplified - use FeatureDriftDetector in production)
        if 'feature_mean_shift' in recent_predictions:
            diagnostics['potential_causes'].append(
                "Feature drift detected - input distribution changed"
            )
        
        # Check 2: Class imbalance shift
        pred_distribution = recent_predictions['prediction'].value_counts(normalize=True)
        if pred_distribution.max() > 0.95:
            diagnostics['potential_causes'].append(
                "Model predicting mostly one class - possible data quality issue"
            )
        
        # Check 3: Low confidence predictions
        if 'confidence' in recent_predictions:
            low_confidence_rate = (recent_predictions['confidence'] < 0.6).mean()
            if low_confidence_rate > 0.3:
                diagnostics['potential_causes'].append(
                    f"High rate of low-confidence predictions ({low_confidence_rate:.1%})"
                )
        
        # Check 4: Feature missing rate
        missing_features = recent_predictions.isnull().sum() / len(recent_predictions)
        if missing_features.any() > 0.1:
            diagnostics['potential_causes'].append(
                f"High missing feature rate: {missing_features[missing_features > 0.1].to_dict()}"
            )
        
        return diagnostics
    
    @staticmethod
    def rollback_procedure(
        current_model_version: str,
        previous_model_version: str
    ) -> List[str]:
        """Generate rollback procedure"""
        
        steps = [
            f"1. Verify previous model ({previous_model_version}) availability",
            f"2. Update model registry to point to {previous_model_version}",
            "3. Restart inference service with new model",
            "4. Verify rollback with health checks",
            "5. Monitor metrics for 15 minutes",
            f"6. If successful, mark {current_model_version} as deprecated",
            "7. Notify team and stakeholders",
            "8. Schedule root cause analysis meeting"
        ]
        
        return steps
    
    @staticmethod
    def mitigation_strategies() -> Dict[str, List[str]]:
        """Common mitigation strategies"""
        
        return {
            "performance_degradation": [
                "Rollback to previous model version",
                "Increase prediction confidence threshold",
                "Enable human review for low-confidence predictions",
                "Switch to champion-challenger with traffic split"
            ],
            "data_drift": [
                "Retrain model with recent data",
                "Update feature engineering pipeline",
                "Enable drift detection alerts",
                "Implement adaptive learning"
            ],
            "high_latency": [
                "Scale up inference service",
                "Enable request batching",
                "Use model distillation/quantization",
                "Add caching layer"
            ],
            "data_quality_issue": [
                "Enable upstream data validation",
                "Implement feature imputation",
                "Add data quality monitoring",
                "Set up data pipeline alerts"
            ]
        }

# Usage
playbook = MLIncidentPlaybook()

# Diagnose issue
recent_data = pd.read_csv('logs/recent_predictions.csv')
diagnosis = playbook.diagnose_performance_drop(
    model_name="fraud_detector",
    current_accuracy=0.82,
    baseline_accuracy=0.89,
    recent_predictions=recent_data
)

print("Diagnosis:")
print(f"Accuracy drop: {diagnosis['accuracy_drop']:.2%}")
print("\nPotential causes:")
for cause in diagnosis['potential_causes']:
    print(f"  - {cause}")

# Get rollback steps
if diagnosis['accuracy_drop'] > 0.05:
    print("\n🚨 Severe degradation - initiating rollback")
    steps = playbook.rollback_procedure("v1.5.0", "v1.4.0")
    for step in steps:
        print(step)
```

---

## Debugging Playbooks

### Common Issues and Solutions

```python
# debugging_playbooks.md

"""
ML Production Debugging Playbooks
==================================

Issue: Model Accuracy Suddenly Drops
------------------------------------
Symptoms:
- Accuracy drops > 5% in production
- More false positives/negatives

Investigation Steps:
1. Check for data drift
   → Run FeatureDriftDetector on recent data
   → Compare feature distributions

2. Verify data quality
   → Check for missing values
   → Verify feature ranges
   → Look for schema changes

3. Check model serving
   → Verify correct model version loaded
   → Check preprocessing pipeline
   → Validate feature engineering

4. Analyze predictions
   → Look at low-confidence predictions
   → Check class distribution
   → Review error cases

Resolution:
- If drift: Retrain with recent data
- If data quality: Fix upstream pipeline
- If serving issue: Rollback to previous version


Issue: Prediction Latency Spike
-------------------------------
Symptoms:
- P95 latency > 200ms (was 50ms)
- Timeouts increasing

Investigation Steps:
1. Check system resources
   → CPU/GPU utilization
   → Memory usage
   → Network latency

2. Profile inference code
   → Time each pipeline stage
   → Identify bottleneck

3. Check batch sizes
   → Verify dynamic batching working
   → Look for single-request processing

4. Review recent changes
   → New model version?
   → Feature pipeline changes?
   → Infrastructure changes?

Resolution:
- If resource: Scale up instances
- If batching: Fix batch configuration
- If model: Use quantization/distillation
- If feature: Optimize feature computation


Issue: Model Serving Errors
---------------------------
Symptoms:
- HTTP 500 errors
- Model loading failures
- Prediction exceptions

Investigation Steps:
1. Check logs
   → Look for stack traces
   → Identify error patterns

2. Verify model compatibility
   → Check model format
   → Verify dependencies
   → Test model locally

3. Check input validation
   → Review failed requests
   → Validate feature schemas

4. Test with sample data
   → Use known-good inputs
   → Isolate the issue

Resolution:
- If model issue: Rollback
- If input issue: Add validation
- If dependency: Update/fix dependencies


Issue: Feature Drift Detected
-----------------------------
Symptoms:
- Statistical drift alerts
- Distribution shifts
- New categories appearing

Investigation Steps:
1. Identify drifted features
   → Run drift detection
   → Visualize distributions

2. Determine root cause
   → Upstream data changes?
   → Seasonal patterns?
   → External factors?

3. Assess impact
   → How much does accuracy drop?
   → Which segments affected?

4. Validate with domain experts
   → Is drift expected?
   → Business logic changes?

Resolution:
- If expected: Update baseline
- If unexpected: Retrain model
- If critical: Enable human review
- Monitor closely


Issue: Memory Leak
-----------------
Symptoms:
- Memory usage steadily increasing
- OOM errors
- Service crashes

Investigation Steps:
1. Profile memory usage
   → Use memory profiler
   → Track over time

2. Check for leaks
   → Unreleased model copies?
   → Growing prediction buffers?
   → Cache not clearing?

3. Review recent changes
   → New features added?
   → Logging changes?

Resolution:
- Add memory limits
- Implement proper cleanup
- Fix buffer management
- Restart service periodically
"""
```

---

## Grafana Dashboard Example

```python
# grafana_dashboard.json - ML Monitoring Dashboard

{
  "dashboard": {
    "title": "ML Model Monitoring",
    "panels": [
      {
        "title": "Prediction Rate (QPS)",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ml_prediction_latency_seconds_bucket)",
            "legendFormat": "p95"
          }
        ]
      },
      {
        "title": "Model Accuracy (Hourly)",
        "targets": [
          {
            "expr": "ml_model_accuracy{time_window=\"1h\"}",
            "legendFormat": "{{model_version}}"
          }
        ]
      },
      {
        "title": "Prediction Confidence Distribution",
        "targets": [
          {
            "expr": "ml_prediction_confidence_bucket",
            "legendFormat": "confidence"
          }
        ]
      },
      {
        "title": "Feature Drift Alerts",
        "targets": [
          {
            "expr": "ml_feature_drift_detected",
            "legendFormat": "{{feature_name}}"
          }
        ]
      }
    ]
  }
}
```

---

## Production Checklist

```markdown
✅ Metrics Collection
  □ Prediction logging enabled
  □ Performance metrics tracked
  □ Latency monitored (p50, p95, p99)
  □ Error rates tracked

✅ Model Monitoring
  □ Accuracy tracking (when ground truth available)
  □ Confidence distribution monitored
  □ Prediction distribution tracked
  □ Performance degradation alerts

✅ Data Monitoring
  □ Feature drift detection
  □ Data quality checks
  □ Missing value tracking
  □ Schema validation

✅ Explainability
  □ SHAP/LIME integration
  □ Feature importance tracking
  □ Prediction explanations available
  □ Audit trail maintained

✅ Alerting
  □ Performance degradation alerts
  □ Drift detection alerts
  □ Latency alerts
  □ Error rate alerts
  □ On-call rotation defined

✅ Incident Response
  □ Playbooks documented
  □ Rollback procedure tested
  □ Root cause analysis process
  □ Post-mortem template
```

---

*This guide covers ML observability. For complete production ML, see [Model Serving Guide](model-serving-guide.md) and [ML Testing Guide](../ml-engineering/ml-testing-guide.md).*
