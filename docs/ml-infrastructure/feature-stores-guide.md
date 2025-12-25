# Feature Stores Production Guide

## Introduction

Feature stores provide centralized storage and management of machine learning features, solving challenges of feature consistency, reusability, and serving latency between training and inference.

**What You'll Learn:**
- Feature store architectures
- Online vs offline serving
- Feature versioning and lineage
- Real-time feature computation
- Production implementation patterns

---

## Table of Contents

1. [Why Feature Stores](#why-feature-stores)
2. [Architecture Overview](#architecture-overview)
3. [Feature Store Comparison](#feature-store-comparison)
4. [Implementation Examples](#implementation-examples)
5. [Real-Time Features](#real-time-features)
6. [Best Practices](#best-practices)

---

## Why Feature Stores

### Problems Feature Stores Solve

```python
# The Feature Engineering Challenge

WITHOUT Feature Store:
❌ Training/serving skew (different code paths)
❌ Feature recomputation (waste of resources)
❌ No feature discovery (teams rebuild same features)
❌ Slow feature serving (query databases repeatedly)
❌ No feature versioning (hard to debug)
❌ Point-in-time correctness issues (data leakage)

WITH Feature Store:
✅ Single source of truth for features
✅ Consistent training/serving
✅ Feature reuse across teams
✅ Fast online serving (ms latency)
✅ Built-in versioning and lineage
✅ Point-in-time correct joins
```

### Use Cases

```python
# When You NEED a Feature Store:

1. Multiple ML models sharing features
2. Real-time predictions (<100ms latency)
3. Large teams building ML models
4. Complex feature pipelines
5. Regulatory requirements (feature lineage)

# When You DON'T NEED a Feature Store (Yet):

1. Single model, small team
2. Batch predictions only
3. Simple features (no transformations)
4. Prototype/research phase
```

---

## Architecture Overview

```python
# Feature Store Architecture

┌─────────────────────────────────────────────────────┐
│                  Feature Definition                 │
│  (Python, SQL, or declarative config)               │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│ Offline Store │    │  Online Store  │
│ (Data Lake)   │    │  (Low Latency) │
│               │    │                │
│ - S3/HDFS     │    │ - Redis        │
│ - Parquet     │    │ - DynamoDB     │
│ - Historical  │    │ - Real-time    │
│ - Training    │    │ - Serving      │
└───────┬───────┘    └────────┬───────┘
        │                     │
        │                     │
        ▼                     ▼
┌───────────────┐    ┌────────────────┐
│   Training    │    │   Inference    │
│   Pipeline    │    │    Service     │
└───────────────┘    └────────────────┘
```

---

## Feature Store Comparison

### Overview Matrix

| Feature Store | Best For | Hosting | Cost | Maturity | Community |
|---------------|----------|---------|------|----------|-----------|
| **Feast** | Open-source, flexible | Self-hosted | Free | High | Large |
| **Tecton** | Enterprise, real-time | Managed | $$$ | Very High | Medium |
| **AWS FeatureStore** | AWS ecosystem | Managed | $$ | High | Large |
| **Databricks FS** | Databricks users | Managed | $$ | High | Large |
| **Hopsworks** | MLOps platform | Both | $/$$$ | High | Medium |

---

## Feast (Open Source)

```python
# Feast - Most Popular Open Source Feature Store

✅ Pros:
• Completely open source
• Cloud-agnostic
• Active community
• Good documentation
• Flexible architecture

❌ Cons:
• Requires ops expertise
• Manual scaling
• Limited UI
• Self-managed infrastructure

🎯 Use When:
• Want open-source solution
• Have ops resources
• Need flexibility
• Multi-cloud or on-prem
```

### Setup and Configuration

```python
# feast_repo/feature_repo.py - Feast Feature Repository

from datetime import timedelta
from feast import (
    Entity,
    Feature,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
    ValueType,
)
from feast.types import Float32, Float64, Int32, Int64, String
from feast.on_demand_feature_view import on_demand_feature_view
import pandas as pd

# Define entity (primary key for features)
user = Entity(
    name="user_id",
    description="User identifier",
)

# Data source (offline)
user_transactions = FileSource(
    path="data/user_transactions.parquet",
    timestamp_field="event_timestamp",
)

# Feature view - defines how to load features
user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    schema=[
        Field(name="total_transactions", dtype=Int64),
        Field(name="total_amount", dtype=Float64),
        Field(name="avg_transaction", dtype=Float64),
        Field(name="max_transaction", dtype=Float64),
        Field(name="days_since_last_transaction", dtype=Int32),
    ],
    source=user_transactions,
    ttl=timedelta(days=1),  # How long features are valid
)

# Real-time push source
user_realtime_source = PushSource(
    name="user_realtime_push",
    batch_source=user_transactions,
)

# Real-time feature view
user_realtime_features = FeatureView(
    name="user_realtime_features",
    entities=[user],
    schema=[
        Field(name="current_session_duration", dtype=Int32),
        Field(name="current_session_clicks", dtype=Int32),
    ],
    source=user_realtime_source,
    ttl=timedelta(minutes=10),
)

# On-demand feature view (computed at request time)
@on_demand_feature_view(
    sources=[
        user_transaction_features,
        RequestSource(
            name="request_data",
            schema=[
                Field(name="requested_amount", dtype=Float64),
            ]
        ),
    ],
    schema=[
        Field(name="transaction_to_avg_ratio", dtype=Float64),
        Field(name="transaction_to_max_ratio", dtype=Float64),
    ],
)
def transaction_ratios(inputs: pd.DataFrame) -> pd.DataFrame:
    """Compute ratios on-demand"""
    df = pd.DataFrame()
    df["transaction_to_avg_ratio"] = (
        inputs["requested_amount"] / inputs["avg_transaction"]
    )
    df["transaction_to_max_ratio"] = (
        inputs["requested_amount"] / inputs["max_transaction"]
    )
    return df
```

### Feature Engineering Pipeline

```python
# feature_pipeline.py - Generate Features for Offline Store

import pandas as pd
from feast import FeatureStore
from datetime import datetime, timedelta
import pyarrow.parquet as pq

class FeatureEngineeringPipeline:
    """Generate and materialize features"""
    
    def __init__(self, feature_repo_path: str):
        self.store = FeatureStore(repo_path=feature_repo_path)
    
    def compute_user_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregate user features"""
        
        features = transactions_df.groupby('user_id').agg({
            'transaction_id': 'count',  # total_transactions
            'amount': ['sum', 'mean', 'max'],  # total, avg, max
        }).reset_index()
        
        features.columns = [
            'user_id',
            'total_transactions',
            'total_amount',
            'avg_transaction',
            'max_transaction'
        ]
        
        # Days since last transaction
        latest_transactions = transactions_df.groupby('user_id')['transaction_date'].max()
        features['days_since_last_transaction'] = (
            datetime.now() - latest_transactions
        ).dt.days
        
        # Add event timestamp
        features['event_timestamp'] = datetime.now()
        
        return features
    
    def materialize_features(self, features_df: pd.DataFrame, output_path: str):
        """Write features to parquet"""
        features_df.to_parquet(output_path, index=False)
        print(f"Features written to {output_path}")
    
    def materialize_to_online_store(
        self,
        start_date: datetime,
        end_date: datetime
    ):
        """Materialize features from offline to online store"""
        
        self.store.materialize(
            start_date=start_date,
            end_date=end_date,
            feature_views=["user_transaction_features"]
        )
        print("Features materialized to online store")

# Usage
pipeline = FeatureEngineeringPipeline(feature_repo_path=".")

# Compute features from raw data
transactions = pd.read_csv("data/transactions.csv")
features = pipeline.compute_user_features(transactions)

# Save to offline store
pipeline.materialize_features(features, "data/user_transactions.parquet")

# Materialize to online store for serving
pipeline.materialize_to_online_store(
    start_date=datetime.now() - timedelta(days=7),
    end_date=datetime.now()
)
```

### Feature Serving

```python
# model_serving.py - Use Features for Training and Inference

from feast import FeatureStore
import pandas as pd
from datetime import datetime

class MLService:
    """ML service using Feast feature store"""
    
    def __init__(self, feature_repo_path: str):
        self.store = FeatureStore(repo_path=feature_repo_path)
    
    def get_training_data(
        self,
        entity_df: pd.DataFrame,
        features: list[str]
    ) -> pd.DataFrame:
        """Get historical features for training (point-in-time correct)"""
        
        # Entity DataFrame must have entity columns + event_timestamp
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
        ).to_df()
        
        return training_df
    
    def get_online_features(
        self,
        entity_rows: list[dict],
        features: list[str]
    ) -> dict:
        """Get features for real-time inference"""
        
        # Fast lookup from online store (Redis, DynamoDB, etc.)
        feature_vector = self.store.get_online_features(
            features=features,
            entity_rows=entity_rows,
        ).to_dict()
        
        return feature_vector
    
    def push_realtime_features(
        self,
        push_source_name: str,
        features_df: pd.DataFrame
    ):
        """Push real-time computed features"""
        
        self.store.push(
            push_source_name=push_source_name,
            df=features_df
        )

# Training: Get historical features with point-in-time correctness
service = MLService(feature_repo_path=".")

entity_df = pd.DataFrame({
    "user_id": [1001, 1002, 1003],
    "event_timestamp": [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 3),
    ],
    "label": [1, 0, 1],  # Target variable
})

training_data = service.get_training_data(
    entity_df=entity_df,
    features=[
        "user_transaction_features:total_transactions",
        "user_transaction_features:avg_transaction",
        "user_transaction_features:days_since_last_transaction",
    ]
)

# Inference: Get online features (low latency)
features = service.get_online_features(
    entity_rows=[
        {"user_id": 1001},
        {"user_id": 1002},
    ],
    features=[
        "user_transaction_features:total_transactions",
        "user_transaction_features:avg_transaction",
        "user_realtime_features:current_session_clicks",
    ]
)

# Use features for prediction
print(features)
# {
#   'user_id': [1001, 1002],
#   'total_transactions': [45, 23],
#   'avg_transaction': [125.50, 89.30],
#   'current_session_clicks': [12, 8]
# }
```

---

## Real-Time Features

### Stream Processing with Feast

```python
# streaming_features.py - Real-time Feature Computation

from feast import FeatureStore
import pandas as pd
from kafka import KafkaConsumer
import json
from datetime import datetime
from typing import Dict

class RealtimeFeatureProcessor:
    """Process streaming events and compute features"""
    
    def __init__(
        self,
        feature_repo_path: str,
        kafka_bootstrap_servers: str,
        kafka_topic: str
    ):
        self.store = FeatureStore(repo_path=feature_repo_path)
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        # State management (use Redis in production)
        self.session_state: Dict[str, Dict] = {}
    
    def process_stream(self):
        """Process events from Kafka and compute features"""
        
        for message in self.consumer:
            event = message.value
            
            # Compute features
            features = self._compute_features(event)
            
            # Push to Feast
            self._push_to_feast(features)
    
    def _compute_features(self, event: Dict) -> pd.DataFrame:
        """Compute real-time features from event"""
        
        user_id = event['user_id']
        event_type = event['event_type']
        timestamp = datetime.fromisoformat(event['timestamp'])
        
        # Update session state
        if user_id not in self.session_state:
            self.session_state[user_id] = {
                'session_start': timestamp,
                'clicks': 0,
                'page_views': 0
            }
        
        session = self.session_state[user_id]
        
        if event_type == 'click':
            session['clicks'] += 1
        elif event_type == 'page_view':
            session['page_views'] += 1
        
        # Calculate session duration
        session_duration = (timestamp - session['session_start']).total_seconds()
        
        # Create features DataFrame
        features_df = pd.DataFrame([{
            'user_id': user_id,
            'current_session_duration': int(session_duration),
            'current_session_clicks': session['clicks'],
            'event_timestamp': timestamp
        }])
        
        return features_df
    
    def _push_to_feast(self, features_df: pd.DataFrame):
        """Push computed features to Feast online store"""
        
        self.store.push(
            push_source_name="user_realtime_push",
            df=features_df
        )

# Usage
processor = RealtimeFeatureProcessor(
    feature_repo_path=".",
    kafka_bootstrap_servers="localhost:9092",
    kafka_topic="user_events"
)

processor.process_stream()
```

### Feature Transformation Pipeline

```python
# feature_transformations.py - Complex Feature Engineering

from feast import FeatureStore
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List

class FeatureTransformationPipeline:
    """Advanced feature transformations"""
    
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
        self.scalers: Dict[str, StandardScaler] = {}
    
    def create_time_based_features(
        self,
        df: pd.DataFrame,
        timestamp_col: str
    ) -> pd.DataFrame:
        """Extract time-based features"""
        
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour_of_day'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour_of_day'].between(9, 17).astype(int)
        
        return df
    
    def create_aggregation_features(
        self,
        df: pd.DataFrame,
        group_col: str,
        agg_col: str,
        windows: List[int] = [7, 14, 30]
    ) -> pd.DataFrame:
        """Create rolling aggregation features"""
        
        df = df.copy()
        df = df.sort_values('event_timestamp')
        
        for window in windows:
            # Rolling mean
            df[f'{agg_col}_mean_{window}d'] = (
                df.groupby(group_col)[agg_col]
                .transform(lambda x: x.rolling(window, min_periods=1).mean())
            )
            
            # Rolling sum
            df[f'{agg_col}_sum_{window}d'] = (
                df.groupby(group_col)[agg_col]
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
            )
            
            # Rolling std
            df[f'{agg_col}_std_{window}d'] = (
                df.groupby(group_col)[agg_col]
                .transform(lambda x: x.rolling(window, min_periods=1).std())
            )
        
        return df
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and interaction features"""
        
        df = df.copy()
        
        # Ratios
        df['amount_to_avg_ratio'] = df['amount'] / df['avg_transaction']
        df['frequency_score'] = df['total_transactions'] / df['days_since_first']
        
        # Interactions
        df['amount_x_frequency'] = df['amount'] * df['total_transactions']
        
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def normalize_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """Normalize numerical features"""
        
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if fit or col not in self.scalers:
                scaler = StandardScaler()
                df[f'{col}_normalized'] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                df[f'{col}_normalized'] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def create_embedding_features(
        self,
        df: pd.DataFrame,
        text_col: str,
        embedding_model
    ) -> pd.DataFrame:
        """Create embeddings from text"""
        
        df = df.copy()
        
        # Generate embeddings
        embeddings = embedding_model.encode(df[text_col].tolist())
        
        # Add as separate columns
        for i in range(embeddings.shape[1]):
            df[f'{text_col}_emb_{i}'] = embeddings[:, i]
        
        return df

# Usage
store = FeatureStore(repo_path=".")
pipeline = FeatureTransformationPipeline(store)

# Load raw data
df = pd.read_csv("data/transactions.csv")

# Apply transformations
df = pipeline.create_time_based_features(df, 'transaction_date')
df = pipeline.create_aggregation_features(df, 'user_id', 'amount', windows=[7, 30])
df = pipeline.create_ratio_features(df)
df = pipeline.normalize_features(df, ['amount', 'total_transactions'], fit=True)

# Save to feature store
df.to_parquet("data/enhanced_features.parquet")
```

---

## Feature Versioning and Lineage

```python
# feature_versioning.py - Track Feature Versions

from feast import FeatureStore
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import json

@dataclass
class FeatureVersion:
    """Feature version metadata"""
    feature_name: str
    version: str
    created_at: datetime
    author: str
    description: str
    schema: Dict
    dependencies: List[str]
    model_versions: List[str]  # Which models use this version

class FeatureVersionManager:
    """Manage feature versions and lineage"""
    
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
        self.versions: Dict[str, List[FeatureVersion]] = {}
    
    def register_feature_version(
        self,
        feature_name: str,
        version: str,
        description: str,
        schema: Dict,
        dependencies: List[str] = None
    ) -> FeatureVersion:
        """Register a new feature version"""
        
        feature_version = FeatureVersion(
            feature_name=feature_name,
            version=version,
            created_at=datetime.now(),
            author="data_team",
            description=description,
            schema=schema,
            dependencies=dependencies or [],
            model_versions=[]
        )
        
        if feature_name not in self.versions:
            self.versions[feature_name] = []
        
        self.versions[feature_name].append(feature_version)
        
        # Save metadata
        self._save_metadata(feature_version)
        
        return feature_version
    
    def link_to_model(self, feature_name: str, version: str, model_version: str):
        """Link feature version to model version"""
        
        for fv in self.versions.get(feature_name, []):
            if fv.version == version:
                fv.model_versions.append(model_version)
                self._save_metadata(fv)
                break
    
    def get_feature_lineage(self, feature_name: str, version: str) -> Dict:
        """Get complete lineage of a feature version"""
        
        for fv in self.versions.get(feature_name, []):
            if fv.version == version:
                return {
                    'feature': fv.feature_name,
                    'version': fv.version,
                    'created_at': fv.created_at.isoformat(),
                    'dependencies': fv.dependencies,
                    'used_by_models': fv.model_versions,
                    'upstream_features': self._get_upstream_features(fv.dependencies)
                }
        
        return {}
    
    def _get_upstream_features(self, dependencies: List[str]) -> List[Dict]:
        """Recursively get upstream feature dependencies"""
        
        upstream = []
        for dep in dependencies:
            # Parse dependency (format: "feature_name:version")
            parts = dep.split(":")
            if len(parts) == 2:
                feature_name, version = parts
                lineage = self.get_feature_lineage(feature_name, version)
                if lineage:
                    upstream.append(lineage)
        
        return upstream
    
    def _save_metadata(self, feature_version: FeatureVersion):
        """Save feature version metadata"""
        metadata = {
            'feature_name': feature_version.feature_name,
            'version': feature_version.version,
            'created_at': feature_version.created_at.isoformat(),
            'author': feature_version.author,
            'description': feature_version.description,
            'schema': feature_version.schema,
            'dependencies': feature_version.dependencies,
            'model_versions': feature_version.model_versions
        }
        
        with open(f'metadata/{feature_version.feature_name}_{feature_version.version}.json', 'w') as f:
            json.dump(metadata, f, indent=2)

# Usage
store = FeatureStore(repo_path=".")
version_manager = FeatureVersionManager(store)

# Register feature version
version_manager.register_feature_version(
    feature_name="user_transaction_features",
    version="v1.2.0",
    description="Added rolling 30-day aggregations",
    schema={
        "total_transactions": "int64",
        "avg_transaction": "float64",
        "sum_30d": "float64"
    },
    dependencies=[
        "raw_transactions:v1.0.0"
    ]
)

# Link to model
version_manager.link_to_model(
    feature_name="user_transaction_features",
    version="v1.2.0",
    model_version="fraud_detector_v3.1"
)

# Get lineage
lineage = version_manager.get_feature_lineage("user_transaction_features", "v1.2.0")
print(json.dumps(lineage, indent=2))
```

---

## Best Practices

### 1. Feature Naming Convention

```python
# feature_naming.py

# Good naming convention:
# {domain}_{entity}_{aggregation}_{window}_{statistic}

GOOD_NAMES = [
    "user_transactions_7d_count",
    "product_views_30d_sum",
    "session_clicks_realtime_count",
    "customer_purchases_90d_avg_amount",
]

# Bad names:
BAD_NAMES = [
    "feature_1",  # Not descriptive
    "x",  # Too short
    "user_feature_thing",  # Vague
]
```

### 2. Feature Monitoring

```python
# feature_monitoring.py

from feast import FeatureStore
import pandas as pd
from scipy import stats
from typing import Dict

class FeatureMonitor:
    """Monitor feature quality and drift"""
    
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
        self.baseline_stats: Dict = {}
    
    def compute_baseline_stats(self, features_df: pd.DataFrame):
        """Compute baseline statistics"""
        
        for col in features_df.select_dtypes(include=[np.number]).columns:
            self.baseline_stats[col] = {
                'mean': features_df[col].mean(),
                'std': features_df[col].std(),
                'min': features_df[col].min(),
                'max': features_df[col].max(),
                'percentiles': features_df[col].quantile([0.25, 0.5, 0.75]).to_dict()
            }
    
    def detect_drift(self, current_features_df: pd.DataFrame) -> Dict:
        """Detect feature drift using statistical tests"""
        
        drift_detected = {}
        
        for col in current_features_df.select_dtypes(include=[np.number]).columns:
            if col not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[col]
            current = current_features_df[col]
            
            # KS test for distribution shift
            ks_statistic, p_value = stats.ks_2samp(
                [baseline['mean']] * len(current),  # Simplified
                current
            )
            
            drift_detected[col] = {
                'drifted': p_value < 0.05,
                'p_value': p_value,
                'mean_shift': current.mean() - baseline['mean'],
                'std_shift': current.std() - baseline['std']
            }
        
        return drift_detected

# Usage
monitor = FeatureMonitor(feature_store)
monitor.compute_baseline_stats(training_features)

# Later, check for drift
drift_report = monitor.detect_drift(current_features)
for feature, drift_info in drift_report.items():
    if drift_info['drifted']:
        print(f"⚠️ Drift detected in {feature}")
```

---

## Production Checklist

```markdown
✅ Feature Store Setup
  □ Feature store selected
  □ Online/offline stores configured
  □ Access controls set up
  □ Backup strategy defined

✅ Feature Development
  □ Naming conventions established
  □ Feature documentation created
  □ Versioning strategy defined
  □ Tests written for transformations

✅ Feature Serving
  □ Online serving latency < 100ms
  □ Offline batch processing optimized
  □ Point-in-time correctness verified
  □ Feature freshness monitored

✅ Monitoring
  □ Feature drift detection
  □ Data quality checks
  □ Serving latency tracked
  □ Feature usage tracked

✅ Operations
  □ Feature backfilling process
  □ Incident response playbook
  □ Feature deprecation process
  □ Cost monitoring
```

---

*This guide covers feature stores. For complete ML infrastructure, see [Vector Databases Guide](vector-databases-guide.md) and [Model Serving Guide](model-serving-guide.md).*
