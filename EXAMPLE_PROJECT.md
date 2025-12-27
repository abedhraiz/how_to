# 🏗️ End-to-End MLOps Project Example

**Build a complete production ML system integrating multiple tools from this repository**

This comprehensive example shows how to build a real-world recommendation system using Docker, Kubernetes, Kafka, Airflow, Databricks, PostgreSQL, W&B, Prometheus, and GitHub Actions.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Phase 1: Local Development](#phase-1-local-development)
- [Phase 2: Data Pipeline](#phase-2-data-pipeline)
- [Phase 3: ML Training](#phase-3-ml-training)
- [Phase 4: Model Deployment](#phase-4-model-deployment)
- [Phase 5: Monitoring](#phase-5-monitoring)
- [Phase 6: CI/CD](#phase-6-cicd)
- [Complete Code](#complete-code)

---

## 🎯 Project Overview

### What We're Building

A production-ready **movie recommendation system** that:
- Ingests user interactions in real-time
- Processes data with batch and streaming pipelines
- Trains and tracks ML models
- Serves predictions via REST API
- Monitors everything in production
- Automates deployment with CI/CD

### Technologies Used

| Component | Tool | Guide Reference |
|-----------|------|-----------------|
| Version Control | Git | [Git Guide](docs/version-control/git/git-guide.md) |
| Containerization | Docker | [Docker Guide](docs/infrastructure-devops/docker/docker-guide.md) |
| Orchestration | Kubernetes | [Kubernetes Guide](docs/infrastructure-devops/kubernetes/kubernetes-guide.md) |
| Streaming | Kafka | [Kafka Guide](docs/data-engineering/streaming/apache-kafka-guide.md) |
| Workflow | Airflow | [Airflow Guide](docs/data-engineering/workflow-orchestration/apache-airflow-guide.md) |
| Big Data | Databricks | [Databricks Guide](docs/cloud-platforms/databricks/databricks-guide.md) |
| Database | PostgreSQL | [PostgreSQL Guide](docs/databases/postgresql/postgresql-guide.md) |
| ML Tracking | W&B | [W&B Guide](docs/data-engineering/ml-ops/wandb-guide.md) |
| Monitoring | Prometheus + Grafana | [Prometheus Guide](docs/monitoring-observability/prometheus-grafana/prometheus-grafana-guide.md) |
| CI/CD | GitHub Actions | [GitHub Actions Guide](docs/cicd-automation/github-actions/github-actions-guide.md) |

### Learning Outcomes

After completing this project, you'll know how to:
- Design end-to-end ML architecture
- Build real-time and batch data pipelines
- Track and manage ML experiments
- Deploy models to production
- Monitor ML systems
- Automate ML workflows with CI/CD

---

## 🏛️ Architecture

### High-Level Architecture

```
┌─────────────────┐
│  Web/Mobile App │
└────────┬────────┘
         │ User events
         ▼
┌─────────────────┐      ┌──────────────┐
│  Kafka Topics   │─────▶│  Airflow     │
│  - clicks       │      │  DAGs        │
│  - views        │      └──────┬───────┘
│  - ratings      │             │
└────────┬────────┘             │
         │                      ▼
         │              ┌──────────────┐
         │              │  Databricks  │
         │              │  Processing  │
         │              └──────┬───────┘
         │                     │
         ▼                     ▼
┌─────────────────┐    ┌──────────────┐
│   PostgreSQL    │◀───│  Feature     │
│   - Users       │    │  Engineering │
│   - Movies      │    └──────┬───────┘
│   - Ratings     │           │
└─────────────────┘           ▼
                      ┌──────────────┐
                      │  ML Training │
                      │  (W&B Track) │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
                      │  Model       │
                      │  Registry    │
                      └──────┬───────┘
                             │
                             ▼
┌─────────────────┐   ┌──────────────┐   ┌──────────────┐
│  Docker Image   │──▶│  Kubernetes  │──▶│  Prometheus  │
│  Model API      │   │  Deployment  │   │  Grafana     │
└─────────────────┘   └──────────────┘   └──────────────┘
         ▲                    │
         │                    │
         │            ┌───────▼────────┐
         └────────────│  GitHub Actions│
                      │  CI/CD Pipeline│
                      └────────────────┘
```

### Data Flow

1. **Real-time Events**: Users interact with movies → Kafka
2. **Stream Processing**: Kafka → Databricks Streaming → Feature Store
3. **Batch Processing**: Airflow orchestrates daily ETL → PostgreSQL → Feature Engineering
4. **Model Training**: Features → Training with W&B tracking → Model Registry
5. **Model Serving**: Docker container → Kubernetes deployment
6. **Monitoring**: Prometheus scrapes metrics → Grafana visualizes
7. **CI/CD**: Git push → GitHub Actions → Test → Build → Deploy

---

## 📋 Prerequisites

### Knowledge Required

- Basic Python programming
- Basic SQL
- Command line proficiency
- Completed at least the following guides:
  - [Docker Guide](docs/infrastructure-devops/docker/docker-guide.md)
  - [Kubernetes Guide](docs/infrastructure-devops/kubernetes/kubernetes-guide.md)
  - [Git Guide](docs/version-control/git/git-guide.md)

### Software Required

```bash
# Check you have these installed
docker --version          # 20.10+
docker-compose --version  # 2.0+
kubectl version          # 1.24+
python --version         # 3.8+
git --version           # 2.30+
```

### Cloud Accounts (Optional but Recommended)

- AWS, GCP, or Azure for production deployment
- Databricks Community Edition (free)
- Weights & Biases (free tier)
- GitHub account

---

## 🔧 Phase 1: Local Development

### Step 1.1: Set Up Project Structure

```bash
# Create project directory
mkdir mlops-recommendation-system
cd mlops-recommendation-system

# Initialize git repo
git init
git branch -M main

# Create directory structure
mkdir -p {data,src,models,notebooks,docker,kubernetes,airflow,tests,monitoring}
mkdir -p src/{api,training,features,streaming}

# Create project structure
touch README.md .gitignore requirements.txt
```

### Step 1.2: Create `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
venv/
env/
.env

# Data
data/*.csv
data/*.parquet
*.db
*.sqlite

# Models
models/*.pkl
models/*.h5
models/*.pt

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Secrets
secrets/
*.pem
*.key
```

### Step 1.3: Dependencies

```txt
# requirements.txt

# Core
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.2

# ML
lightgbm==3.3.5
optuna==3.1.0

# Data
pyarrow==11.0.0
sqlalchemy==2.0.0
psycopg2-binary==2.9.5

# API
fastapi==0.95.0
uvicorn==0.21.0
pydantic==1.10.7

# Tracking
wandb==0.15.0

# Streaming
kafka-python==2.0.2

# Orchestration
apache-airflow==2.6.0

# Monitoring
prometheus-client==0.16.0

# Testing
pytest==7.3.0
pytest-cov==4.0.0

# Utils
python-dotenv==1.0.0
requests==2.28.2
```

### Step 1.4: Docker Compose for Local Development

```yaml
# docker-compose.yml

version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: recommendations
      POSTGRES_USER: mlops
      POSTGRES_PASSWORD: mlops123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql

  # Apache Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  # Airflow (simplified for local)
  airflow:
    image: apache/airflow:2.6.0
    depends_on:
      - postgres
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://mlops:mlops123@postgres/recommendations
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    ports:
      - "8080:8080"
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/logs:/opt/airflow/logs
      - ./src:/opt/airflow/src
    command: webserver

  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  # Grafana
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres_data:
  prometheus_data:
  grafana_data:
```

### Step 1.5: Start Local Environment

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps

# View logs
docker-compose logs -f kafka postgres
```

---

## 📊 Phase 2: Data Pipeline

### Step 2.1: Database Schema

```sql
-- sql/init.sql

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE movies (
    movie_id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genre VARCHAR(100),
    release_year INTEGER,
    avg_rating FLOAT DEFAULT 0
);

CREATE TABLE ratings (
    rating_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    movie_id INTEGER REFERENCES movies(movie_id),
    rating FLOAT CHECK (rating >= 0 AND rating <= 5),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, movie_id)
);

CREATE TABLE user_interactions (
    interaction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    movie_id INTEGER REFERENCES movies(movie_id),
    interaction_type VARCHAR(50), -- 'view', 'click', 'rating'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ratings_user ON ratings(user_id);
CREATE INDEX idx_ratings_movie ON ratings(movie_id);
CREATE INDEX idx_interactions_user ON user_interactions(user_id);
CREATE INDEX idx_interactions_timestamp ON user_interactions(timestamp);
```

### Step 2.2: Kafka Producer (Real-time Events)

```python
# src/streaming/event_producer.py

import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

class EventProducer:
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = 'user_interactions'
    
    def send_interaction(self, user_id, movie_id, interaction_type):
        """Send user interaction event to Kafka"""
        event = {
            'user_id': user_id,
            'movie_id': movie_id,
            'interaction_type': interaction_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        self.producer.send(self.topic, value=event)
        self.producer.flush()
        print(f"Sent event: {event}")
    
    def simulate_user_activity(self, num_events=100):
        """Simulate user interactions for testing"""
        interaction_types = ['view', 'click', 'rating']
        
        for _ in range(num_events):
            user_id = random.randint(1, 1000)
            movie_id = random.randint(1, 500)
            interaction_type = random.choice(interaction_types)
            
            self.send_interaction(user_id, movie_id, interaction_type)
            time.sleep(0.1)  # 100ms between events

if __name__ == '__main__':
    producer = EventProducer()
    producer.simulate_user_activity(num_events=1000)
```

### Step 2.3: Kafka Consumer (Stream Processing)

```python
# src/streaming/event_consumer.py

import json
from kafka import KafkaConsumer
import psycopg2
from datetime import datetime

class EventConsumer:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'user_interactions',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='interaction_processor',
            auto_offset_reset='earliest'
        )
        
        # Database connection
        self.conn = psycopg2.connect(
            host='localhost',
            database='recommendations',
            user='mlops',
            password='mlops123'
        )
        self.cursor = self.conn.cursor()
    
    def process_events(self):
        """Consume and process events from Kafka"""
        print("Starting event consumer...")
        
        for message in self.consumer:
            event = message.value
            self.save_to_database(event)
    
    def save_to_database(self, event):
        """Save interaction to PostgreSQL"""
        query = """
            INSERT INTO user_interactions 
            (user_id, movie_id, interaction_type, timestamp)
            VALUES (%s, %s, %s, %s)
        """
        
        try:
            self.cursor.execute(query, (
                event['user_id'],
                event['movie_id'],
                event['interaction_type'],
                datetime.fromisoformat(event['timestamp'])
            ))
            self.conn.commit()
            print(f"Saved: {event['interaction_type']} for user {event['user_id']}")
        except Exception as e:
            print(f"Error saving to database: {e}")
            self.conn.rollback()

if __name__ == '__main__':
    consumer = EventConsumer()
    consumer.process_events()
```

### Step 2.4: Airflow DAG (Batch Processing)

```python
# airflow/dags/feature_engineering_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime, timedelta
import pandas as pd
import psycopg2

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'feature_engineering',
    default_args=default_args,
    description='Daily feature engineering for recommendation model',
    schedule_interval='@daily',
    catchup=False
)

def extract_user_features(**context):
    """Extract user-level features"""
    conn = psycopg2.connect(
        host='postgres',
        database='recommendations',
        user='mlops',
        password='mlops123'
    )
    
    query = """
        SELECT 
            user_id,
            COUNT(DISTINCT movie_id) as movies_rated,
            AVG(rating) as avg_rating,
            STDDEV(rating) as rating_variance,
            MAX(timestamp) as last_interaction
        FROM ratings
        WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY user_id
    """
    
    df = pd.read_sql(query, conn)
    df.to_csv('/tmp/user_features.csv', index=False)
    conn.close()
    
    print(f"Extracted features for {len(df)} users")

def extract_movie_features(**context):
    """Extract movie-level features"""
    conn = psycopg2.connect(
        host='postgres',
        database='recommendations',
        user='mlops',
        password='mlops123'
    )
    
    query = """
        SELECT 
            movie_id,
            COUNT(*) as num_ratings,
            AVG(rating) as avg_rating,
            STDDEV(rating) as rating_variance
        FROM ratings
        WHERE timestamp >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY movie_id
        HAVING COUNT(*) >= 10
    """
    
    df = pd.read_sql(query, conn)
    df.to_csv('/tmp/movie_features.csv', index=False)
    conn.close()
    
    print(f"Extracted features for {len(df)} movies")

def create_training_dataset(**context):
    """Combine features into training dataset"""
    user_features = pd.read_csv('/tmp/user_features.csv')
    movie_features = pd.read_csv('/tmp/movie_features.csv')
    
    # Get recent interactions
    conn = psycopg2.connect(
        host='postgres',
        database='recommendations',
        user='mlops',
        password='mlops123'
    )
    
    interactions_query = """
        SELECT * FROM ratings
        WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
    """
    
    interactions = pd.read_sql(interactions_query, conn)
    conn.close()
    
    # Merge features
    dataset = interactions.merge(user_features, on='user_id', how='left')
    dataset = dataset.merge(movie_features, on='movie_id', how='left')
    
    dataset.to_csv('/tmp/training_dataset.csv', index=False)
    
    print(f"Created training dataset with {len(dataset)} samples")

# Define tasks
extract_users = PythonOperator(
    task_id='extract_user_features',
    python_callable=extract_user_features,
    dag=dag
)

extract_movies = PythonOperator(
    task_id='extract_movie_features',
    python_callable=extract_movie_features,
    dag=dag
)

create_dataset = PythonOperator(
    task_id='create_training_dataset',
    python_callable=create_training_dataset,
    dag=dag
)

# Set dependencies
[extract_users, extract_movies] >> create_dataset
```

---

## 🤖 Phase 3: ML Training

### Step 3.1: Feature Engineering

```python
# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
    
    def create_user_features(self, df):
        """Create user-level features"""
        user_features = df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count'],
            'timestamp': 'max'
        })
        
        user_features.columns = ['_'.join(col).strip() for col in user_features.columns.values]
        user_features = user_features.reset_index()
        
        # Calculate days since last interaction
        user_features['days_since_last'] = (
            pd.Timestamp.now() - pd.to_datetime(user_features['timestamp_max'])
        ).dt.days
        
        return user_features
    
    def create_movie_features(self, df):
        """Create movie-level features"""
        movie_features = df.groupby('movie_id').agg({
            'rating': ['mean', 'std', 'count'],
            'user_id': 'nunique'
        })
        
        movie_features.columns = ['_'.join(col).strip() for col in movie_features.columns.values]
        movie_features = movie_features.reset_index()
        movie_features.columns = ['movie_id', 'rating_mean', 'rating_std', 'rating_count', 'unique_users']
        
        # Popularity score
        movie_features['popularity_score'] = (
            movie_features['rating_mean'] * np.log1p(movie_features['rating_count'])
        )
        
        return movie_features
    
    def create_interaction_features(self, df, user_features, movie_features):
        """Create interaction-level features"""
        # Merge user and movie features
        df = df.merge(user_features, on='user_id', how='left')
        df = df.merge(movie_features, on='movie_id', how='left')
        
        # User-movie interaction features
        df['user_rating_diff'] = df['rating'] - df['rating_mean_x']  # user mean
        df['movie_rating_diff'] = df['rating'] - df['rating_mean_y']  # movie mean
        
        # Time-based features
        df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def fit_transform(self, X):
        """Scale features"""
        return self.scaler.fit_transform(X)
    
    def transform(self, X):
        """Transform features using fitted scaler"""
        return self.scaler.transform(X)
```

### Step 3.2: Model Training with W&B

```python
# src/training/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import wandb
import joblib
from datetime import datetime

class RecommendationModel:
    def __init__(self, project_name='movie-recommendations'):
        self.project_name = project_name
        self.model = None
        
    def train(self, X_train, y_train, X_val, y_val, config=None):
        """Train LightGBM model with W&B tracking"""
        
        # Initialize W&B
        run = wandb.init(
            project=self.project_name,
            config=config or {
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'n_estimators': 100,
                'objective': 'regression',
                'metric': 'rmse'
            }
        )
        
        # Get config
        config = wandb.config
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Training parameters
        params = {
            'objective': config.objective,
            'metric': config.metric,
            'learning_rate': config.learning_rate,
            'num_leaves': config.num_leaves,
            'max_depth': config.max_depth,
            'verbose': -1
        }
        
        # Train model
        callbacks = [
            lgb.log_evaluation(period=10),
            wandb.lightgbm.wandb_callback()
        ]
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=config.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_mae = mean_absolute_error(y_val, val_pred)
        
        # Log metrics
        wandb.log({
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        })
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance()
        }).sort_values('importance', ascending=False)
        
        wandb.log({
            'feature_importance': wandb.Table(dataframe=importance_df)
        })
        
        # Save model
        model_path = f'models/model_{run.id}.txt'
        self.model.save_model(model_path)
        
        # Log model artifact
        artifact = wandb.Artifact(
            name='recommendation_model',
            type='model',
            description='LightGBM recommendation model'
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)
        
        wandb.finish()
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, path='models/model.pkl'):
        """Save model"""
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path='models/model.pkl'):
        """Load model"""
        model = cls()
        model.model = joblib.load(path)
        return model

if __name__ == '__main__':
    # Load data
    df = pd.read_csv('/tmp/training_dataset.csv')
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['rating', 'user_id', 'movie_id', 'timestamp']]
    X = df[feature_cols]
    y = df['rating']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RecommendationModel()
    model.train(X_train, y_train, X_val, y_val)
    
    # Save model
    model.save('models/recommendation_model.pkl')
```

---

## 🚀 Phase 4: Model Deployment

### Step 4.1: FastAPI Service

```python
# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import time

# Prometheus metrics
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
error_counter = Counter('errors_total', 'Total number of errors')

app = FastAPI(title="Movie Recommendation API")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load('/app/models/recommendation_model.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

class PredictionRequest(BaseModel):
    user_id: int
    movie_ids: List[int]
    user_features: dict
    movie_features: List[dict]

class PredictionResponse(BaseModel):
    user_id: int
    recommendations: List[dict]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Get movie recommendations for a user"""
    
    start_time = time.time()
    
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        features_list = []
        for movie_id, movie_features in zip(request.movie_ids, request.movie_features):
            features = {
                **request.user_features,
                **movie_features,
                'movie_id': movie_id
            }
            features_list.append(features)
        
        X = pd.DataFrame(features_list)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Sort by prediction
        results = pd.DataFrame({
            'movie_id': request.movie_ids,
            'predicted_rating': predictions
        }).sort_values('predicted_rating', ascending=False)
        
        recommendations = results.to_dict('records')
        
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(time.time() - start_time)
        
        return PredictionResponse(
            user_id=request.user_id,
            recommendations=recommendations
        )
    
    except Exception as e:
        error_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 4.2: Dockerfile

```dockerfile
# docker/Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/api /app/api
COPY models /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 4.3: Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-api
  labels:
    app: recommendation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommendation-api
  template:
    metadata:
      labels:
        app: recommendation-api
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: api
        image: recommendation-api:v1
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: MODEL_PATH
          value: "/app/models/recommendation_model.pkl"
        - name: LOG_LEVEL
          value: "INFO"

---
apiVersion: v1
kind: Service
metadata:
  name: recommendation-api
  labels:
    app: recommendation-api
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
    protocol: TCP
    name: http
  selector:
    app: recommendation-api

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: recommendation-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: recommendation-api
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

---

## 📈 Phase 5: Monitoring

### Step 5.1: Prometheus Configuration

```yaml
# monitoring/prometheus.yml

global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'recommendation-api'
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

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka-exporter:9308']
```

### Step 5.2: Grafana Dashboard (JSON)

Create a custom dashboard in Grafana showing:
- Request rate
- Latency (p50, p95, p99)
- Error rate
- Prediction distribution
- Model performance metrics
- Infrastructure metrics

---

## 🔄 Phase 6: CI/CD

### Step 6.1: GitHub Actions Workflow

```yaml
# .github/workflows/deploy.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  DOCKER_REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/recommendation-api

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.DOCKER_REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: |
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
            ${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          method: kubeconfig
          kubeconfig: ${{ secrets.KUBE_CONFIG }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/recommendation-api \
            api=${{ env.DOCKER_REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          
          kubectl rollout status deployment/recommendation-api
      
      - name: Run smoke tests
        run: |
          API_URL=$(kubectl get svc recommendation-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
          curl -f http://$API_URL/health || exit 1
```

---

## ✅ Complete Setup Checklist

### Infrastructure
- [ ] Docker and Docker Compose installed
- [ ] Kubernetes cluster running (minikube/kind for local)
- [ ] PostgreSQL database set up
- [ ] Kafka cluster running
- [ ] Airflow scheduler running

### Code
- [ ] Git repository initialized
- [ ] All Python packages installed
- [ ] Database schema created
- [ ] Feature engineering code tested
- [ ] Model training code tested
- [ ] API code tested

### Deployment
- [ ] Docker image built
- [ ] Kubernetes manifests applied
- [ ] Service accessible
- [ ] Health checks passing

### Monitoring
- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards created
- [ ] Alerts configured
- [ ] Logs centralized

### CI/CD
- [ ] GitHub Actions workflow configured
- [ ] Automated tests passing
- [ ] Automatic deployment working
- [ ] Rollback strategy tested

---

## 🎯 Next Steps & Extensions

### Week 1-2: Get It Running
1. Set up local environment
2. Run data pipeline
3. Train first model
4. Deploy locally

### Week 3-4: Production Deploy
1. Deploy to cloud Kubernetes
2. Set up monitoring
3. Configure CI/CD
4. Load testing

### Month 2: Enhance
1. Add A/B testing
2. Implement feature store
3. Add model explainability
4. Set up model monitoring

### Advanced Features
- **Feature Store**: Use Feast or Databricks Feature Store
- **Model Serving**: Add Seldon or KFServing
- **A/B Testing**: Implement experimentation framework
- **Data Quality**: Add Great Expectations
- **Cost Monitoring**: Track cloud costs
- **Multi-model**: Ensemble multiple models
- **Real-time Features**: Streaming feature computation
- **AutoML**: Add Optuna hyperparameter optimization

---

## 📚 Related Guides

- [Docker Guide](docs/infrastructure-devops/docker/docker-guide.md) - Containerization
- [Kubernetes Guide](docs/infrastructure-devops/kubernetes/kubernetes-guide.md) - Orchestration
- [Apache Kafka Guide](docs/data-engineering/streaming/apache-kafka-guide.md) - Streaming
- [Apache Airflow Guide](docs/data-engineering/workflow-orchestration/apache-airflow-guide.md) - Orchestration
- [Databricks Guide](docs/cloud-platforms/databricks/databricks-guide.md) - Big data
- [PostgreSQL Guide](docs/databases/postgresql/postgresql-guide.md) - Database
- [Weights & Biases Guide](docs/data-engineering/ml-ops/wandb-guide.md) - ML tracking
- [Feature Engineering Guide](docs/data-engineering/ml-ops/feature-engineering-guide.md) - Features
- [Prometheus & Grafana Guide](docs/monitoring-observability/prometheus-grafana/prometheus-grafana-guide.md) - Monitoring
- [GitHub Actions Guide](docs/cicd-automation/github-actions/github-actions-guide.md) - CI/CD

---

## 🤝 Contributing

Found improvements? Submit a PR! This example project is meant to be educational and can always be improved.

---

[← Back to README](README.md) | [Get Started](GETTING_STARTED.md)
