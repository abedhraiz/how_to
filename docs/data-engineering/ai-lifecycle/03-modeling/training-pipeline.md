# Training Pipeline

## Purpose

Build reproducible, automated workflows for training, validating, and managing machine learning models.

## 1. Data Splitting Strategy

### Train/Validation/Test Split

```python
from sklearn.model_selection import train_test_split

# Standard split: 70% train, 15% validation, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
```

### Time-Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

# For time series data, don't shuffle
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate
```

### Stratified Split

```python
from sklearn.model_selection import StratifiedKFold

# For imbalanced classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate
```

## 2. Training Configuration

### Hyperparameter Specification

```python
# Define training parameters
training_config = {
    'model': 'GradientBoostingClassifier',
    'hyperparameters': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'subsample': 0.8,
        'random_state': 42,
    },
    'training': {
        'batch_size': 32,
        'epochs': 100,
        'early_stopping': True,
        'validation_split': 0.2,
    },
    'preprocessing': {
        'scaler': 'StandardScaler',
        'feature_selection': 'SelectKBest',
    }
}

import json
with open('training_config.json', 'w') as f:
    json.dump(training_config, f)
```

## 3. Model Training

### Basic Training Loop

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Save
joblib.dump(pipeline, 'models/model_v1.pkl')
```

### Neural Network Training

```python
import tensorflow as tf
from tensorflow import keras

# Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()],
)

# Train with early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1,
)

# Save
model.save('models/model_v1.h5')
```

## 4. Cross-Validation Training

```python
from sklearn.model_selection import cross_validate
import numpy as np

# Cross-validation with multiple metrics
cv_results = cross_validate(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True,
    n_jobs=-1,  # Parallel processing
)

# Summarize results
print("Cross-Validation Results:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    train = cv_results[f'train_{metric}']
    test = cv_results[f'test_{metric}']
    print(f"{metric}: {test.mean():.3f} (+/- {test.std():.3f})")
```

## 5. Model Versioning

### Experiment Tracking

```python
import mlflow
from datetime import datetime

# Initialize MLflow
mlflow.set_experiment("model_development")

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'model_type': 'GradientBoosting',
        'n_estimators': 100,
        'learning_rate': 0.1,
    })
    
    # Train model
    model.fit(X_train, y_train)
    
    # Log metrics
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    
    mlflow.log_metrics({
        'train_accuracy': train_score,
        'val_accuracy': val_score,
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("training_config.json")
```

### Model Registry

```python
# Register model
run_id = mlflow.search_runs()[-1].run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(
    model_uri=model_uri,
    name="production_model"
)

# Transition to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="production_model",
    version=model_details.version,
    stage="Production",
)
```

## 6. Training Monitoring

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train,
    y=y_train,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1,
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.legend()
plt.title('Learning Curves')
plt.show()
```

### Training History (Neural Networks)

```python
# Plot training history
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(history.history['val_loss'], label='Validation Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Model Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.show()
```

## 7. Reproducible Pipeline

### With Scikit-learn

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier

# Create reproducible pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=20)),
    ('model', GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )),
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
score = pipeline.score(X_test, y_test)
print(f"Test Score: {score:.3f}")

# Save pipeline
joblib.dump(pipeline, 'models/pipeline_v1.pkl')
```

## 8. Ensemble Training

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

# Create ensemble
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ],
    voting='soft',
)

# Train
ensemble.fit(X_train, y_train)

# Evaluate
score = ensemble.score(X_test, y_test)
print(f"Ensemble Score: {score:.3f}")
```

## 9. Training Checklist

- [ ] Data properly split (no leakage)
- [ ] Preprocessing fitted only on training data
- [ ] Model trained with cross-validation
- [ ] Hyperparameters tuned
- [ ] Model saved with version
- [ ] Metrics logged
- [ ] Training config documented
- [ ] Training time tracked
- [ ] Memory usage acceptable
- [ ] Reproducibility verified

## Best Practices

1. ✅ Use pipelines for reproducibility
2. ✅ Prevent data leakage
3. ✅ Use cross-validation
4. ✅ Version models and configs
5. ✅ Log all experiments
6. ✅ Monitor training progress
7. ✅ Save checkpoints
8. ✅ Document decisions

## Common Pitfalls

- ❌ Data leakage (fitting on full data)
- ❌ No cross-validation
- ❌ Manual hyperparameter tuning
- ❌ Not versioning models
- ❌ Losing training configs
- ❌ Not reproducing results

---

## Related Documents

- [Model Selection](./model-selection.md) - Choose algorithms
- [Hyperparameter Tuning](./hyperparameter-tuning.md) - Optimize parameters
- [Model Evaluation](./model-evaluation.md) - Assess performance
- [Experiment Log](../templates/experiment-log.md) - Track experiments

---

*Reproducible pipelines enable consistent, maintainable model training*
