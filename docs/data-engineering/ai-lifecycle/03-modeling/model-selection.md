# Model Selection

## Purpose

Choose the most appropriate machine learning algorithms and model architectures for the problem at hand.

## 1. Problem Classification

### Problem Type

**Classification vs. Regression vs. Clustering**
```
Classification: Predicting discrete categories
  Examples: Churn (yes/no), Sentiment (positive/negative/neutral)
  Metrics: Accuracy, Precision, Recall, F1, AUC-ROC
  Models: Logistic Regression, Decision Trees, SVM, Neural Networks

Regression: Predicting continuous values
  Examples: Price prediction, Sales forecasting, Temperature
  Metrics: RMSE, MAE, R², MAPE
  Models: Linear Regression, Ridge/Lasso, Gradient Boosting, Neural Networks

Clustering: Finding groups in unlabeled data
  Examples: Customer segmentation, Anomaly detection
  Metrics: Silhouette score, Inertia, Calinski-Harabasz
  Models: K-Means, DBSCAN, Hierarchical clustering, Gaussian Mixture Models

Time Series: Predicting sequential data
  Examples: Stock prices, Demand forecasting, Weather
  Metrics: RMSE, MAPE, Directional accuracy
  Models: ARIMA, Prophet, LSTM, GRU

NLP: Processing text data
  Examples: Text classification, Named Entity Recognition, Translation
  Metrics: Precision, Recall, F1, BLEU, Perplexity
  Models: Transformers, BERT, RNNs, CNNs

Computer Vision: Processing image data
  Examples: Image classification, Object detection, Segmentation
  Metrics: Accuracy, mAP, IoU
  Models: CNNs, ResNet, YOLO, Vision Transformers
```

### Data Characteristics

```python
# Determine problem characteristics
characteristics = {
    'data_size': 'large',  # small/medium/large/huge
    'feature_count': 'high',  # low/medium/high
    'feature_types': 'mixed',  # numerical/categorical/mixed
    'class_balance': 'imbalanced',  # balanced/imbalanced
    'noise_level': 'moderate',  # low/moderate/high
    'interpretability_required': True,  # Yes/No
    'latency_requirement': 'milliseconds',  # seconds/milliseconds/none
}
```

## 2. Algorithm Selection Framework

### Model Families & Characteristics

#### Linear Models
```python
from sklearn.linear_model import LogisticRegression, LinearRegression

models = {
    'logistic_regression': {
        'type': 'Classification',
        'interpretability': 'High',
        'training_time': 'Fast',
        'scalability': 'Large',
        'best_for': 'Linear relationships, high interpretability needed',
        'pros': ['Simple', 'Fast', 'Interpretable', 'Few hyperparameters'],
        'cons': ['Assumes linear relationship', 'May underfit complex data'],
    }
}
```

#### Tree-Based Models
```python
from sklearn.ensemble import RandomForest, GradientBoostingClassifier, XGBClassifier

models = {
    'random_forest': {
        'type': 'Classification/Regression',
        'interpretability': 'Medium',
        'training_time': 'Medium',
        'scalability': 'Medium',
        'best_for': 'Mixed feature types, non-linear relationships',
        'pros': ['Handles non-linearity', 'Feature importance', 'Parallel training'],
        'cons': ['Can overfit', 'Memory intensive', 'Slow prediction'],
    },
    'gradient_boosting': {
        'type': 'Classification/Regression',
        'interpretability': 'Medium',
        'training_time': 'Slow',
        'scalability': 'Medium',
        'best_for': 'High accuracy, competition-winning models',
        'pros': ['High accuracy', 'Handles interactions', 'Robust'],
        'cons': ['Slow training', 'Prone to overfitting', 'Complex tuning'],
    },
}
```

#### Distance-Based Models
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

models = {
    'knn': {
        'type': 'Classification/Regression',
        'interpretability': 'Low',
        'training_time': 'None (lazy)',
        'scalability': 'Small',
        'best_for': 'Small datasets, local patterns',
        'pros': ['Simple', 'No training needed', 'Non-parametric'],
        'cons': ['Slow prediction', 'Sensitive to scaling', 'Needs large k'],
    },
    'svm': {
        'type': 'Classification/Regression',
        'interpretability': 'Low',
        'training_time': 'Medium',
        'scalability': 'Medium',
        'best_for': 'Binary classification, high-dimensional data',
        'pros': ['Works in high dimensions', 'Memory efficient', 'Robust'],
        'cons': ['Slow for large datasets', 'Needs scaling', 'Hard to interpret'],
    },
}
```

#### Neural Networks
```python
from sklearn.neural_network import MLPClassifier
from tensorflow import keras

models = {
    'neural_network': {
        'type': 'Classification/Regression',
        'interpretability': 'Very Low',
        'training_time': 'Slow',
        'scalability': 'Large',
        'best_for': 'Complex non-linear patterns, large datasets',
        'pros': ['High flexibility', 'Learns complex patterns', 'Parallelizable'],
        'cons': ['Black box', 'Needs lots of data', 'Slow training', 'Hyperparameter sensitive'],
    },
}
```

### Algorithm Comparison Matrix

| Algorithm | Accuracy | Interpretability | Training Speed | Scalability | Best For |
|-----------|----------|------------------|-----------------|-------------|----------|
| Logistic Regression | Medium | High | Fast | Large | Linear, interpretable |
| Decision Tree | Medium | High | Fast | Medium | Non-linear, clear rules |
| Random Forest | High | Medium | Medium | Medium | Mixed features |
| Gradient Boosting | Very High | Low | Slow | Medium | Maximum accuracy |
| SVM | High | Low | Medium | Medium | High dimensions |
| Neural Network | Very High | Very Low | Slow | Large | Complex patterns |

## 3. Model Selection Process

### Step 1: Baseline Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Start with simple models
baseline_models = {
    'logistic_regression': LogisticRegression(random_state=42),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
}

# Quick evaluation
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"{name}: train={train_score:.3f}, test={test_score:.3f}")
```

### Step 2: Model Comparison

```python
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train multiple models
results = []

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
    })

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print(results_df)
```

### Step 3: Algorithm Selection

```python
# Decision logic
if data_size < 10000 and interpretability_required:
    selected_algorithms = ['Logistic Regression', 'Decision Tree']
elif data_size > 1000000 or training_time_critical:
    selected_algorithms = ['Linear Model', 'Neural Network with efficient architecture']
elif maximum_accuracy_required:
    selected_algorithms = ['Gradient Boosting', 'Neural Network', 'Ensemble']
else:
    selected_algorithms = ['Random Forest', 'Gradient Boosting']
```

## 4. Cross-Validation Strategy

```python
from sklearn.model_selection import cross_validate, StratifiedKFold

# Stratified k-fold for imbalanced data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate with cross-validation
cv_results = cross_validate(
    model, X_train, y_train,
    cv=cv,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True
)

print(f"Cross-validation scores:")
print(f"  Train Accuracy: {cv_results['train_accuracy'].mean():.3f} (+/- {cv_results['train_accuracy'].std():.3f})")
print(f"  Test Accuracy: {cv_results['test_accuracy'].mean():.3f} (+/- {cv_results['test_accuracy'].std():.3f})")
```

## 5. Model Selection Report

### Template

```markdown
# Model Selection Report

## Problem Statement
[Describe the problem]

## Selected Algorithms
1. [Algorithm 1]
   - Rationale: [Why chosen]
   - Cross-validation score: [Score]
   
2. [Algorithm 2]
   - Rationale: [Why chosen]
   - Cross-validation score: [Score]

## Algorithms Tested
- [Tested but not selected]
- [Tested but not selected]

## Justification
[Why these algorithms]

## Next Steps
- Hyperparameter tuning
- Ensemble methods
- Feature engineering refinement
```

## 6. Implementation Template

```python
class ModelSelector:
    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.results = []
    
    def evaluate_model(self, name, model, cv=5):
        """Evaluate model with cross-validation"""
        from sklearn.model_selection import cross_validate
        
        cv_results = cross_validate(
            model, self.X_train, self.y_train,
            cv=cv, scoring='f1'
        )
        
        return {
            'name': name,
            'cv_score': cv_results['test_score'].mean(),
            'cv_std': cv_results['test_score'].std(),
            'model': model,
        }
    
    def select_models(self, models_dict, cv=5):
        """Evaluate all models and return best"""
        for name, model in models_dict.items():
            result = self.evaluate_model(name, model, cv)
            self.results.append(result)
        
        # Sort by CV score
        self.results = sorted(self.results, key=lambda x: x['cv_score'], reverse=True)
        return self.results
```

## Best Practices

1. ✅ Start simple
2. ✅ Understand each algorithm
3. ✅ Use cross-validation
4. ✅ Consider domain constraints
5. ✅ Compare multiple algorithms
6. ✅ Document rationale
7. ✅ Don't fall in love with one model
8. ✅ Consider ensemble approaches

## Common Pitfalls

- ❌ Choosing model before understanding problem
- ❌ Using only default hyperparameters
- ❌ No cross-validation
- ❌ Overly complex models for simple problems
- ❌ Ignoring computational constraints
- ❌ Not trying baseline models

---

## Related Documents

- [Training Pipeline](./training-pipeline.md) - Model training
- [Hyperparameter Tuning](./hyperparameter-tuning.md) - Optimization
- [Model Evaluation](./model-evaluation.md) - Assessment

---

*Choose models based on problem characteristics and constraints, not hype*
