# Hyperparameter Tuning

## Purpose

Optimize model hyperparameters to maximize performance on validation data while preventing overfitting.

## 1. Hyperparameter Types

### Model-Specific Examples

**Gradient Boosting:**
- n_estimators: Number of trees
- learning_rate: Step size for weight updates
- max_depth: Maximum tree depth
- min_samples_split: Min samples to split node
- subsample: Fraction of samples used per tree

**Neural Network:**
- layers: Number and size of hidden layers
- learning_rate: Step size for gradient descent
- dropout: Regularization dropout rate
- batch_size: Training batch size
- epochs: Number of training iterations

**SVM:**
- C: Regularization parameter
- kernel: Type of kernel (linear, rbf, poly)
- gamma: Kernel coefficient
- epsilon: Margin of tolerance

## 2. Hyperparameter Search Strategies

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 1.0],
    'max_depth': [3, 5, 7],
    'min_samples_split': [5, 10, 20],
}

# Grid search
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.001, 1.0),
    'max_depth': randint(2, 10),
    'min_samples_split': randint(2, 20),
}

# Random search
random_search = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
)

random_search.fit(X_train, y_train)
```

### Bayesian Optimization

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space
search_spaces = {
    'n_estimators': Integer(50, 300),
    'learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'max_depth': Integer(2, 10),
    'min_samples_split': Integer(2, 20),
}

# Bayesian search
bayes_search = BayesSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    search_spaces=search_spaces,
    n_iter=20,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42,
)

bayes_search.fit(X_train, y_train)
```

### Optuna Framework

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
    }
    
    model = GradientBoostingClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()
    return score

# Create study and optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value:.3f}")
```

## 3. Learning Rate Scheduling

```python
# For neural networks
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

def lr_scheduler(epoch, lr):
    """Reduce learning rate by 0.1 every 10 epochs"""
    if epoch > 0 and epoch % 10 == 0:
        lr = lr * 0.1
    return lr

lr_callback = LearningRateScheduler(lr_scheduler)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[lr_callback],
)
```

## 4. Early Stopping

```python
from tensorflow.keras.callbacks import EarlyStopping

# Stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1,
)

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    callbacks=[early_stopping],
)
```

## 5. Regularization Techniques

### L1/L2 Regularization

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)  # Higher alpha = more regularization

# Lasso (L1 regularization)
lasso = Lasso(alpha=0.1)

# ElasticNet (L1 + L2)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

### Dropout (Neural Networks)

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),  # Drop 30% of neurons
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1, activation='sigmoid'),
])
```

### Tree-Specific Parameters

```python
# Tree depth limits overfitting
model = GradientBoostingClassifier(
    max_depth=5,  # Shallower trees = more regularization
    min_samples_split=10,  # Need more samples to split
    min_samples_leaf=5,  # Minimum samples in leaf
    subsample=0.8,  # Use 80% of data per iteration
)
```

## 6. Tuning Strategy

### Phase 1: Coarse Search
```python
# Wide parameter ranges, few iterations
param_grid = {
    'n_estimators': [50, 200, 500],
    'learning_rate': [0.001, 0.1, 1.0],
    'max_depth': [2, 5, 10],
}
```

### Phase 2: Fine Search
```python
# Narrow ranges around best parameters
param_grid = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5],
}
```

### Phase 3: Fine Tuning
```python
# Very narrow ranges
param_grid = {
    'n_estimators': [120, 140, 160],
    'learning_rate': [0.08, 0.1, 0.12],
    'max_depth': [4, 5, 6],
}
```

## 7. Monitoring Tuning Progress

```python
import pandas as pd

# Track all trials
results_df = pd.DataFrame(grid_search.cv_results_)

# Sort by mean test score
results_df = results_df.sort_values('mean_test_score', ascending=False)

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(results_df['mean_test_score'], label='Mean Test Score')
plt.fill_between(
    range(len(results_df)),
    results_df['mean_test_score'] - results_df['std_test_score'],
    results_df['mean_test_score'] + results_df['std_test_score'],
    alpha=0.3
)
plt.xlabel('Trial')
plt.ylabel('Score')
plt.title('Hyperparameter Tuning Progress')
plt.legend()
plt.show()
```

## 8. Hyperparameter Importance

```python
# Analyze which parameters matter most
param_importance = {}

for param in param_grid.keys():
    scores = []
    for idx, row in results_df.iterrows():
        scores.append(row['mean_test_score'])
    
    param_importance[param] = np.std(scores)

# Sort by importance
sorted_importance = sorted(param_importance.items(), 
                          key=lambda x: x[1], 
                          reverse=True)
print("Parameter importance:")
for param, importance in sorted_importance:
    print(f"  {param}: {importance:.4f}")
```

## 9. Tuning Checklist

- [ ] Baseline model established
- [ ] Validation set reserved
- [ ] Search space defined
- [ ] Search strategy chosen
- [ ] Computational resources sufficient
- [ ] Cross-validation configured
- [ ] Best parameters documented
- [ ] Test performance verified
- [ ] Overfitting checked
- [ ] Results reproducible

## Best Practices

1. ✅ Start with coarse grid search
2. ✅ Use validation set (not test)
3. ✅ Log all experiments
4. ✅ Visualize tuning progress
5. ✅ Check for overfitting
6. ✅ Don't optimize for training accuracy
7. ✅ Consider computational cost
8. ✅ Document final parameters

## Common Pitfalls

- ❌ Tuning on test data (leakage)
- ❌ Grid too small or too large
- ❌ Only tuning one parameter
- ❌ No early stopping
- ❌ Ignoring overfitting
- ❌ Not documenting parameters

---

## Related Documents

- [Model Selection](./model-selection.md) - Choose algorithms
- [Training Pipeline](./training-pipeline.md) - Training workflows
- [Model Evaluation](./model-evaluation.md) - Performance assessment

---

*Systematic hyperparameter tuning balances exploration and exploitation*
