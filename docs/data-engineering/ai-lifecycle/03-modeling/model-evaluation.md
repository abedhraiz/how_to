# Model Evaluation

## Purpose

Comprehensively assess model performance using multiple metrics and techniques to ensure reliability and readiness for deployment.

## 1. Classification Metrics

### Binary Classification

```python
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, auc
)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Basic Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Probabilistic Metrics
auc_score = roc_auc_score(y_test, y_pred_proba)

# Print summary
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
print(f"AUC-ROC:   {auc_score:.3f}")
```

### Multi-Class Classification

```python
from sklearn.metrics import classification_report

# Detailed report
print(classification_report(y_test, y_pred, target_names=class_names))

# Macro-averaging
macro_f1 = f1_score(y_test, y_pred, average='macro')

# Weighted-averaging (for imbalanced classes)
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
```

## 2. Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

# Error metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# Variance explained
r2 = r2_score(y_test, y_pred)

print(f"MAE:    {mae:.3f}")
print(f"RMSE:   {rmse:.3f}")
print(f"MAPE:   {mape:.1f}%")
print(f"R²:     {r2:.3f}")
```

## 3. Visualization

### ROC Curve

```python
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

### Confusion Matrix Heatmap

```python
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

### Residual Plot (Regression)

```python
residuals = y_test - y_pred

plt.figure(figsize=(10, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()
```

## 4. Cross-Validation Evaluation

```python
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, X_train, y_train,
    cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
    return_train_score=True,
)

# Print CV results
print("Cross-Validation Results:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric}:")
    print(f"  Train: {train_scores.mean():.3f} (+/- {train_scores.std():.3f})")
    print(f"  Test:  {test_scores.mean():.3f} (+/- {test_scores.std():.3f})")
```

## 5. Overfitting Analysis

```python
# Compare train vs test performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

overfitting_gap = train_score - test_score

print(f"Train Score: {train_score:.3f}")
print(f"Test Score:  {test_score:.3f}")
print(f"Gap:         {overfitting_gap:.3f}")

if overfitting_gap > 0.05:
    print("⚠ Warning: Model shows signs of overfitting")
```

## 6. Feature Importance

```python
# Tree-based models
import pandas as pd

feature_importance = model.feature_importances_
features_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features_df['feature'][:20], features_df['importance'][:20])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
```

## 7. Fairness Evaluation

```python
# Check fairness across groups
def evaluate_fairness(y_true, y_pred, protected_attribute):
    """Evaluate model fairness by protected group"""
    
    groups = protected_attribute.unique()
    fairness_metrics = {}
    
    for group in groups:
        mask = protected_attribute == group
        group_accuracy = accuracy_score(y_true[mask], y_pred[mask])
        group_f1 = f1_score(y_true[mask], y_pred[mask])
        
        fairness_metrics[group] = {
            'accuracy': group_accuracy,
            'f1': group_f1,
        }
    
    return fairness_metrics

fairness = evaluate_fairness(y_test, y_pred, protected_attribute)
print("Fairness Analysis:")
for group, metrics in fairness.items():
    print(f"{group}: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}")
```

## 8. Threshold Analysis

```python
# For binary classification, analyze threshold effects
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find optimal threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"F1 at optimal threshold: {f1_scores[optimal_idx]:.3f}")
```

## 9. Evaluation Report

### Template

```markdown
# Model Evaluation Report

## Dataset
- Training set size: 10,000
- Test set size: 2,000
- Feature count: 50

## Model Configuration
- Algorithm: Gradient Boosting
- Hyperparameters: [List]

## Performance Metrics

### Classification Metrics
- Accuracy: 0.92
- Precision: 0.89
- Recall: 0.85
- F1-Score: 0.87
- AUC-ROC: 0.91

### Cross-Validation
- CV Accuracy: 0.91 (+/- 0.02)
- CV F1-Score: 0.86 (+/- 0.03)

## Analysis
[Key findings and observations]

## Fairness & Bias
[Fairness evaluation by group]

## Recommendations
[Next steps and improvements]
```

## 10. Evaluation Checklist

- [ ] Test set separate from training
- [ ] Cross-validation performed
- [ ] Multiple metrics calculated
- [ ] Baseline comparison included
- [ ] Overfitting analyzed
- [ ] Feature importance reviewed
- [ ] Fairness evaluated
- [ ] Error cases analyzed
- [ ] Results reproducible
- [ ] Report documented

## Best Practices

1. ✅ Use multiple metrics
2. ✅ Always use separate test set
3. ✅ Cross-validate results
4. ✅ Compare against baselines
5. ✅ Analyze errors
6. ✅ Check for fairness
7. ✅ Document everything
8. ✅ Visualize results

## Common Pitfalls

- ❌ Using only accuracy
- ❌ Training on test data
- ❌ No cross-validation
- ❌ Ignoring class imbalance
- ❌ No fairness analysis
- ❌ Not understanding trade-offs

---

## Related Documents

- [Model Selection](./model-selection.md) - Algorithm selection
- [Training Pipeline](./training-pipeline.md) - Model training
- [Model Card](../templates/model-card.md) - Model documentation

---

*Thorough evaluation ensures reliable models in production*
