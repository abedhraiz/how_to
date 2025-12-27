# Feature Engineering

## Purpose

Create and select predictive features that enable models to learn effectively and make accurate predictions.

## 1. Feature Creation

### Domain-Driven Features

#### Business Logic
```python
# Customer lifetime value
df['customer_ltv'] = df['annual_revenue'] * df['retention_years']

# Risk score
df['risk_score'] = (df['default_probability'] * 0.5 + 
                   df['age'] / 100 * 0.3 + 
                   df['debt_ratio'] * 0.2)

# Engagement index
df['engagement_index'] = (df['logins'] * 0.4 + 
                         df['posts'] * 0.3 + 
                         df['shares'] * 0.3)
```

#### Interaction Features
```python
# Feature interactions
df['age_income_interaction'] = df['age'] * df['income']
df['high_income_young'] = ((df['income'] > df['income'].median()) & 
                          (df['age'] < df['age'].median())).astype(int)
df['wealth_per_year'] = df['wealth'] / (df['age'] + 1)
```

#### Aggregated Features
```python
# Temporal aggregations
df['avg_transaction_amount'] = df.groupby('customer_id')['amount'].transform('mean')
df['transaction_count'] = df.groupby('customer_id')['amount'].transform('count')
df['max_daily_spending'] = df.groupby(['customer_id', 'date'])['amount'].transform('sum')

# Relative comparisons
df['income_vs_avg'] = df['income'] / df.groupby('region')['income'].transform('mean')
df['spending_vs_peers'] = df['spending'] / df.groupby('age_group')['spending'].transform('mean')
```

### Statistical Features

```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
features_poly = poly.fit_transform(df[['age', 'income']])

# Log transformations
df['log_income'] = np.log1p(df['income'])
df['log_tenure'] = np.log1p(df['tenure'])

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                         labels=['<25', '25-35', '35-50', '50-65', '65+'])

# Exponential decay
df['recent_activity_weight'] = np.exp(-0.1 * df['days_since_activity'])
```

### Time-Based Features

```python
# Temporal patterns
df['date'] = pd.to_datetime(df['date'])
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['is_month_end'] = df['date'].dt.is_month_end
df['quarter'] = df['date'].dt.quarter

# Lag features
df['revenue_lag_1'] = df.groupby('customer_id')['revenue'].shift(1)
df['revenue_lag_7'] = df.groupby('customer_id')['revenue'].shift(7)

# Rolling statistics
df['revenue_rolling_mean_30'] = df.groupby('customer_id')['revenue'].rolling(30).mean()
df['revenue_volatility_30'] = df.groupby('customer_id')['revenue'].rolling(30).std()

# Trend
from scipy.stats import linregress
def calculate_trend(series):
    x = np.arange(len(series))
    slope, _, _, _, _ = linregress(x, series)
    return slope

df['revenue_trend'] = df.groupby('customer_id')['revenue'].apply(
    lambda x: calculate_trend(x) if len(x) > 1 else 0
)
```

### Text-Based Features

```python
# Length features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()

# Statistical features
df['uppercase_ratio'] = df['text'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
)

# Complexity measures
df['unique_word_ratio'] = df['text'].apply(
    lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_features = vectorizer.fit_transform(df['text'])
tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                        columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
```

## 2. Feature Selection

### Variance Analysis

```python
# Remove low-variance features
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
selector.fit(df[numerical_cols])
high_variance_cols = df[numerical_cols].columns[selector.get_support()]
```

**Rationale:** Features with very low variance have little predictive power

### Correlation Analysis

```python
# Remove highly correlated features
correlation_matrix = df[numerical_cols].corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
df = df.drop(columns=to_drop)
```

**Rationale:** Correlated features add multicollinearity without new information

### Univariate Selection

```python
from sklearn.feature_selection import SelectKBest, f_regression, f_classif

# For regression
selector = SelectKBest(score_func=f_regression, k=10)
selector.fit(X, y)

# For classification
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X, y)

# Get selected features
selected_features = X.columns[selector.get_support()]
```

### Mutual Information

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Calculate mutual information
mi_scores = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)

# Keep top features
top_features = mi_df.head(20)['Feature'].tolist()
```

### Permutation Importance

```python
from sklearn.inspection import permutation_importance

# Train model
model.fit(X_train, y_train)

# Calculate permutation importance
importance = permutation_importance(model, X_test, y_test, n_repeats=10)
perm_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance.importances_mean
}).sort_values('Importance', ascending=False)
```

### Model-Based Feature Selection

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Train model with all features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Select top features
top_features = feature_importance.head(20)['Feature'].tolist()
```

### Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE

# Recursively remove features
selector = RFE(RandomForestClassifier(), n_features_to_select=10)
selector.fit(X_train, y_train)

# Get selected features
selected_features = X.columns[selector.get_support()]
```

## 3. Feature Validation

### Predictive Power

```python
# Calculate correlation with target
correlations = df[all_features].corrwith(df['target']).abs().sort_values(ascending=False)

print("Features ranked by correlation with target:")
print(correlations)

# Flag weak features
weak_features = correlations[correlations < 0.01].index.tolist()
```

### Information Value (for classification)

```python
def calculate_information_value(df, feature, target):
    """Calculate Information Value for categorical features"""
    df_copy = df[[feature, target]].copy()
    
    # Calculate distributions
    dist = pd.crosstab(df_copy[feature], df_copy[target], normalize='columns')
    
    # Calculate IV
    dist['iv'] = (dist[1] - dist[0]) * np.log(dist[1] / dist[0])
    
    return dist['iv'].sum()

# Calculate IV for all features
iv_scores = {}
for feature in categorical_features:
    iv_scores[feature] = calculate_information_value(df, feature, 'target')

iv_df = pd.DataFrame(list(iv_scores.items()), columns=['Feature', 'IV'])
```

### Stability Testing

```python
# Test feature on different data splits
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
feature_importance_across_folds = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    feature_importance_across_folds.append(model.feature_importances_)

# Check consistency across folds
stability = np.std(feature_importance_across_folds, axis=0)
unstable_features = X.columns[stability > threshold]
```

## 4. Feature Engineering Workflow

### Iterative Process

```
Initial Features
    ↓
EDA & Analysis
    ↓
Create Domain Features
    ↓
Statistical Features
    ↓
Feature Selection
    ↓
Model Training
    ↓
Evaluate & Iterate
```

## 5. Feature Documentation

### Feature Registry

```python
feature_registry = {
    'age': {
        'type': 'numerical',
        'source': 'raw',
        'description': 'Customer age in years',
        'range': [18, 100],
    },
    'income_log': {
        'type': 'numerical',
        'source': 'engineered',
        'description': 'Log transformation of annual income',
        'transformation': 'np.log1p(income)',
    },
    'engagement_index': {
        'type': 'numerical',
        'source': 'engineered',
        'description': 'Composite engagement score',
        'formula': 'logins*0.4 + posts*0.3 + shares*0.3',
    },
}
```

## Best Practices

1. ✅ Start simple, add complexity iteratively
2. ✅ Understand why each feature matters
3. ✅ Document feature logic clearly
4. ✅ Validate features before using
5. ✅ Remove low-value features
6. ✅ Avoid data leakage
7. ✅ Create reproducible feature engineering
8. ✅ Monitor feature drift in production

## Common Pitfalls

- ❌ Creating features without understanding them
- ❌ Data leakage from test set
- ❌ Overfitting to training data
- ❌ Too many features (curse of dimensionality)
- ❌ No documentation
- ❌ Features that don't generalize
- ❌ Ignoring domain knowledge

---

## Related Documents

- [Data Preprocessing](./data-preprocessing.md) - Data cleaning
- [Feature Engineering Guide](../../ml-ops/feature-engineering-guide.md) - Detailed techniques
- [Model Selection](../03-modeling/model-selection.md) - Choosing models for features

---

*Good features are more important than complex models*
