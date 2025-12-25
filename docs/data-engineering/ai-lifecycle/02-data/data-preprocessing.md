# Data Preprocessing

## Purpose

Transform raw data into clean, consistent, and properly formatted data ready for model training.

## 1. Handling Missing Values

### Analysis & Strategy

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# Assess missing data
missing_summary = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': 100 * df.isnull().sum() / len(df),
    'Type': df.dtypes
})
```

### Imputation Methods

#### Complete Case Analysis (Drop)
```python
# Drop rows with any missing values
df_complete = df.dropna()

# Drop rows with missing in specific columns
df_cleaned = df.dropna(subset=['critical_column'])

# Drop columns with >50% missing
threshold = 0.5
cols_to_drop = missing_summary[
    missing_summary['Missing_Percent'] > threshold * 100
]['Column']
df = df.drop(columns=cols_to_drop)
```

**Use When:** <5% missing, data missing completely at random

#### Mean/Median Imputation
```python
# Numerical features - median usually better (robust to outliers)
numerical_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])

# Categorical features - mode
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
```

**Use When:** <10% missing, missing at random, limited historical data

#### Forward/Backward Fill (Time Series)
```python
# For time series data
df.sort_values('date', inplace=True)
df['value'] = df['value'].fillna(method='ffill')  # Forward fill
df['value'] = df['value'].fillna(method='bfill')  # Backward fill
```

**Use When:** Time series data with temporal patterns

#### KNN Imputation
```python
# Use K nearest neighbors
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df),
    columns=df.columns
)
```

**Use When:** Complex patterns, moderate missing data, small datasets

#### Domain Knowledge / Rule-Based
```python
# Example: Missing income might be inferred from job
df.loc[df['income'].isnull() & (df['job'] == 'CEO'), 'income'] = 150000
df.loc[df['income'].isnull() & (df['job'] == 'Manager'), 'income'] = 90000

# Missing age might be inferred from other fields
df['age'].fillna(df.groupby('generation')['age'].transform('median'), inplace=True)
```

**Use When:** Domain knowledge supports inference

### Missing Data Documentation

```python
# Document imputation decisions
imputation_decisions = {
    'age': {'method': 'median', 'value': 35},
    'income': {'method': 'knn', 'k': 5},
    'employment_status': {'method': 'mode', 'value': 'employed'},
    'optional_field': {'method': 'drop_column'},
}
```

## 2. Outlier Handling

### Detection Methods

#### Statistical Detection
```python
# Z-score (for normally distributed data)
from scipy import stats
z_scores = stats.zscore(df[numerical_cols])
outliers = (z_scores > 3).any(axis=1)

# Interquartile Range (IQR)
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['feature'] < Q1 - 1.5*IQR) | (df['feature'] > Q3 + 1.5*IQR)

# Modified Z-score (more robust)
median = df['feature'].median()
mad = np.median(np.abs(df['feature'] - median))
modified_z = 0.6745 * (df['feature'] - median) / mad
outliers = modified_z > 3.5
```

#### Domain-Based Detection
```python
# Business logic
df.loc[(df['age'] < 18) | (df['age'] > 120), 'age_outlier'] = True
df.loc[df['income'] < 0, 'income_outlier'] = True
```

### Handling Strategies

#### Cap/Floor (Winsorization)
```python
# Cap at percentiles
df['income'] = df['income'].clip(
    lower=df['income'].quantile(0.01),
    upper=df['income'].quantile(0.99)
)
```

**Use When:** Extreme values likely measurement error

#### Remove Outliers
```python
# Remove rows with outliers
df_clean = df[~outliers]

# Or flag for manual review
df['outlier_flag'] = outliers
```

**Use When:** Confirmed data quality issues

#### Transform Feature
```python
# Log transform reduces outlier impact
df['log_income'] = np.log1p(df['income'])
```

**Use When:** Right-skewed distribution with extreme values

#### Keep as Feature
```python
# Create indicator for outlier status
df['is_high_income'] = df['income'] > df['income'].quantile(0.95)
```

**Use When:** Outliers have predictive value

## 3. Duplicate Handling

### Detection & Removal

```python
# Find duplicates
df.duplicated().sum()  # Rows identical across all columns
df.duplicated(subset=['customer_id']).sum()  # By specific columns

# Remove duplicates
df = df.drop_duplicates()
df = df.drop_duplicates(subset=['customer_id'], keep='last')
```

### Decision Matrix

| Scenario | Action |
|----------|--------|
| Exact duplicates | Remove |
| Same key, different timestamp | Keep latest |
| Same person, different events | Keep (not duplicate) |
| Suspected data entry error | Investigate |

## 4. Data Type Conversion

```python
# Explicit type conversion
df['age'] = df['age'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['category'] = df['category'].astype('category')

# Infer correct types
df = df.infer_objects()

# Convert to numeric, coerce errors to NaN
df['value'] = pd.to_numeric(df['value'], errors='coerce')
```

## 5. Text Cleaning

### Standardization

```python
# Lowercase
df['category'] = df['category'].str.lower()

# Remove whitespace
df['category'] = df['category'].str.strip()

# Remove special characters
df['text'] = df['text'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# Standardize format
df['email'] = df['email'].str.lower().str.strip()
```

### Text Normalization

```python
import re
from unidecode import unidecode

def clean_text(text):
    if pd.isna(text):
        return text
    # Remove accents
    text = unidecode(text)
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

df['text'] = df['text'].apply(clean_text)
```

## 6. Feature Scaling & Normalization

### Standardization (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Result: mean=0, std=1
```

**Use When:** Algorithms sensitive to feature scale (KNN, SVM, neural networks)

### Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Result: values between 0 and 1
```

**Use When:** Need bounded values (e.g., 0-1 range)

### Robust Scaling
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Result: robust to outliers (uses median and IQR)
```

**Use When:** Data has outliers

## 7. Encoding Categorical Variables

### One-Hot Encoding
```python
# For categorical variables with low cardinality
df = pd.get_dummies(df, columns=['color', 'category'], drop_first=True)
```

**Use When:** <10 categories, tree-based models

### Ordinal Encoding
```python
# For ordinal categorical variables
order = {'low': 1, 'medium': 2, 'high': 3}
df['priority'] = df['priority'].map(order)
```

**Use When:** Natural ordering exists

### Target Encoding
```python
# Encode based on target mean
target_encoding = df.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(target_encoding)
```

**Use When:** High cardinality categorical with clear target relationship

### Frequency Encoding
```python
# Encode based on frequency
freq_encoding = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(freq_encoding)
```

**Use When:** Frequency has predictive value

## 8. Date/Time Features

### Feature Extraction

```python
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# Time since reference
reference_date = pd.Timestamp('2020-01-01')
df['days_since'] = (df['date'] - reference_date).dt.days
```

## 9. Data Consistency Checks

### Validate Relationships

```python
# Business logic validation
assert (df['end_date'] >= df['start_date']).all(), "End date before start date"
assert (df['age'] >= 0).all(), "Negative age values found"
assert df['percentage'].between(0, 100).all(), "Percentage out of range"

# Referential integrity
assert df['customer_id'].isin(customers_df['id']).all(), "Unknown customer IDs"
```

## 10. Preprocessing Pipeline

### Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

# Fit on training data
X_train_processed = pipeline.fit_transform(X_train)

# Transform test data
X_test_processed = pipeline.transform(X_test)
```

## 11. Documentation & Reproducibility

### Preprocessing Configuration

```python
preprocessing_config = {
    'missing_values': {
        'age': 'median',
        'income': 'knn',
        'category': 'mode',
    },
    'outliers': {
        'income': 'winsorize_99',
        'age': 'remove',
    },
    'scaling': {
        'numerical_cols': 'standard_scaler',
    },
    'encoding': {
        'category': 'one_hot',
        'priority': 'ordinal',
    },
}

# Save for reproducibility
import json
with open('preprocessing_config.json', 'w') as f:
    json.dump(preprocessing_config, f)
```

## Best Practices

1. ✅ Document every preprocessing decision
2. ✅ Keep raw data unchanged
3. ✅ Handle missing data explicitly
4. ✅ Detect and address outliers
5. ✅ Use domain knowledge
6. ✅ Validate data quality after each step
7. ✅ Version preprocessing code
8. ✅ Test reproducibility

## Common Pitfalls

- ❌ Data leakage (fitting on full dataset)
- ❌ Inconsistent preprocessing (train vs. test)
- ❌ Removing too much data
- ❌ Ignoring domain knowledge
- ❌ No documentation
- ❌ Hard-coded preprocessing logic

---

## Related Documents

- [Data Exploration](./data-exploration.md) - Understand data before preprocessing
- [Feature Engineering](./feature-engineering.md) - Create features after preprocessing
- [Data Validation](./data-validation.md) - Verify preprocessing quality

---

*Good preprocessing foundation enables better models*
