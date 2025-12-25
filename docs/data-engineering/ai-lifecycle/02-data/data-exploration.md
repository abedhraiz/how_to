# Data Exploration

## Purpose

Develop deep understanding of data characteristics, distributions, relationships, and patterns through exploratory data analysis (EDA).

## 1. Data Overview & Summary

### Dataset Dimensions
```python
import pandas as pd

df = pd.read_parquet('data/raw/dataset.parquet')

# Basic info
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

**Report:**
- **Total Records:** [Number of rows]
- **Total Features:** [Number of columns]
- **Time Span:** [Date range]
- **Data Size:** [Total size in MB/GB]

### Data Types
```
Integer:     [columns]
Float:       [columns]
String:      [columns]
DateTime:    [columns]
Boolean:     [columns]
Categorical: [columns]
```

## 2. Univariate Analysis

### Numerical Features

#### Distribution Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df['age'].hist(ax=axes[0, 0], bins=50)
df['income'].hist(ax=axes[0, 1], bins=50)
df['credit_score'].hist(ax=axes[1, 0], bins=50)
df['tenure'].hist(ax=axes[1, 1], bins=50)
plt.show()

# Box plot for outliers
sns.boxplot(data=df, y='income')
plt.show()
```

**Summary Statistics:**
```
Feature: age
  Count:    10,000
  Mean:     35.2
  Std:      12.4
  Min:      18
  25%:      26
  50%:      33
  75%:      44
  Max:      65
  Skewness: 0.3
  Kurtosis: -0.5
```

#### Missing Data Analysis
```python
# Missing value analysis
missing = df.isnull().sum()
missing_pct = 100 * missing / len(df)

pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Pct': missing_pct
}).sort_values('Missing_Pct', ascending=False)
```

**Report:**
- **Total Missing:** [X values]
- **Missing %:** [X%]
- **Columns with Missing:** [List]
- **Missing Pattern:** [Random, systematic, etc.]

### Categorical Features

#### Frequency Analysis
```python
# Value counts
print(df['category'].value_counts())
print(df['category'].value_counts(normalize=True))

# Visualize
df['category'].value_counts().plot(kind='bar')
plt.title('Category Distribution')
plt.show()
```

**Report:**
```
Feature: gender
  Unique Values:  3
  
  Value         Count    Percentage
  ─────────────────────────────────
  Male          5,234    52.3%
  Female        4,512    45.1%
  Other         254      2.5%
```

#### Imbalance Analysis
```python
# Check class balance
balance = df['target'].value_counts(normalize=True)
print(balance)

# Imbalance ratio
if balance.min() > 0:
    imbalance_ratio = balance.max() / balance.min()
    print(f"Imbalance Ratio: {imbalance_ratio:.2f}")
```

## 3. Bivariate Analysis

### Feature-Target Relationships

#### Correlation Analysis
```python
# Numerical features
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# Feature-target correlation
target_corr = df[numerical_features].corrwith(df['target']).sort_values(ascending=False)
```

**Report:**
```
Feature          Correlation with Target
──────────────────────────────────────────
income                 0.62
credit_score           0.58
age                    0.45
tenure                 0.32
transactions           0.28
```

#### Target Distribution by Features
```python
# Box plot: numerical feature vs target
fig, axes = plt.subplots(1, 2)
df.boxplot(column='income', by='target', ax=axes[0])
df.boxplot(column='credit_score', by='target', ax=axes[1])
plt.show()

# Bar plot: categorical feature vs target
pd.crosstab(df['gender'], df['target'], normalize='columns').plot(kind='bar')
plt.title('Target Distribution by Gender')
plt.show()
```

### Feature-Feature Relationships

```python
# Scatter plots
sns.pairplot(df[['age', 'income', 'credit_score', 'target']], 
             hue='target', diag_kind='kde')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.show()
```

## 4. Multivariate Analysis

### Clustering & Grouping

```python
# Group analysis
group_stats = df.groupby('segment').agg({
    'income': 'mean',
    'age': 'mean',
    'credit_score': 'mean',
    'target': 'mean'
})
```

### Interactions

```python
# Feature interactions
df['income_age_interaction'] = df['income'] * df['age']
df['high_income_young'] = (df['income'] > df['income'].median()) & \
                           (df['age'] < df['age'].median())
```

## 5. Temporal Analysis

### Time Series Exploration

```python
# Time series plot
df.set_index('date').resample('D')['target'].mean().plot()
plt.title('Daily Average Target')
plt.xlabel('Date')
plt.ylabel('Target Value')
plt.show()

# Trend analysis
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df.set_index('date')['target'], period=365)
decomposition.plot()
plt.show()
```

**Analysis:**
- **Trend:** [Increasing/Decreasing/Stable]
- **Seasonality:** [Yes/No - what pattern?]
- **Stationarity:** [Yes/No - ADF test result]
- **Anomalies:** [Notable spikes/dips]

## 6. Outlier Analysis

### Outlier Detection

```python
# Statistical approach (Z-score)
from scipy import stats
z_scores = stats.zscore(df[numerical_features])
outliers = (z_scores > 3).any(axis=1)

# IQR approach
Q1 = df['income'].quantile(0.25)
Q3 = df['income'].quantile(0.75)
IQR = Q3 - Q1
outliers = (df['income'] < Q1 - 1.5*IQR) | (df['income'] > Q3 + 1.5*IQR)

print(f"Outlier Percentage: {100 * outliers.sum() / len(df):.2f}%")
```

**Report:**
```
Feature          Outlier Count   Outlier %
──────────────────────────────────────────
income                   234        2.3%
credit_score             156        1.6%
transactions             345        3.4%
```

### Outlier Handling Decision
- **Keep:** Natural variation in data
- **Investigate:** Potential data errors
- **Cap:** Extreme values only
- **Remove:** Data quality issues

## 7. Data Quality Assessment

### Quality Metrics

```python
# Completeness
completeness = (1 - df.isnull().sum() / len(df)) * 100

# Uniqueness
uniqueness = df.nunique() / len(df) * 100

# Validity
# [Define what's valid for each feature]

# Consistency
# [Check for logical inconsistencies]
```

**Report:**
```
Feature                  Completeness   Quality Issues
────────────────────────────────────────────────────
age                         100%         None
income                       98%         2% missing
credit_score                 99%         2 outliers detected
employment_status           100%         None
```

## 8. Key Findings Summary

### Insights
1. **Key Finding 1:** [Description and implication]
2. **Key Finding 2:** [Description and implication]
3. **Key Finding 3:** [Description and implication]

### Anomalies & Concerns
- **Concern 1:** [Description and action needed]
- **Concern 2:** [Description and action needed]

### Feature Importance (Preliminary)
Based on correlation and information value:
1. [Feature] - [Reason]
2. [Feature] - [Reason]
3. [Feature] - [Reason]

## 9. Recommendations for Preprocessing

### Handling Missing Values
```python
# Document decisions
missing_handling = {
    'age': 'median_imputation',      # Low missing rate
    'income': 'forward_fill',         # Temporal data
    'category': 'mode_imputation',    # Categorical
    'optional_field': 'drop_column',  # >50% missing
}
```

### Handling Outliers
- **outlier_feature1:** Cap at 95th percentile
- **outlier_feature2:** Domain knowledge review needed
- **outlier_feature3:** Remove (data quality issue)

### Feature Transformations Needed
- **income:** Log transformation (right-skewed)
- **age:** No transformation needed
- **count_features:** Square root transformation

### Categorical Encoding Strategy
- **high_cardinality_feature:** Target encoding
- **binary_feature:** One-hot encoding
- **ordinal_feature:** Ordinal encoding

## 10. EDA Report Template

**Executive Summary**
- Dataset overview
- Key findings
- Recommendations

**Detailed Analysis**
- Univariate analysis with visualizations
- Bivariate analysis with key relationships
- Multivariate analysis and interactions
- Data quality assessment
- Anomalies and concerns

**Recommendations for Next Steps**
- Data preprocessing actions
- Feature engineering ideas
- Potential modeling approaches

## Best Practices

1. ✅ Visualize extensively
2. ✅ Look for patterns and anomalies
3. ✅ Document all findings
4. ✅ Involve domain experts
5. ✅ Consider business context
6. ✅ Generate hypotheses for modeling
7. ✅ Record all decisions

## Common Pitfalls

- ❌ Skipping EDA (missing insights)
- ❌ No visualizations (hard to spot patterns)
- ❌ Ignoring domain knowledge
- ❌ Overfitting to training data patterns
- ❌ Not documenting findings
- ❌ Making decisions without analysis

---

## Related Documents

- [Data Collection](./data-collection.md) - Data gathering
- [Data Preprocessing](./data-preprocessing.md) - Cleaning data
- [Feature Engineering](./feature-engineering.md) - Creating features
- [Dataset Card](../templates/dataset-card.md) - Dataset documentation

---

*Thorough EDA uncovers patterns, guides preprocessing, and informs feature engineering*
