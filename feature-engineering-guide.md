# Feature Engineering Guide

## What is Feature Engineering?

Feature engineering is the process of using domain knowledge to create, transform, and select features (input variables) that make machine learning algorithms work more effectively. It's often considered one of the most important steps in building successful machine learning models.

**Why Feature Engineering Matters:**
- Improves model accuracy
- Reduces training time
- Enhances model interpretability
- Captures domain knowledge
- Handles missing data
- Reduces overfitting

## Prerequisites

- Python programming
- Basic statistics and mathematics
- Understanding of machine learning concepts
- Familiarity with pandas and NumPy
- Domain knowledge of your data

## Installation

```bash
# Essential libraries
pip install pandas numpy scikit-learn

# Feature engineering libraries
pip install feature-engine category-encoders

# Visualization
pip install matplotlib seaborn plotly

# Advanced feature engineering
pip install featuretools tsfresh

# Model interpretation
pip install shap eli5

# All at once
pip install pandas numpy scikit-learn feature-engine category-encoders \
            matplotlib seaborn plotly featuretools tsfresh shap eli5
```

## Core Concepts

### Types of Features

1. **Numerical Features**: Continuous or discrete numbers
2. **Categorical Features**: Finite set of discrete values
3. **Ordinal Features**: Categorical with meaningful order
4. **Binary Features**: Boolean or 0/1 values
5. **Datetime Features**: Date and time information
6. **Text Features**: Unstructured text data
7. **Geospatial Features**: Location coordinates

## Numerical Feature Transformations

### Scaling and Normalization

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 75000, 90000, 120000],
    'years_experience': [2, 5, 8, 12, 15]
})

# Standardization (z-score normalization)
scaler = StandardScaler()
data_standardized = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)
print("Standardized:\n", data_standardized)

# Min-Max Scaling (0-1 range)
minmax_scaler = MinMaxScaler()
data_minmax = pd.DataFrame(
    minmax_scaler.fit_transform(data),
    columns=data.columns
)
print("\nMin-Max Scaled:\n", data_minmax)

# Robust Scaling (resistant to outliers)
robust_scaler = RobustScaler()
data_robust = pd.DataFrame(
    robust_scaler.fit_transform(data),
    columns=data.columns
)
print("\nRobust Scaled:\n", data_robust)

# Log transformation
data['salary_log'] = np.log1p(data['salary'])
print("\nLog transformed salary:\n", data[['salary', 'salary_log']])
```

### Binning and Discretization

```python
# Equal-width binning
data['age_binned'] = pd.cut(
    data['age'],
    bins=3,
    labels=['Young', 'Middle', 'Senior']
)

# Equal-frequency binning
data['salary_binned'] = pd.qcut(
    data['salary'],
    q=3,
    labels=['Low', 'Medium', 'High']
)

# Custom bins
salary_bins = [0, 60000, 80000, 150000]
data['salary_category'] = pd.cut(
    data['salary'],
    bins=salary_bins,
    labels=['Entry', 'Mid', 'Senior']
)

print(data[['age', 'age_binned', 'salary', 'salary_category']])
```

### Power Transformations

```python
from sklearn.preprocessing import PowerTransformer

# Box-Cox transformation (positive values only)
pt_boxcox = PowerTransformer(method='box-cox')
data['salary_boxcox'] = pt_boxcox.fit_transform(data[['salary']])

# Yeo-Johnson transformation (works with negative values)
pt_yeo = PowerTransformer(method='yeo-johnson')
data['salary_yeojohnson'] = pt_yeo.fit_transform(data[['salary']])

# Square root transformation
data['salary_sqrt'] = np.sqrt(data['salary'])

# Square transformation
data['age_squared'] = data['age'] ** 2

print(data[['salary', 'salary_boxcox', 'salary_yeojohnson', 'salary_sqrt']])
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X = data[['age', 'years_experience']]
poly_features = poly.fit_transform(X)

# Get feature names
feature_names = poly.get_feature_names_out(['age', 'years_experience'])
poly_df = pd.DataFrame(poly_features, columns=feature_names)

print("Original features:")
print(X.head())
print("\nPolynomial features:")
print(poly_df.head())
```

### Interaction Features

```python
# Manual interaction features
data['age_experience_interaction'] = data['age'] * data['years_experience']
data['age_experience_ratio'] = data['age'] / (data['years_experience'] + 1)

# Multiple interactions
data['salary_per_experience'] = data['salary'] / (data['years_experience'] + 1)
data['age_salary_interaction'] = data['age'] * data['salary']

print(data[['age', 'years_experience', 'age_experience_interaction']])
```

## Categorical Feature Encoding

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# Sample data
df = pd.DataFrame({
    'city': ['New York', 'London', 'Paris', 'Tokyo', 'New York', 'London'],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 'HR'],
    'performance': ['Good', 'Excellent', 'Average', 'Good', 'Excellent', 'Average']
})

# Label encoding
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
df['department_encoded'] = le.fit_transform(df['department'])

print("Label Encoded:")
print(df[['city', 'city_encoded', 'department', 'department_encoded']])
```

### One-Hot Encoding

```python
# One-hot encoding with pandas
df_onehot = pd.get_dummies(df, columns=['city', 'department'], prefix=['city', 'dept'])
print("\nOne-Hot Encoded:")
print(df_onehot.head())

# One-hot encoding with sklearn
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid multicollinearity
city_encoded = ohe.fit_transform(df[['city']])
city_columns = ohe.get_feature_names_out(['city'])
df_city_encoded = pd.DataFrame(city_encoded, columns=city_columns)

print("\nScikit-learn One-Hot Encoded:")
print(df_city_encoded.head())
```

### Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
performance_order = ['Average', 'Good', 'Excellent']

# Ordinal encoding
ordinal_encoder = OrdinalEncoder(categories=[performance_order])
df['performance_encoded'] = ordinal_encoder.fit_transform(df[['performance']])

print("Ordinal Encoded:")
print(df[['performance', 'performance_encoded']])

# Manual ordinal encoding
performance_mapping = {'Average': 1, 'Good': 2, 'Excellent': 3}
df['performance_manual'] = df['performance'].map(performance_mapping)
```

### Target Encoding (Mean Encoding)

```python
import category_encoders as ce

# Sample data with target
df_target = pd.DataFrame({
    'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B'],
    'target': [1, 0, 1, 1, 0, 1, 0, 1]
})

# Target encoding
target_encoder = ce.TargetEncoder(cols=['category'])
df_target['category_encoded'] = target_encoder.fit_transform(
    df_target['category'],
    df_target['target']
)

print("Target Encoded:")
print(df_target)

# Manual target encoding
target_means = df_target.groupby('category')['target'].mean()
df_target['category_manual'] = df_target['category'].map(target_means)
print("\nManual Target Encoding:")
print(df_target)
```

### Frequency Encoding

```python
# Count/Frequency encoding
df['city_frequency'] = df['city'].map(df['city'].value_counts())
df['city_frequency_normalized'] = df['city'].map(
    df['city'].value_counts(normalize=True)
)

print("Frequency Encoded:")
print(df[['city', 'city_frequency', 'city_frequency_normalized']])
```

### Binary Encoding

```python
# Binary encoding for high cardinality features
binary_encoder = ce.BinaryEncoder(cols=['city'])
df_binary = binary_encoder.fit_transform(df[['city']])

print("Binary Encoded:")
print(df_binary)
```

### Hash Encoding

```python
# Hash encoding for very high cardinality
hash_encoder = ce.HashingEncoder(cols=['city'], n_components=4)
df_hashed = hash_encoder.fit_transform(df[['city']])

print("Hash Encoded:")
print(df_hashed)
```

## Handling Missing Values

### Imputation Techniques

```python
from sklearn.impute import SimpleImputer, KNNImputer

# Sample data with missing values
data_missing = pd.DataFrame({
    'age': [25, np.nan, 35, 40, np.nan],
    'salary': [50000, 60000, np.nan, 90000, 120000],
    'score': [85, 90, 88, np.nan, 92]
})

print("Original data with missing values:")
print(data_missing)

# Mean imputation
mean_imputer = SimpleImputer(strategy='mean')
data_mean = pd.DataFrame(
    mean_imputer.fit_transform(data_missing),
    columns=data_missing.columns
)
print("\nMean Imputation:")
print(data_mean)

# Median imputation (robust to outliers)
median_imputer = SimpleImputer(strategy='median')
data_median = pd.DataFrame(
    median_imputer.fit_transform(data_missing),
    columns=data_missing.columns
)

# Mode imputation (for categorical)
mode_imputer = SimpleImputer(strategy='most_frequent')

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=2)
data_knn = pd.DataFrame(
    knn_imputer.fit_transform(data_missing),
    columns=data_missing.columns
)
print("\nKNN Imputation:")
print(data_knn)

# Forward fill
data_ffill = data_missing.fillna(method='ffill')

# Backward fill
data_bfill = data_missing.fillna(method='bfill')

# Interpolation
data_interpolate = data_missing.interpolate(method='linear')
```

### Missing Indicator

```python
from sklearn.impute import MissingIndicator

# Create missing indicators
missing_indicator = MissingIndicator()
missing_mask = missing_indicator.fit_transform(data_missing)

# Add as features
data_with_indicators = data_missing.copy()
data_with_indicators['age_missing'] = missing_mask[:, 0]
data_with_indicators['salary_missing'] = missing_mask[:, 1]
data_with_indicators['score_missing'] = missing_mask[:, 2]

print("Data with missing indicators:")
print(data_with_indicators)
```

## Feature Selection

### Filter Methods

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import load_iris

# Load sample data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

# Feature scores
scores = pd.DataFrame({
    'feature': X.columns,
    'score': selector.scores_
}).sort_values('score', ascending=False)
print("\nFeature scores:")
print(scores)

# Mutual information
mi_selector = SelectKBest(score_func=mutual_info_classif, k=2)
mi_selector.fit(X, y)
mi_scores = pd.DataFrame({
    'feature': X.columns,
    'mi_score': mi_selector.scores_
}).sort_values('mi_score', ascending=False)
print("\nMutual Information scores:")
print(mi_scores)
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=2)
rfe.fit(X, y)

# Get selected features
selected_features_rfe = X.columns[rfe.support_].tolist()
print("RFE Selected features:", selected_features_rfe)

# Feature ranking
feature_ranking = pd.DataFrame({
    'feature': X.columns,
    'ranking': rfe.ranking_
}).sort_values('ranking')
print("\nFeature ranking:")
print(feature_ranking)
```

### Embedded Methods

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV

# Random Forest feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("Random Forest Feature Importance:")
print(feature_importance)

# Lasso for feature selection
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

lasso_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': np.abs(lasso.coef_)
}).sort_values('coefficient', ascending=False)

print("\nLasso Feature Importance:")
print(lasso_importance)
```

### Correlation Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Create correlation matrix
correlation_matrix = X.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')

# Find highly correlated features
threshold = 0.8
high_corr = []

for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            high_corr.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })

if high_corr:
    print("Highly correlated features:")
    print(pd.DataFrame(high_corr))
```

### Variance Threshold

```python
from sklearn.feature_selection import VarianceThreshold

# Remove low variance features
variance_selector = VarianceThreshold(threshold=0.1)
X_high_variance = variance_selector.fit_transform(X)

# Get remaining features
remaining_features = X.columns[variance_selector.get_support()].tolist()
print("Features after variance threshold:", remaining_features)

# Feature variances
variances = pd.DataFrame({
    'feature': X.columns,
    'variance': X.var()
}).sort_values('variance', ascending=False)
print("\nFeature variances:")
print(variances)
```

## Datetime Feature Engineering

```python
import pandas as pd
from datetime import datetime, timedelta

# Sample datetime data
dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
df_time = pd.DataFrame({'date': dates})

# Add random sales data
np.random.seed(42)
df_time['sales'] = np.random.randint(100, 1000, size=100)

# Extract date components
df_time['year'] = df_time['date'].dt.year
df_time['month'] = df_time['date'].dt.month
df_time['day'] = df_time['date'].dt.day
df_time['day_of_week'] = df_time['date'].dt.dayofweek
df_time['day_of_year'] = df_time['date'].dt.dayofyear
df_time['week_of_year'] = df_time['date'].dt.isocalendar().week
df_time['quarter'] = df_time['date'].dt.quarter

# Time-based features
df_time['is_weekend'] = df_time['day_of_week'].isin([5, 6]).astype(int)
df_time['is_month_start'] = df_time['date'].dt.is_month_start.astype(int)
df_time['is_month_end'] = df_time['date'].dt.is_month_end.astype(int)
df_time['is_quarter_start'] = df_time['date'].dt.is_quarter_start.astype(int)

# Cyclical encoding (for periodic features)
df_time['month_sin'] = np.sin(2 * np.pi * df_time['month'] / 12)
df_time['month_cos'] = np.cos(2 * np.pi * df_time['month'] / 12)
df_time['day_of_week_sin'] = np.sin(2 * np.pi * df_time['day_of_week'] / 7)
df_time['day_of_week_cos'] = np.cos(2 * np.pi * df_time['day_of_week'] / 7)

# Time since reference
reference_date = pd.Timestamp('2024-01-01')
df_time['days_since_start'] = (df_time['date'] - reference_date).dt.days

# Lag features
df_time['sales_lag_1'] = df_time['sales'].shift(1)
df_time['sales_lag_7'] = df_time['sales'].shift(7)

# Rolling window features
df_time['sales_rolling_mean_7'] = df_time['sales'].rolling(window=7).mean()
df_time['sales_rolling_std_7'] = df_time['sales'].rolling(window=7).std()
df_time['sales_rolling_min_7'] = df_time['sales'].rolling(window=7).min()
df_time['sales_rolling_max_7'] = df_time['sales'].rolling(window=7).max()

# Expanding window features
df_time['sales_expanding_mean'] = df_time['sales'].expanding().mean()

# Difference features
df_time['sales_diff_1'] = df_time['sales'].diff(1)
df_time['sales_diff_7'] = df_time['sales'].diff(7)

print(df_time.head(10))
```

## Text Feature Engineering

### Basic Text Features

```python
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample text data
texts = [
    "Machine learning is amazing!",
    "Deep learning models are powerful.",
    "Natural language processing is fascinating.",
    "AI and machine learning are the future.",
    "Data science combines statistics and programming."
]

df_text = pd.DataFrame({'text': texts})

# Basic text statistics
df_text['char_count'] = df_text['text'].apply(len)
df_text['word_count'] = df_text['text'].apply(lambda x: len(x.split()))
df_text['avg_word_length'] = df_text['text'].apply(
    lambda x: np.mean([len(word) for word in x.split()])
)
df_text['uppercase_count'] = df_text['text'].apply(
    lambda x: sum(1 for c in x if c.isupper())
)
df_text['punctuation_count'] = df_text['text'].apply(
    lambda x: sum(1 for c in x if c in '.,!?;:')
)

# Sentiment indicators
df_text['exclamation_count'] = df_text['text'].str.count('!')
df_text['question_count'] = df_text['text'].str.count('\?')

print("Basic Text Features:")
print(df_text)
```

### Bag of Words

```python
# Count Vectorizer (Bag of Words)
count_vectorizer = CountVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2)  # unigrams and bigrams
)

bow_features = count_vectorizer.fit_transform(df_text['text'])
bow_df = pd.DataFrame(
    bow_features.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)

print("\nBag of Words Features:")
print(bow_df)
```

### TF-IDF

```python
# TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2)
)

tfidf_features = tfidf_vectorizer.fit_transform(df_text['text'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(),
    columns=tfidf_vectorizer.get_feature_names_out()
)

print("\nTF-IDF Features:")
print(tfidf_df.head())
```

### Word Embeddings

```python
# Using pre-trained embeddings (example with sentence transformers)
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(df_text['text'].tolist())

# Create DataFrame with embeddings
embedding_df = pd.DataFrame(
    embeddings,
    columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
)

print("\nText Embeddings shape:", embedding_df.shape)
print(embedding_df.head())
```

## Automated Feature Engineering

### Featuretools

```python
import featuretools as ft

# Sample relational data
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'age': [25, 30, 35, 40, 45],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA']
})

transactions = pd.DataFrame({
    'transaction_id': range(1, 11),
    'customer_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'amount': [100, 150, 200, 80, 120, 90, 300, 250, 180, 220],
    'timestamp': pd.date_range('2024-01-01', periods=10, freq='D')
})

# Create EntitySet
es = ft.EntitySet(id='customer_data')

# Add entities (tables)
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers,
    index='customer_id'
)

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions,
    index='transaction_id',
    time_index='timestamp'
)

# Add relationship
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Generate features
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2,
    verbose=True
)

print("\nAutomatically generated features:")
print(feature_matrix)
```

## Domain-Specific Features

### E-commerce Features

```python
# Sample e-commerce data
ecommerce_data = pd.DataFrame({
    'customer_id': range(1, 101),
    'total_orders': np.random.randint(1, 50, 100),
    'total_spent': np.random.uniform(100, 5000, 100),
    'days_since_first_order': np.random.randint(1, 365, 100),
    'days_since_last_order': np.random.randint(1, 90, 100),
    'avg_order_value': np.random.uniform(20, 500, 100),
    'returned_orders': np.random.randint(0, 10, 100)
})

# Feature engineering
ecommerce_data['order_frequency'] = (
    ecommerce_data['total_orders'] / 
    ecommerce_data['days_since_first_order'] * 30  # Orders per month
)

ecommerce_data['customer_lifetime_value'] = (
    ecommerce_data['total_spent'] / 
    ecommerce_data['days_since_first_order'] * 365  # Annual CLV
)

ecommerce_data['return_rate'] = (
    ecommerce_data['returned_orders'] / 
    ecommerce_data['total_orders']
)

ecommerce_data['recency_score'] = 1 / (ecommerce_data['days_since_last_order'] + 1)

ecommerce_data['is_active'] = (
    ecommerce_data['days_since_last_order'] < 30
).astype(int)

ecommerce_data['customer_segment'] = pd.cut(
    ecommerce_data['total_spent'],
    bins=[0, 500, 2000, 10000],
    labels=['Low', 'Medium', 'High']
)

print("E-commerce Features:")
print(ecommerce_data.head())
```

### Financial Features

```python
# Sample financial data
financial_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100, freq='D'),
    'stock_price': np.random.uniform(100, 200, 100),
    'volume': np.random.randint(1000000, 10000000, 100)
})

# Technical indicators
financial_data['price_change'] = financial_data['stock_price'].diff()
financial_data['price_change_pct'] = financial_data['stock_price'].pct_change()

# Moving averages
financial_data['ma_7'] = financial_data['stock_price'].rolling(window=7).mean()
financial_data['ma_30'] = financial_data['stock_price'].rolling(window=30).mean()

# Exponential moving average
financial_data['ema_12'] = financial_data['stock_price'].ewm(span=12).mean()
financial_data['ema_26'] = financial_data['stock_price'].ewm(span=26).mean()

# MACD
financial_data['macd'] = financial_data['ema_12'] - financial_data['ema_26']

# Bollinger Bands
rolling_mean = financial_data['stock_price'].rolling(window=20).mean()
rolling_std = financial_data['stock_price'].rolling(window=20).std()
financial_data['bollinger_upper'] = rolling_mean + (rolling_std * 2)
financial_data['bollinger_lower'] = rolling_mean - (rolling_std * 2)

# RSI (Relative Strength Index)
delta = financial_data['stock_price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
financial_data['rsi'] = 100 - (100 / (1 + rs))

# Volatility
financial_data['volatility'] = financial_data['stock_price'].rolling(window=20).std()

print("Financial Features:")
print(financial_data.tail())
```

## Feature Engineering Pipeline

### Complete Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Sample dataset
data = pd.DataFrame({
    'age': [25, 30, np.nan, 40, 45],
    'salary': [50000, 60000, 75000, np.nan, 120000],
    'city': ['NYC', 'LA', 'Chicago', 'NYC', 'LA'],
    'department': ['Sales', 'IT', 'HR', 'Sales', 'IT'],
    'experience': [2, 5, 8, 12, 15]
})

# Define numeric and categorical features
numeric_features = ['age', 'salary', 'experience']
categorical_features = ['city', 'department']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform
data_transformed = preprocessor.fit_transform(data)

print("Transformed data shape:", data_transformed.shape)
print("Transformed data:")
print(data_transformed)
```

### Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for domain-specific feature engineering"""
    
    def __init__(self, create_interactions=True):
        self.create_interactions = create_interactions
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Create age groups
        X['age_group'] = pd.cut(
            X['age'],
            bins=[0, 25, 35, 50, 100],
            labels=['Young', 'Adult', 'Middle', 'Senior']
        )
        
        # Salary bins
        X['salary_category'] = pd.qcut(
            X['salary'],
            q=3,
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )
        
        # Experience to age ratio
        X['exp_age_ratio'] = X['experience'] / X['age']
        
        # Salary per experience
        X['salary_per_exp'] = X['salary'] / (X['experience'] + 1)
        
        if self.create_interactions:
            X['age_exp_interaction'] = X['age'] * X['experience']
        
        return X

# Use custom transformer
custom_engineer = DomainFeatureEngineer(create_interactions=True)
data_engineered = custom_engineer.fit_transform(data)

print("Engineered features:")
print(data_engineered.head())
```

## Best Practices

### 1. Feature Engineering Workflow

```python
class FeatureEngineeringWorkflow:
    """Complete feature engineering workflow"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.feature_stats = {}
    
    def analyze_features(self):
        """Analyze feature distributions and statistics"""
        print("=== Feature Analysis ===")
        print(f"Shape: {self.data.shape}")
        print(f"\nData types:\n{self.data.dtypes}")
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nBasic statistics:\n{self.data.describe()}")
        
        return self
    
    def handle_missing_values(self, strategy='auto'):
        """Handle missing values"""
        print("\n=== Handling Missing Values ===")
        
        for col in self.data.columns:
            if self.data[col].isnull().sum() > 0:
                if self.data[col].dtype in ['int64', 'float64']:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        return self
    
    def engineer_features(self):
        """Create new features"""
        print("\n=== Engineering Features ===")
        
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Log transformations for skewed features
        for col in numeric_cols:
            if self.data[col].skew() > 1:
                self.data[f'{col}_log'] = np.log1p(self.data[col])
        
        # Polynomial features for numeric columns
        if len(numeric_cols) >= 2:
            col1, col2 = list(numeric_cols)[:2]
            self.data[f'{col1}_{col2}_interaction'] = self.data[col1] * self.data[col2]
        
        return self
    
    def select_features(self, target, k=10):
        """Select top k features"""
        print(f"\n=== Selecting Top {k} Features ===")
        
        numeric_features = self.data.select_dtypes(include=['int64', 'float64']).columns
        X = self.data[numeric_features]
        y = self.data[target]
        
        selector = SelectKBest(score_func=f_classif, k=min(k, len(numeric_features)))
        selector.fit(X, y)
        
        selected = X.columns[selector.get_support()].tolist()
        print(f"Selected features: {selected}")
        
        return self.data[selected + [target]]
    
    def get_data(self):
        """Return engineered data"""
        return self.data

# Usage
# workflow = FeatureEngineeringWorkflow(data)
# result = workflow.analyze_features() \
#                  .handle_missing_values() \
#                  .engineer_features() \
#                  .get_data()
```

### 2. Avoid Data Leakage

```python
from sklearn.model_selection import train_test_split

# WRONG: Fit on entire dataset
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data)
# X_train, X_test = train_test_split(data_scaled)

# CORRECT: Fit only on training data
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit
```

### 3. Feature Importance Analysis

```python
from sklearn.ensemble import RandomForestClassifier
import shap

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Important Features:")
print(feature_importance.head(10))

# SHAP values for model interpretation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test, plot_type="bar")
```

### 4. Feature Validation

```python
def validate_features(df, expected_dtypes=None, value_ranges=None):
    """Validate engineered features"""
    
    issues = []
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            issues.append(f"{col} contains infinite values")
    
    # Check for NaN values
    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        issues.append(f"Missing values: {null_counts[null_counts > 0].to_dict()}")
    
    # Check data types
    if expected_dtypes:
        for col, expected_type in expected_dtypes.items():
            if col in df.columns and df[col].dtype != expected_type:
                issues.append(f"{col} has type {df[col].dtype}, expected {expected_type}")
    
    # Check value ranges
    if value_ranges:
        for col, (min_val, max_val) in value_ranges.items():
            if col in df.columns:
                if df[col].min() < min_val or df[col].max() > max_val:
                    issues.append(f"{col} values outside range [{min_val}, {max_val}]")
    
    if issues:
        print("Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("All validations passed!")
        return True

# Example usage
# validate_features(
#     df,
#     expected_dtypes={'age': 'int64', 'salary': 'float64'},
#     value_ranges={'age': (0, 120), 'salary': (0, 1000000)}
# )
```

## Common Pitfalls to Avoid

### 1. Data Leakage

```python
# BAD: Using future information
df['target_mean'] = df.groupby('category')['target'].transform('mean')

# GOOD: Use target encoding properly with cross-validation
from category_encoders import TargetEncoder
encoder = TargetEncoder()
# Fit only on training data, transform both train and test
```

### 2. Overfitting with Too Many Features

```python
# Monitor feature to sample ratio
def check_feature_ratio(X, min_ratio=10):
    """Check if enough samples per feature"""
    n_samples, n_features = X.shape
    ratio = n_samples / n_features
    
    if ratio < min_ratio:
        print(f"Warning: Low sample to feature ratio ({ratio:.1f})")
        print(f"Consider reducing features or gathering more data")
    else:
        print(f"Sample to feature ratio: {ratio:.1f} (Good)")
    
    return ratio

# check_feature_ratio(X_train)
```

### 3. Inconsistent Transformations

```python
# Create a feature engineering config
feature_config = {
    'numeric_features': ['age', 'salary', 'experience'],
    'categorical_features': ['city', 'department'],
    'date_features': ['hire_date'],
    'transformations': {
        'age': 'standard_scale',
        'salary': 'log_transform',
        'city': 'one_hot_encode'
    }
}

# Apply same transformations to train and test
def apply_transformations(df, config, fitted_transformers=None):
    """Apply consistent transformations"""
    transformers = fitted_transformers or {}
    
    for feature, transformation in config['transformations'].items():
        if transformation == 'standard_scale':
            if feature not in transformers:
                transformers[feature] = StandardScaler()
                df[feature] = transformers[feature].fit_transform(df[[feature]])
            else:
                df[feature] = transformers[feature].transform(df[[feature]])
    
    return df, transformers
```

## Resources

- **Scikit-learn**: https://scikit-learn.org/
- **Feature-engine**: https://feature-engine.readthedocs.io/
- **Featuretools**: https://www.featuretools.com/
- **Category Encoders**: https://contrib.scikit-learn.org/category_encoders/
- **TSFRESH**: https://tsfresh.readthedocs.io/ (Time series features)

## Quick Reference

### Common Transformations

```python
# Numeric transformations
StandardScaler()  # z-score normalization
MinMaxScaler()  # 0-1 scaling
RobustScaler()  # robust to outliers
np.log1p()  # log transformation
PowerTransformer()  # Box-Cox, Yeo-Johnson

# Categorical encoding
LabelEncoder()  # ordinal encoding
OneHotEncoder()  # one-hot encoding
TargetEncoder()  # target/mean encoding
BinaryEncoder()  # binary encoding

# Feature selection
SelectKBest()  # univariate selection
RFE()  # recursive feature elimination
feature_importances_  # model-based selection

# Missing values
SimpleImputer()  # mean/median/mode
KNNImputer()  # KNN-based imputation
fillna()  # pandas methods
```

---

*This guide covers essential feature engineering techniques for building better machine learning models. Practice on real datasets to develop your feature engineering intuition!*
