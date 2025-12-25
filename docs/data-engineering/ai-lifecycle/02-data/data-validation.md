# Data Validation

## Purpose

Verify data quality and ensure data meets requirements before model training through comprehensive validation checks.

## 1. Data Quality Dimensions

### Completeness

```python
# Missing values per feature
missing_analysis = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Percent': 100 * df.isnull().sum() / len(df),
    'Acceptable': df.isnull().sum() / len(df) <= 0.05  # 5% threshold
})

# Summary
completeness_score = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
```

### Accuracy

```python
# Verify values match expected patterns
def validate_accuracy(df):
    checks = {
        'age_range': ((df['age'] >= 18) & (df['age'] <= 120)).all(),
        'email_format': df['email'].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$').all(),
        'phone_format': df['phone'].str.match(r'^\d{10}$').all(),
        'date_valid': pd.to_datetime(df['date'], errors='coerce').notna().all(),
    }
    return checks

accuracy_checks = validate_accuracy(df)
accuracy_score = sum(accuracy_checks.values()) / len(accuracy_checks) * 100
```

### Consistency

```python
# Cross-field consistency
def validate_consistency(df):
    checks = {
        'end_after_start': (df['end_date'] >= df['start_date']).all(),
        'age_matches_dob': ((datetime.now().year - df['birth_year']) - df['age']).abs() <= 1,
        'state_valid': df['state'].isin(VALID_STATES).all(),
        'logical_ranges': (df['percentage'] >= 0) & (df['percentage'] <= 100),
    }
    return checks

consistency_checks = validate_consistency(df)
```

### Timeliness

```python
# Check data freshness
import datetime

def validate_timeliness(df):
    max_date = df['date'].max()
    days_old = (datetime.datetime.now() - max_date).days
    
    return {
        'max_date': max_date,
        'days_old': days_old,
        'is_fresh': days_old <= 1,  # Data < 1 day old
    }

freshness = validate_timeliness(df)
```

### Validity

```python
# Values within valid ranges
def validate_ranges(df):
    validations = {
        'age': (df['age'] >= 0) & (df['age'] <= 150),
        'salary': (df['salary'] >= 0) & (df['salary'] <= 1000000),
        'score': (df['score'] >= 0) & (df['score'] <= 100),
    }
    
    for field, check in validations.items():
        invalid_count = (~check).sum()
        if invalid_count > 0:
            print(f"{field}: {invalid_count} invalid values")
    
    return all(validations.values())
```

### Uniqueness

```python
# Check for duplicates
duplicates_analysis = {
    'exact_duplicates': df.duplicated().sum(),
    'by_id': df.duplicated(subset=['id']).sum(),
    'by_email': df.duplicated(subset=['email']).sum(),
}

# Flag records
df['is_duplicate'] = df.duplicated(subset=['id'], keep=False)
df['is_exact_duplicate'] = df.duplicated(keep=False)
```

## 2. Data Quality Metrics

### Automated Quality Scoring

```python
def calculate_quality_score(df):
    """Calculate overall data quality score 0-100"""
    
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
    
    # Validity checks
    validity_checks = []
    for col in df.select_dtypes(include=['number']).columns:
        valid = ((df[col] >= df[col].quantile(0.001)) & 
                (df[col] <= df[col].quantile(0.999))).sum() / len(df) * 100
        validity_checks.append(valid)
    validity = np.mean(validity_checks) if validity_checks else 100
    
    # Overall score: weighted average
    quality_score = (completeness * 0.4 + 
                    validity * 0.3 + 
                    uniqueness * 0.3)
    
    return quality_score

quality_score = calculate_quality_score(df)
print(f"Data Quality Score: {quality_score:.2f}/100")
```

## 3. Schema Validation

### Expected Schema

```python
expected_schema = {
    'customer_id': 'int64',
    'age': ('int64', 'float64'),
    'email': 'object',
    'signup_date': 'datetime64[ns]',
    'is_active': 'bool',
    'revenue': ('int64', 'float64'),
}

def validate_schema(df, schema):
    issues = []
    for col, dtype in schema.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
        elif isinstance(dtype, tuple):
            if df[col].dtype not in dtype:
                issues.append(f"{col}: expected {dtype}, got {df[col].dtype}")
        else:
            if df[col].dtype != dtype:
                issues.append(f"{col}: expected {dtype}, got {df[col].dtype}")
    
    return issues
```

### Value Range Validation

```python
# Define acceptable ranges
ranges = {
    'age': (18, 120),
    'salary': (0, 1000000),
    'score': (0, 100),
}

def validate_ranges(df, ranges):
    issues = []
    for col, (min_val, max_val) in ranges.items():
        out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
        if out_of_range > 0:
            issues.append(f"{col}: {out_of_range} values outside [{min_val}, {max_val}]")
    return issues
```

## 4. Statistical Validation

### Distribution Checks

```python
from scipy import stats

def validate_distributions(df_train, df_test):
    """Check if test distribution matches training"""
    
    issues = []
    for col in df_train.select_dtypes(include=['number']).columns:
        # Kolmogorov-Smirnov test
        stat, p_value = stats.ks_2samp(df_train[col], df_test[col])
        if p_value < 0.05:
            issues.append(f"{col}: distribution significantly different (p={p_value:.4f})")
    
    return issues
```

### Outlier Detection

```python
def validate_outliers(df, threshold=3):
    """Detect statistical outliers"""
    
    from scipy import stats
    z_scores = stats.zscore(df.select_dtypes(include=['number']))
    outliers = (z_scores > threshold).sum()
    
    for col in outliers.index:
        if outliers[col] > 0:
            print(f"{col}: {outliers[col]} outliers detected (z-score > {threshold})")
    
    return outliers
```

## 5. Business Logic Validation

### Referential Integrity

```python
def validate_referential_integrity(df):
    """Check references to other tables"""
    
    issues = []
    
    # Check customer_id exists in customers table
    unknown_customers = ~df['customer_id'].isin(customers_df['id'])
    if unknown_customers.any():
        issues.append(f"{unknown_customers.sum()} unknown customer IDs")
    
    # Check order dates in valid range
    invalid_dates = (df['order_date'] > datetime.now()).sum()
    if invalid_dates > 0:
        issues.append(f"{invalid_dates} future order dates")
    
    return issues
```

### Business Rule Validation

```python
def validate_business_rules(df):
    """Validate domain-specific business rules"""
    
    issues = []
    
    # Rule 1: Premium customers must have high credit score
    premium_low_credit = ((df['tier'] == 'premium') & (df['credit_score'] < 700)).sum()
    if premium_low_credit > 0:
        issues.append(f"{premium_low_credit} premium customers with low credit")
    
    # Rule 2: Refund amount cannot exceed purchase amount
    invalid_refunds = (df['refund_amount'] > df['purchase_amount']).sum()
    if invalid_refunds > 0:
        issues.append(f"{invalid_refunds} refunds exceeding purchase amount")
    
    return issues
```

## 6. Validation Report

### Standard Report Format

```python
class ValidationReport:
    def __init__(self, df, config):
        self.df = df
        self.config = config
        self.issues = []
        self.warnings = []
        self.quality_metrics = {}
    
    def run_all_validations(self):
        self.check_schema()
        self.check_completeness()
        self.check_ranges()
        self.check_duplicates()
        self.check_business_rules()
    
    def generate_report(self):
        report = {
            'timestamp': datetime.datetime.now(),
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'quality_score': calculate_quality_score(self.df),
            'issues': self.issues,
            'warnings': self.warnings,
            'metrics': self.quality_metrics,
            'status': 'PASS' if not self.issues else 'FAIL',
        }
        return report
    
    def export_to_json(self, filepath):
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, default=str, indent=2)

# Usage
validator = ValidationReport(df, config)
validator.run_all_validations()
report = validator.generate_report()
validator.export_to_json('validation_report.json')
```

## 7. Continuous Data Validation

### Production Monitoring

```python
# Monitor data in production
def create_data_quality_dashboard(df_recent):
    """Create monitoring metrics for production data"""
    
    metrics = {
        'timestamp': datetime.datetime.now(),
        'record_count': len(df_recent),
        'completeness': (1 - df_recent.isnull().sum() / len(df_recent)) * 100,
        'freshness_hours': (datetime.datetime.now() - df_recent['date'].max()).total_seconds() / 3600,
        'outlier_percentage': calculate_outlier_percentage(df_recent),
    }
    
    # Set alerts
    if metrics['completeness'].min() < 95:
        alert("Data completeness below threshold")
    if metrics['freshness_hours'] > 24:
        alert("Data freshness SLA missed")
    
    return metrics
```

## 8. Validation Gates

### Pre-Training Gate

```python
def validate_before_training(X, y):
    """Gate: only proceed if data passes checks"""
    
    checks = {
        'row_count': len(X) >= 1000,
        'column_count': X.shape[1] >= 5,
        'completeness': X.isnull().sum().sum() == 0,
        'no_duplicates': len(X) == len(X.drop_duplicates()),
        'target_balance': min(y.value_counts()) >= 50,
    }
    
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        raise ValueError(f"Data validation failed: {failed}")
    
    return True
```

### Pre-Deployment Gate

```python
def validate_before_deployment(df, baseline_metrics):
    """Gate: verify production data matches training profile"""
    
    checks = {
        'quality_score': calculate_quality_score(df) >= 85,
        'distribution_match': compare_distributions(df, baseline_metrics) < 0.1,
        'no_new_categories': not has_new_categories(df, baseline_metrics),
        'completeness': (1 - df.isnull().sum() / len(df)) >= 0.95,
    }
    
    if not all(checks.values()):
        raise ValueError("Production data validation failed")
    
    return True
```

## Best Practices

1. ✅ Automate validation checks
2. ✅ Define explicit acceptance criteria
3. ✅ Validate at multiple stages
4. ✅ Document validation rules
5. ✅ Monitor continuously in production
6. ✅ Track validation history
7. ✅ Alert on failures
8. ✅ Create blockers for gate failures

## Common Pitfalls

- ❌ No validation (garbage in, garbage out)
- ❌ Manual validation (not scalable)
- ❌ Validation only at the start (miss production issues)
- ❌ No clear thresholds
- ❌ Ignoring edge cases
- ❌ Missing data quality metrics

---

## Related Documents

- [Data Collection](./data-collection.md) - Gathering data
- [Data Exploration](./data-exploration.md) - Understanding data
- [Data Preprocessing](./data-preprocessing.md) - Cleaning data

---

*Validation prevents bad data from destroying model quality*
