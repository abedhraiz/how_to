# Dataset Card Template

## Purpose

Provide a structured overview of datasets used in AI/ML projects for transparency, reproducibility, and governance.

Based on [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)

---

## Dataset Card

### 1. Dataset Overview

**Dataset Name:** [e.g., "LoanPerformance-2015-2022"]

**Version:** [e.g., "1.0"]

**Release Date:** [YYYY-MM-DD]

**Last Updated:** [YYYY-MM-DD]

**Curator:** [Person/Team]

**License:** [e.g., "CC BY 4.0", "Proprietary"]

**Dataset Size:**
- Rows: [Number of samples]
- Columns: [Number of features]
- Storage: [MB/GB]

**Dataset Description:**

[2-3 paragraph description of dataset]

Example:
"This dataset contains loan application and performance data for a major US bank from 2015-2022. It includes applicant characteristics, loan terms, and performance outcomes (default/prepayment). The dataset is used for credit risk modeling and underwriting system development."

---

### 2. Data Composition

**Data Types:**

```
Numerical (14 features):
- Continuous: age, income, loan_amount, interest_rate
- Discrete: employment_years, number_of_accounts
- Ratios: debt_to_income_ratio, credit_utilization

Categorical (8 features):
- Binary: employed (yes/no), homeowner (yes/no)
- Nominal: employment_type, education_level, state
- Ordinal: credit_score_range, risk_rating

Temporal (2 features):
- application_date
- funding_date

Target Variable:
- default (binary: yes/no)
```

**Data Distribution:**

```
Positive Class (Default): 8,500 (3.4%)
Negative Class (No Default): 241,500 (96.6%)
Class Imbalance Ratio: 1:28

Geographic Distribution:
- California: 25%
- Texas: 15%
- New York: 12%
- Other: 48%

Time Distribution:
- 2015: 12%
- 2016: 15%
- 2017: 18%
- 2018: 20%
- 2019: 18%
- 2020: 12%
- 2021: 4%
- 2022: 1%
```

**Feature Statistics:**

| Feature | Type | Min | Max | Mean | Std | Missing |
|---------|------|-----|-----|------|-----|---------|
| age | int | 21 | 85 | 42.3 | 12.1 | 0% |
| income | float | 15000 | 500000 | 65000 | 45000 | 0.2% |
| credit_score | int | 300 | 850 | 700 | 85 | 0% |
| loan_amount | float | 5000 | 500000 | 150000 | 120000 | 0% |

---

### 3. Data Collection

**Collection Process:**

```
Data Source: Loan Application System Database
├── Application Records (automated capture)
├── Credit Bureau Data (third-party)
├── Bank Systems (loan performance)
└── Manual Reviews (fraud checks)
```

**Collection Method:**

- **Automated**: Application system captures customer data
- **Third-party**: Credit scores obtained from bureaus
- **Administrative**: Loan payment history from servicing system

**Collection Period:** [Start date - End date]

Example: "January 1, 2015 - December 31, 2022"

**Collection Frequency:**
- Daily for new applications
- Monthly for performance updates
- Quarterly for fraud reviews

**Sampling Method:**
- All loan applications in period (no sampling)
- Filter: Only approved loans (excluded rejected applications)

**Data Sources:**

| Source | Data | Collection | Frequency |
|--------|------|-----------|-----------|
| Loan System | Application info | Automated | Real-time |
| Credit Bureau | Credit score | API | Real-time |
| Servicing System | Payment history | Batch | Monthly |
| Manual Reviews | Fraud flags | Manual | Quarterly |

---

### 4. Data Preprocessing

**Raw Data Issues:**

```
Missing Values:
- income: 0.2% (likely non-disclosure)
- employment_type: 0.1% (data entry gaps)

Outliers:
- income > $500,000: 0.05% (verified legitimate)
- interest_rate extremes: 0.01% (data errors)

Inconsistencies:
- Some applicants with age < 18: 12 records
- Negative loan amounts: 3 records
- Future application dates: 5 records
```

**Preprocessing Steps:**

```
Step 1: Data Cleaning
├── Remove invalid records (age < 18): 12 rows
├── Remove negative amounts: 3 rows
├── Remove future dates: 5 rows
└── Result: 250,000 → 249,980 rows

Step 2: Missing Value Handling
├── income: Impute with median by employment_type
├── employment_type: Impute with mode
└── Result: 0% missing values

Step 3: Standardization
├── Normalize income: StandardScaler
├── Normalize loan_amount: MinMaxScaler
├── Standardize categorical encoding
└── Result: Consistent format

Step 4: Feature Engineering
├── Create age_bucket: age ranges
├── Create income_bucket: income ranges
├── Create dti_ratio: debt_to_income calculation
└── Result: 250,000 → 250,000 with 22 features
```

**Quality Checks:**

```
Data Validation:
✓ Age range: 18-85
✓ Credit score: 300-850
✓ Interest rate: 2%-15%
✓ No nulls in key columns
✓ No negative values
✓ Date sequence valid
```

---

### 5. Data Characteristics

**Demographics:**

```
Age Distribution:
- 18-25: 8%
- 26-35: 25%
- 36-45: 28%
- 46-55: 22%
- 56-65: 12%
- 65+: 5%

Gender:
- Male: 60%
- Female: 40%

Geographic:
- Urban: 70%
- Suburban: 20%
- Rural: 10%
```

**Temporal Patterns:**

```
Application Volume:
- Peak: January (new year, refinancing)
- Lowest: August (summer slowdown)
- Trend: Increasing over period

Default Rate by Year:
- 2015: 2.8%
- 2016: 2.9%
- 2017: 3.1%
- 2018: 3.5% (recession impact)
- 2019: 3.4%
- 2020: 4.2% (COVID impact)
- 2021: 3.8%
- 2022: 3.2%
```

**Correlations:**

```
Strong Correlations (> 0.6):
- credit_score vs default: -0.75 (expected)
- income vs credit_score: 0.65 (expected)
- age vs default: -0.42 (weak negative)

Weak Correlations (< 0.3):
- education_level vs default: 0.12
- employment_type vs default: 0.18
```

---

### 6. Known Issues & Limitations

**Data Quality Issues:**

```
Issue 1: Class Imbalance
- Only 3.4% defaults
- Impact: Model may favor majority class
- Mitigation: Stratified sampling, class weights

Issue 2: Temporal Bias
- Economic conditions changed 2015-2022
- Impact: Model may not generalize to current conditions
- Mitigation: Recent data weighted more heavily

Issue 3: Selection Bias
- Only includes approved applications
- Impact: Model can't assess rejected applications
- Mitigation: Acknowledge limitation, test on other populations

Issue 4: Measurement Error
- Credit scores from multiple bureaus (different scales)
- Impact: Score consistency
- Mitigation: Normalized score ranges
```

**Limitations:**

- Dataset only includes bank's customers (not representative of full population)
- Limited to applicants with valid credit history
- Missing non-traditional credit factors
- Geographic concentration in major US metros
- Time period (2015-2022) may not represent current market

**Potential Biases:**

```
Historical Bias:
- Reflects lending practices of that era
- May encode discriminatory patterns from past

Representation Bias:
- Only includes approved loans (survivalship bias)
- May underrepresent high-risk applicants

Measurement Bias:
- Credit scores may be less predictive for certain groups
- Income verification varies by applicant type
```

---

### 7. Data Uses & Ethics

**Intended Uses:**

✅ Credit risk modeling  
✅ Loan underwriting  
✅ Portfolio risk assessment  
✅ Academic research (anonymized)  

**Prohibited Uses:**

❌ Individual credit ratings  
❌ Determine employment eligibility  
❌ Insurance pricing  
❌ Identify or contact individuals  

**Ethical Considerations:**

```
Privacy:
- Contains sensitive financial information
- Not to be shared without authorization
- De-identified version available for research

Fairness:
- Dataset reflects historical lending patterns
- May encode past discrimination
- Fairness testing required before use

Consent:
- Applicants consented to data use
- Consent covers modeling and analytics
- Additional uses require explicit consent
```

---

### 8. Accessibility & Access Control

**Data Access:**

```
Open Access:
- De-identified, aggregated dataset (research use)
- Located: /datasets/loan_performance_public/

Restricted Access:
- Full dataset with PII (internal use only)
- Located: Secure database (production)
- Requires: Data governance approval

Access Request Process:
1. Describe use case
2. Sign data use agreement
3. Complete compliance training
4. Get manager approval
5. Get data governance approval
6. Receive credentials
```

**Data Protection:**

- PII: Encrypted at rest, masked in logs
- Network: Restricted to internal networks
- Access: Logged and audited
- Retention: Retained per regulatory requirements

---

### 9. Maintenance & Updates

**Update Schedule:**

```
Weekly: New application data added
Monthly: Performance data updated
Quarterly: Quality review and validation
Annually: Full audit and documentation review
```

**Version History:**

| Version | Date | Changes | Notes |
|---------|------|---------|-------|
| 1.0 | 2024-01-01 | Initial release | All data through 2022 |

**Deprecation Plan:**

```
2025-Q1: Begin collecting new version (2023 data)
2025-Q2: New version released
2025-Q3: Old version marked as deprecated
2026-Q1: Old version archived
2026-Q2: Old version deleted (backup retained)
```

---

### 10. References & Contacts

**Data Owner:** [Name/Team]  
**Data Steward:** [Name]  
**Contact:** [Email]

**Related Documentation:**
- [Model Card](./model-card.md) - Models using this dataset
- [Experiment Log](./experiment-log.md) - Analysis using this data
- Data Dictionary: [Link]

**Laws & Regulations:**
- GDPR: Personal data of EU residents
- CCPA: Personal data of CA residents
- Fair Lending Laws: No discrimination in lending
- HIPAA: Not applicable (financial data)

---

## Using This Template

1. **Create** dataset card when collecting/curating data
2. **Complete** all sections with accurate information
3. **Get Reviewed** by data steward and compliance
4. **Publish** with dataset documentation
5. **Update** when data changes significantly
6. **Archive** when dataset is deprecated

---

## Best Practices

- Complete card before model development
- Be honest about limitations and biases
- Document preprocessing thoroughly
- Include data quality metrics
- Track all versions
- Restrict access appropriately
- Update regularly with changes
- Share with all stakeholders
