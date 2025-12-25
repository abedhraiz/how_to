# Model Card Template

## Purpose

Provide a structured, comprehensive overview of a machine learning model for transparency, reproducibility, and governance.

Based on [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)

---

## Model Card

### 1. Model Details

**Model Name:** [e.g., "Credit Risk Assessment Model v2.1.0"]

**Model Type:** [e.g., "Binary Classification / Regression / NLP"]

**Version:** [e.g., "2.1.0"]

**Released Date:** [YYYY-MM-DD]

**Last Updated:** [YYYY-MM-DD]

**Owned By:** [Team/Person Name]

**Contact:** [Email]

**Model Description:**
[2-3 sentence high-level overview of what the model does]

Example:
"This model predicts the probability of credit default for loan applicants. It uses historical loan data and applicant characteristics to assess credit risk. The model is used in our underwriting process to support lending decisions."

---

### 2. Model Use

**Primary Use Case:** [What should this model be used for?]

Example: "Assess credit risk in loan underwriting process"

**Secondary Use Cases:** [Other valid applications]

Example: "Portfolio risk assessment, pricing optimization"

**Out-of-Scope Use Cases:** [What should NOT use this model?]

Example: 
- Individual credit ratings
- Regulatory capital requirements
- Real-time trading decisions

**Users:** [Who uses this model?]

Example:
- Loan officers
- Risk analysts
- Automated underwriting system

---

### 3. Model Architecture & Training

**Algorithm:** [e.g., "Gradient Boosting (XGBoost)"]

**Framework & Version:** [e.g., "XGBoost 1.7.0"]

**Input Features:** [List key features]

```
Numerical Features (8):
- age (years)
- income (annual salary)
- credit_score (0-850)
- loan_amount (dollars)
- debt_to_income_ratio (%)
- employment_years (years)
- previous_defaults (count)
- credit_utilization (%)

Categorical Features (4):
- employment_type (8 categories)
- education_level (5 categories)
- marital_status (4 categories)
- state (50 categories)
```

**Output:** [What does the model predict?]

Example: "Probability of credit default within 12 months (0-1)"

**Model Size:** [e.g., "50 MB"]

**Inference Time:** [e.g., "10ms per prediction"]

**Training Data:**

| Aspect | Details |
|--------|---------|
| Data Source | Loan performance database |
| Time Period | 2015-2022 |
| Total Samples | 250,000 loans |
| Training Set | 200,000 (80%) |
| Validation Set | 25,000 (10%) |
| Test Set | 25,000 (10%) |

**Training Process:**

```
Data Preparation
  ↓
  - Missing value imputation
  - Outlier handling
  - Feature scaling
  - Categorical encoding
  ↓
Feature Engineering
  ↓
  - Polynomial features
  - Interaction terms
  - Feature selection (24 final features)
  ↓
Model Training
  ↓
  - Cross-validation (5-fold)
  - Hyperparameter tuning (Bayesian optimization)
  - Training time: 2.5 hours on GPU
  ↓
Model Evaluation
  ↓
  - Tested on holdout test set
  - Evaluated for fairness
  - Performance validated
```

**Hyperparameters:**

```yaml
algorithm: xgboost
learning_rate: 0.05
n_estimators: 500
max_depth: 6
subsample: 0.8
colsample_bytree: 0.8
reg_alpha: 0.1
reg_lambda: 1.0
random_state: 42
```

---

### 4. Model Performance

**Overall Metrics (Test Set):**

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| AUC-ROC | 0.89 | Very good discrimination |
| Accuracy | 0.85 | Correctly classifies 85% |
| Precision | 0.82 | 82% of defaults predicted are correct |
| Recall | 0.76 | Catches 76% of actual defaults |
| F1 Score | 0.79 | Good balance of precision/recall |

**Performance by Subgroup:**

| Group | Size | Accuracy | AUC | Notes |
|-------|------|----------|-----|-------|
| Age < 30 | 15% | 0.83 | 0.87 | Slightly lower accuracy |
| Age 30-50 | 60% | 0.86 | 0.90 | Peak performance |
| Age > 50 | 25% | 0.84 | 0.88 | Similar to overall |
| Income < $40k | 20% | 0.81 | 0.85 | Lower performance |
| Income $40k-$100k | 50% | 0.86 | 0.90 | Good performance |
| Income > $100k | 30% | 0.87 | 0.91 | Best performance |

**Comparison to Baseline:**

| Model | AUC | Accuracy | F1 |
|-------|-----|----------|-----|
| Baseline (Logistic Regression) | 0.72 | 0.78 | 0.68 |
| Our Model (XGBoost) | 0.89 | 0.85 | 0.79 |
| Improvement | +24% | +9% | +16% |

**Performance Graphs:**

```
ROC Curve
1.0 ┌────────────────
    │       ╱╱
    │    ╱╱
0.5 ├  ╱╱────────
    │╱╱
    └────────────────── 1.0
    0.0  False Positive Rate

Precision-Recall Curve
1.0 ┌────────────────
    │╲
    │ ╲╲
0.5 ├──╲╲────────
    │    ╲╲
    └────────────────── 1.0
    0.0  Recall

Feature Importance
credit_score      ████████░░
income           ██████░░░░
employment_type  █████░░░░░
age              ████░░░░░░
```

---

### 5. Limitations & Bias

**Known Limitations:**

1. **Training Data Bias**
   - Training data from 2015-2022 may not represent current economic conditions
   - Historical bias against certain groups may be encoded

2. **Feature Coverage**
   - Model only uses basic demographics and financial variables
   - Does not capture non-traditional credit factors (e.g., employment stability)
   - Missing behavioral and psychological factors

3. **Distribution Shift**
   - Performance degrades when used on populations different from training data
   - Economic downturns not well-represented in training data

4. **Edge Cases**
   - Performance undefined for applicants with very high/low credit scores
   - Limited experience with very large loan amounts

**Fairness Analysis:**

```
Fairness Metric Comparison (Target: AUC-ROC difference < 5%)

Protected Group: Race
- White applicants: AUC 0.89
- Black applicants: AUC 0.87 (difference: 2%)
- Hispanic applicants: AUC 0.88 (difference: 1%)
- Asian applicants: AUC 0.90 (difference: 1%)
Status: ✓ PASS (all differences < 5%)

Protected Group: Gender
- Male applicants: AUC 0.89
- Female applicants: AUC 0.88 (difference: 1%)
Status: ✓ PASS

Protected Group: Age
- Age < 30: AUC 0.87 (difference: 2%)
- Age 30-50: AUC 0.90 (difference: 1%)
- Age > 50: AUC 0.88 (difference: 1%)
Status: ✓ PASS

Protected Group: Income
- Low income: AUC 0.85 (difference: 4%)
- Medium income: AUC 0.90 (difference: 1%)
- High income: AUC 0.91 (difference: 2%)
Status: ✓ PASS
```

**Bias Mitigation:**

- Stratified sampling to ensure representation
- Fairness constraints during training
- Regular fairness audits
- Decision review process for edge cases

**Known Failure Cases:**

- Predicts default when credit score improves unexpectedly
- Overestimates risk for career changers
- Underestimates risk during economic downturns

---

### 6. Ethical Considerations

**Ethical Impact Assessment:**

| Dimension | Analysis |
|-----------|----------|
| Discrimination | Regularly audited for fairness |
| Transparency | Offers explanations for decisions |
| Accountability | Clear escalation for appeals |
| Privacy | Minimal personal data retained |
| Human Oversight | All high-impact decisions reviewed |

**Responsible Use Guidelines:**

✅ DO:
- Use as support tool for human decision makers
- Review model explanations
- Monitor performance over time
- Appeal process for decisions
- Regular fairness audits

❌ DON'T:
- Use as sole decision maker
- Ignore fairness metrics
- Deploy without monitoring
- Skip reviews for efficiency
- Make high-stakes decisions without human oversight

---

### 7. Maintenance & Monitoring

**Monitoring Plan:**

```
Performance Monitoring:
- Weekly: Check for data drift
- Monthly: Calculate performance metrics
- Quarterly: Fairness audit
- Annually: Full model review

Metrics Tracked:
- Model accuracy by subgroup
- Prediction volume and distribution
- Feature drift detection
- Fairness metrics
- User feedback / appeals
```

**Retraining Schedule:**

- Quarterly retraining with latest data
- Trigger retraining if accuracy drops > 2%
- A/B test new models before deployment
- Keep previous version as fallback

**Version History:**

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| 2.1.0 | 2024-01-15 | Improved fairness | Production |
| 2.0.0 | 2023-10-01 | New features | Archived |
| 1.5.0 | 2023-07-15 | Bug fixes | Deprecated |

---

### 8. References & Links

**Related Documents:**
- [Dataset Card](./dataset-card.md) - Training data documentation
- [Experiment Log](./experiment-log.md) - Model development log
- [Model Registry](../06-governance/model-registry.md) - Version control

**Code Repository:**
- Repository: [GitHub link]
- Branch: [Branch name]
- Model artifact: [Storage location]

**Papers & References:**
- Model Cards for Model Reporting ([https://arxiv.org/abs/1810.03993](https://arxiv.org/abs/1810.03993))
- Algorithm paper: [Citation]

---

## Using This Template

1. **Create** a new model card when model is ready for review
2. **Complete** all sections with accurate information
3. **Get Reviewed** by data scientist and domain expert
4. **Get Approved** by governance/compliance team
5. **Publish** in model registry
6. **Update** as model is maintained and improved
7. **Archive** when model is deprecated

---

## Best Practices

- Keep model cards up-to-date
- Be honest about limitations
- Document fairness thoroughly
- Include contact information
- Link related documents
- Version your model cards
- Share with stakeholders
