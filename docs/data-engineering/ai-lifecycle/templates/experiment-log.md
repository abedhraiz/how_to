# Experiment Log Template

## Purpose

Track ML experiments in a structured format to enable reproducibility, comparison, and learning.

---

## Experiment Log Entry

### Basic Information

**Project:** [Project name]  
**Date Started:** [YYYY-MM-DD]  
**Date Completed:** [YYYY-MM-DD]  
**Experimenter:** [Name]  
**Experiment ID:** [auto-generated or manual ID]  
**Git Commit:** [commit hash]  

### Objective

**Goal:** [What are you trying to achieve?]

**Hypothesis:** [What do you expect to happen?]

**Expected Impact:** [How will this improve things?]

### Data

**Training Data:**
- Source: [Where does data come from?]
- Version: [Data version/date]
- Samples: [Number of training samples]
- Features: [Number of features]
- Target variable: [What are we predicting?]

**Validation Data:**
- Samples: [Number]
- Source: [Training set split / separate dataset]

**Test Data:**
- Samples: [Number]
- Source: [Holdout set]

**Data Preprocessing:**
- Missing value handling: [Method used]
- Outlier handling: [Method used]
- Scaling/normalization: [Method used]
- Feature selection: [Method used]

### Model Configuration

**Algorithm:** [Model type]

**Hyperparameters:**
```
learning_rate: 0.01
n_estimators: 100
max_depth: 10
batch_size: 32
```

**Architecture (if neural network):**
```
Input Layer: 50 features
Hidden Layer 1: 128 neurons, ReLU
Hidden Layer 2: 64 neurons, ReLU
Hidden Layer 3: 32 neurons, ReLU
Output Layer: 10 neurons, Softmax
Dropout: 0.2
```

### Training

**Training Duration:** [Hours/minutes]

**Training Stopped:** [When and why?]
- Reason: [Convergence / max epochs / manual stop]
- Epoch: [Which epoch stopped at]

**Key Observations:**
- Training loss trajectory: [Decreasing / fluctuating / etc]
- Validation loss trajectory: [Plateaued after epoch X]
- Signs of overfitting: [Yes / No - details]
- Training issues: [Any problems encountered]

### Results

**Performance Metrics:**

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| Accuracy | 0.95 | 0.92 | 0.91 |
| Precision | 0.94 | 0.91 | 0.90 |
| Recall | 0.93 | 0.89 | 0.88 |
| F1 Score | 0.935 | 0.90 | 0.89 |
| AUC-ROC | 0.98 | 0.96 | 0.95 |

**Comparison to Baseline:**
- Baseline accuracy: 0.85
- New model accuracy: 0.91
- Improvement: +6% (absolute), +7.1% (relative)

**Confusion Matrix (Test Set):**
```
                Predicted
                Negative  Positive
Actual Negative     850       50
       Positive      60      40
```

**Feature Importance:**
```
Top 10 Most Important Features:
1. feature_a: 0.25
2. feature_b: 0.18
3. feature_c: 0.15
4. feature_d: 0.10
5. feature_e: 0.08
... (5 more)
```

### Error Analysis

**Failure Cases:**
- [Example 1]: Expected X, got Y, because...
- [Example 2]: Expected X, got Y, because...

**Patterns in Errors:**
- Errors more common in: [Specific data subset]
- Related to: [Which features]

### Fairness & Bias

**Protected Groups Analyzed:**
- [Group 1]: Performance metrics
- [Group 2]: Performance metrics

**Disparate Impact:**
- [Group 1] accuracy: 0.91
- [Group 2] accuracy: 0.88
- Difference: 0.03 (within acceptable threshold of 0.05)

**Bias Mitigation Applied:**
- Technique: [Which technique]
- Effectiveness: [How well it worked]

### Insights & Learnings

**What Worked Well:**
1. [Insight 1]
2. [Insight 2]
3. [Insight 3]

**What Didn't Work:**
1. [What we tried that failed]
2. [Why it didn't work]
3. [What we learned]

**Key Takeaways:**
- [Main learning 1]
- [Main learning 2]
- [Main learning 3]

### Next Steps

**Recommended Actions:**
- [ ] Use this model for [purpose]
- [ ] Run experiment with [modification]
- [ ] Investigate [specific issue]
- [ ] Test on [different data]

**Alternative Approaches to Try:**
1. [Different algorithm]
2. [Different hyperparameters]
3. [Different features]

### Reproducibility

**Code Repository:**
- Repo: [GitHub/GitLab link]
- Branch: [Branch name]
- Commit: [Commit hash]

**Dependencies:**
```
scikit-learn==1.0.2
tensorflow==2.10.0
numpy==1.23.0
pandas==1.5.0
```

**How to Reproduce:**
```bash
# Step 1: Setup
python setup.py

# Step 2: Prepare data
python prepare_data.py --version 2024-01-15

# Step 3: Train model
python train.py \
  --model gradient_boosting \
  --learning_rate 0.01 \
  --n_estimators 100

# Step 4: Evaluate
python evaluate.py --model_path models/latest/
```

**Results Reproduced:** [Yes / No / Partially]

### Attachments

- [ ] Training curves plot
- [ ] Confusion matrix visualization
- [ ] Feature importance plot
- [ ] Error analysis report
- [ ] Model artifact (saved model file)
- [ ] Detailed metrics file

### Approval & Sign-off

**Reviewed by:** [Reviewer name]  
**Review Date:** [YYYY-MM-DD]  
**Approved:** [Yes / No / Conditional]  
**Comments:** [Any reviewer feedback]

---

## Tips for Using This Template

1. **Fill out as you experiment** - Don't wait until the end
2. **Be specific** - Use numbers, not vague descriptions
3. **Document failures** - Failed experiments are valuable
4. **Include context** - Explain the why, not just the what
5. **Attach artifacts** - Include plots, data, models
6. **Be honest** - Document limitations and issues
7. **Reference related work** - Link to other experiments

---

## Related Documents

- [Model Card](./model-card.md) - Final model documentation
- [Dataset Card](./dataset-card.md) - Data documentation
- [Model Evaluation](../03-modeling/model-evaluation.md) - Evaluation methods
