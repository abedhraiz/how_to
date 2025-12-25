# Risk Assessment

## Purpose

Identify, analyze, and plan mitigation strategies for risks that could impact project success.

## 1. Risk Management Framework

### Risk Definition
A risk is an uncertain event that, if it occurs, will have a positive or negative impact on project objectives.

### Risk Components
- **Event:** What could happen?
- **Probability:** How likely is it?
- **Impact:** What would happen if it occurs?
- **Mitigation:** How can we prevent or reduce it?

### Risk Response Strategies
- **Avoid:** Eliminate the risk
- **Mitigate:** Reduce probability or impact
- **Transfer:** Shift responsibility to third party
- **Accept:** Accept the risk and plan contingency

## 2. Risk Categories & Assessment

### Technical Risks

#### Data Quality Risk
- **Description:** Poor quality data leading to model failures
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Establish data quality standards early
  - Implement data validation checks
  - Create data quality monitoring dashboard
  - Build data cleaning procedures
- **Contingency:** Alternative data sources identified
- **Owner:** Data Engineer

#### Data Availability Risk
- **Description:** Required data unavailable or inaccessible
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Document all data dependencies
  - Establish data access early
  - Identify alternative data sources
  - Build redundancy into data pipelines
- **Contingency:** Use synthetic/historical data
- **Owner:** Data Owner

#### Model Performance Risk
- **Description:** Model doesn't meet performance requirements
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Start with realistic performance expectations
  - Use iterative development approach
  - Establish performance baselines early
  - Plan for alternative approaches
- **Contingency:** Hybrid model + rule-based approach
- **Owner:** ML Lead

#### Integration Risk
- **Description:** Difficulty integrating model with production systems
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Engage DevOps early
  - Design APIs early
  - Test integrations regularly
  - Document integration requirements
- **Contingency:** Gradual rollout plan
- **Owner:** DevOps Engineer

### Organizational Risks

#### Resource Availability Risk
- **Description:** Team members unavailable when needed
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Cross-train team members
  - Document critical processes
  - Plan for turnover
  - Allocate buffer time
- **Contingency:** Hire contractors/consultants
- **Owner:** Project Manager

#### Skill Gap Risk
- **Description:** Team lacks required expertise
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Assess skill gaps upfront
  - Plan training/hiring
  - Use external experts as needed
  - Build knowledge sharing
- **Contingency:** External consulting
- **Owner:** Project Manager

#### Stakeholder Misalignment Risk
- **Description:** Stakeholders disagree on requirements/approach
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Engage stakeholders early and often
  - Document decisions and rationale
  - Have regular alignment meetings
  - Clear communication
- **Contingency:** Escalation to executive sponsor
- **Owner:** Project Manager

### Operational Risks

#### Deployment Risk
- **Description:** Issues during deployment to production
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Develop detailed deployment plan
  - Stage rollout (canary, blue-green)
  - Test in production-like environment
  - Have rollback plan
- **Contingency:** Immediate rollback capability
- **Owner:** DevOps Engineer

#### Performance Degradation Risk
- **Description:** Model performance degrades in production
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Monitor model performance continuously
  - Set up data/model drift detection
  - Plan retraining strategy
  - Establish performance thresholds
- **Contingency:** Automated retraining
- **Owner:** ML Lead

#### Security Risk
- **Description:** Security vulnerability or breach
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Security review before deployment
  - Implement access controls
  - Encrypt sensitive data
  - Regular security audits
- **Contingency:** Incident response plan
- **Owner:** Security Officer

### Business Risks

#### Market Risk
- **Description:** Business needs change or market shifts
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Stay flexible in design
  - Regular business reviews
  - Monitor market conditions
  - Plan for pivots
- **Contingency:** Redefine scope/timeline
- **Owner:** Product Manager

#### ROI Risk
- **Description:** Project doesn't deliver expected ROI
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Validate assumptions early
  - Define clear ROI metrics
  - Regular ROI tracking
  - Adjust approach based on results
- **Contingency:** Project termination decision
- **Owner:** Executive Sponsor

#### Adoption Risk
- **Description:** Users don't adopt the model/system
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Involve users in design
  - Plan comprehensive training
  - Address trust concerns
  - Easy integration with workflows
- **Contingency:** Change management program
- **Owner:** Change Manager

### Compliance & Ethical Risks

#### Bias Risk
- **Description:** Model exhibits unfair bias against groups
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Conduct fairness assessment
  - Monitor for bias in production
  - Use fair algorithms/data
  - Regular bias audits
- **Contingency:** Model adjustment or retraining
- **Owner:** Ethics Officer

#### Privacy Risk
- **Description:** Privacy breach or regulatory violation
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Privacy impact assessment
  - Data minimization
  - Anonymization/encryption
  - Access controls
- **Contingency:** Incident response
- **Owner:** Privacy Officer

#### Compliance Risk
- **Description:** Non-compliance with regulations
- **Probability:** [High/Medium/Low]
- **Impact:** [High/Medium/Low]
- **Severity:** Probability × Impact = [High/Medium/Low]
- **Mitigation:**
  - Identify applicable regulations
  - Compliance review process
  - Documentation and audit trails
  - Regular compliance checks
- **Contingency:** Remediation plan
- **Owner:** Compliance Officer

## 3. Risk Register

### Risk ID Template
```
Risk-001
Title: [Risk title]
Category: [Category]
Description: [Detailed description]
Root Cause: [What causes this risk?]
Probability: [H/M/L]
Impact: [H/M/L]
Severity: [H/M/L]
Mitigation Strategy: [How to prevent]
Contingency Plan: [If risk occurs]
Owner: [Who is responsible]
Status: [Open/Monitoring/Closed]
Last Review: [Date]
```

### Risk Prioritization Matrix

```
         Low Impact    Medium Impact    High Impact
High P   [Medium]      [High]           [Critical]
Med P    [Low]         [Medium]         [High]
Low P    [Low]         [Low]            [Medium]
```

## 4. Risk Monitoring & Response

### Monitoring Schedule
- **Weekly:** Review high-priority risks
- **Bi-weekly:** Review medium-priority risks
- **Monthly:** Full risk review
- **Quarterly:** Risk assessment update

### Monitoring Indicators
For each risk, define what we monitor:
- Example (Data Quality): Quality metrics dashboard
- Example (Performance): Model performance tracking
- Example (Adoption): User adoption metrics

### Risk Escalation
- **When:** Risk probability or impact increases
- **Who:** Project Manager → Functional Manager → Sponsor
- **Action:** Review mitigation, adjust plan, increase resources

## 5. Risk Response Plan

### If Risk Occurs
1. **Detect:** How do we know the risk occurred?
2. **Notify:** Who must be informed?
3. **Assess:** What is the actual impact?
4. **Activate:** Which contingency plan?
5. **Execute:** Implement response
6. **Monitor:** Track resolution
7. **Learn:** What did we learn?

## Risk Response Decision Tree
```
Risk Triggered?
  ├─ Yes, Apply Mitigation
  │   ├─ Successful? Continue
  │   └─ Failed? Activate Contingency
  └─ No, Continue Monitoring
```

## 6. Template: Risk Register Entry

| Risk ID | Risk Title | Category | Description | P | I | Severity | Mitigation | Contingency | Owner | Status |
|---------|-----------|----------|-------------|---|---|----------|-----------|-------------|-------|--------|
| R-001 | | | | | | | | | | |

## Best Practices

1. ✅ Identify risks early
2. ✅ Involve entire team in risk assessment
3. ✅ Be specific about risks
4. ✅ Prioritize ruthlessly
5. ✅ Have concrete mitigation plans
6. ✅ Monitor actively
7. ✅ Learn from near-misses
8. ✅ Update register regularly

## Anti-Patterns

- ❌ Only executive risks (technical risks matter)
- ❌ Generic risks (be specific)
- ❌ No contingency plans
- ❌ Risk register never updated
- ❌ No one owns risks
- ❌ Ignoring low-probability/high-impact risks
- ❌ Not communicating risks

---

## Related Documents

- [Ethical Considerations](./ethical-considerations.md) - Ethical and bias risks
- [Data Strategy](./data-strategy.md) - Data-related risks
- [Compliance Checklist](../06-governance/compliance-checklist.md) - Regulatory requirements

---

*Active risk management reduces surprises and increases project success*
