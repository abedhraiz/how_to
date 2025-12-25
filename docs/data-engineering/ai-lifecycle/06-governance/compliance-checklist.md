# Compliance Checklist

## Purpose

Ensure AI/ML systems meet regulatory, legal, and organizational compliance requirements.

## 1. Regulatory Compliance

### GDPR (General Data Protection Regulation)

**Scope:** EU residents, global companies processing EU data

```
Requirements:
☐ Lawful basis for data processing documented
☐ Data minimization - collect only necessary data
☐ Purpose limitation - use data only as stated
☐ Accuracy - keep personal data accurate/up-to-date
☐ Storage limitation - don't keep longer than needed
☐ Integrity & confidentiality - secure personal data
☐ Right to access - users can request their data
☐ Right to be forgotten - can request data deletion
☐ Right to rectification - can correct inaccurate data
☐ Right to object - can opt-out of processing
☐ Data Protection Impact Assessment (DPIA) completed
☐ Privacy policy published and accessible
☐ Data Processing Agreement (DPA) signed with vendors
☐ Data breach notification plan in place
☐ Consent management system implemented
```

### CCPA (California Consumer Privacy Act)

**Scope:** California residents

```
Requirements:
☐ Privacy notice published
☐ Right to know implemented
  - Users can request what data is collected
☐ Right to delete implemented
  - Users can request deletion of personal data
☐ Right to opt-out implemented
  - Users can opt-out of data sales
☐ Non-discrimination enforced
  - Don't penalize users for exercising rights
☐ Data minimization practiced
☐ Vendor contracts updated with CCPA clauses
☐ Consumer request process established
☐ Opt-out mechanism visible
```

### HIPAA (Health Insurance Portability and Accountability Act)

**Scope:** Healthcare organizations, health data

```
Requirements:
☐ Protected Health Information (PHI) identified
☐ Access controls implemented
☐ Audit controls enabled
☐ Integrity controls in place
☐ Transmission security configured
☐ Encryption of PHI at rest and in transit
☐ Minimum necessary principle applied
☐ Authorization & consent documented
☐ Business Associate Agreement (BAA) signed
☐ Breach notification procedure established
☐ Employee training completed
☐ Regular risk assessments performed
```

## 2. Algorithmic Fairness & Bias

```
Requirements:
☐ Fairness assessment completed
☐ Protected groups identified
☐ Disparate impact analysis done
☐ Bias mitigation techniques applied
☐ Fairness metrics tracked
☐ Regular fairness audits scheduled
☐ Biased outcome thresholds defined
☐ Remediation procedures documented
☐ Transparency requirements met
☐ Stakeholder notification plan ready
```

## 3. Data Quality & Documentation

```
Requirements:
☐ Data quality standards defined
☐ Data profiling completed
☐ Missing data assessment done
☐ Outlier analysis performed
☐ Data validation checks implemented
☐ Data dictionary maintained
☐ Data lineage documented
☐ Quality metrics tracked
☐ Regular audits scheduled
☐ Issues logged and tracked
```

## 4. Model Documentation

```
Requirements:
☐ Model card completed
☐ Technical documentation written
☐ Training data documented
☐ Features documented
☐ Limitations documented
☐ Known issues listed
☐ Performance metrics documented
☐ Ethical considerations addressed
☐ Version control maintained
☐ Code reviewed by second person
```

## 5. Explainability & Interpretability

```
Requirements:
☐ Model is interpretable (or explanation provided)
☐ SHAP values or feature importance calculated
☐ Model explanations documented
☐ User-facing explanations created
☐ Transparency report published
☐ Decision rationale documented
☐ Edge cases identified
☐ Failure modes documented
☐ Model behavior tested
☐ Explanations validated by domain experts
```

## 6. Security & Access Control

```
Requirements:
☐ Access controls implemented
☐ Authentication configured
☐ Authorization rules defined
☐ Encryption at rest enabled
☐ Encryption in transit enabled
☐ Secrets management configured
☐ API security implemented
☐ Network security configured
☐ Security testing completed
☐ Vulnerability assessment done
☐ Incident response plan established
```

## 7. Monitoring & Maintenance

```
Requirements:
☐ Performance monitoring configured
☐ Data drift monitoring enabled
☐ Model drift monitoring enabled
☐ Alerts configured
☐ Logging enabled
☐ Audit trails maintained
☐ Regular reviews scheduled
☐ Retraining plan documented
☐ Maintenance procedures established
☐ SLA defined and monitored
```

## 8. Third-Party & Vendor Management

```
Requirements:
☐ Data processing agreements signed
☐ Security requirements specified
☐ Compliance requirements specified
☐ Data ownership clarified
☐ Liability allocated
☐ Audit rights included
☐ Termination procedures defined
☐ Data return/deletion procedures
☐ Subprocessor agreements
☐ Regular vendor audits scheduled
```

## 9. Compliance Audit Checklist

### Pre-Deployment Review

```
Technical:
  ☐ Code review completed
  ☐ Security assessment passed
  ☐ Performance tests passed
  ☐ Load tests passed
  ☐ Integration tests passed

Operational:
  ☐ Runbooks documented
  ☐ Monitoring configured
  ☐ Alerts configured
  ☐ Incident response plan ready
  ☐ Rollback procedure tested

Compliance:
  ☐ Regulatory requirements met
  ☐ Privacy requirements met
  ☐ Security requirements met
  ☐ Documentation complete
  ☐ Stakeholders notified
```

### Post-Deployment Review

```
Performance:
  ☐ Metrics meet targets
  ☐ No unexpected errors
  ☐ Latency acceptable
  ☐ User feedback positive

Compliance:
  ☐ Fair treatment across groups
  ☐ No privacy violations
  ☐ No security incidents
  ☐ Monitoring working
  ☐ Logs complete

Business:
  ☐ ROI targets being met
  ☐ User adoption acceptable
  ☐ No customer complaints
  ☐ Cost within budget
```

## 10. Compliance Reporting

### Quarterly Report Template

```
Compliance Report - Q1 2024

Regulatory Status:
- GDPR: ✓ Compliant
- CCPA: ✓ Compliant
- Company Policy: ✓ Compliant

Security:
- Incidents: 0
- Vulnerabilities found & fixed: 2
- Security audits completed: 1

Fairness & Ethics:
- Bias audits completed: 3
- Protected group analysis: Complete
- Fairness metrics: All within threshold

Data Quality:
- Average completeness: 99.2%
- Validation failures: <0.1%
- Data breaches: 0

Performance:
- Availability: 99.95%
- Error rate: 0.2%
- User complaints: 2 (resolved)

Issues & Resolutions:
1. Data quality issue in region X - FIXED
2. Fairness concern in age group - MONITORING

Recommendations:
- Implement additional fairness checks
- Increase monitoring frequency
- Schedule annual third-party audit
```

## Best Practices

1. ✅ Know your regulations
2. ✅ Document everything
3. ✅ Regular audits
4. ✅ Stay current with regulations
5. ✅ Engage compliance team early
6. ✅ Implement controls proactively
7. ✅ Monitor continuously
8. ✅ Train your team

---

## Related Documents

- [Documentation Standards](./documentation-standards.md) - Writing guidelines
- [Ethical Considerations](../01-planning/ethical-considerations.md) - Ethics framework
- [Model Registry](./model-registry.md) - Version control
