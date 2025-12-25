# Success Criteria

## Purpose

Define measurable, objective criteria that will be used to determine if the project has achieved its goals.

## Success Criteria Framework

### 1. Business Metrics

#### Revenue Impact
- **Metric:** [Metric name]
- **Target:** [Quantified target]
- **Baseline:** [Current state]
- **Timeline:** [When to measure]
- **Owner:** [Who tracks this]

#### Cost Reduction
- **Metric:** [Metric name]
- **Target:** [Quantified target]
- **Baseline:** [Current state]
- **Timeline:** [When to measure]
- **Owner:** [Who tracks this]

#### Efficiency Gains
- **Metric:** [Metric name]
- **Target:** [Quantified target]
- **Baseline:** [Current state]
- **Timeline:** [When to measure]
- **Owner:** [Who tracks this]

#### Customer Satisfaction
- **Metric:** [Metric name]
- **Target:** [Quantified target]
- **Baseline:** [Current state]
- **Timeline:** [When to measure]
- **Owner:** [Who tracks this]

### 2. Technical Metrics

#### Model Performance

| Metric | Target | Baseline | Acceptance Threshold |
|---|---|---|---|
| Accuracy | | | |
| Precision | | | |
| Recall | | | |
| F1-Score | | | |
| AUC-ROC | | | |
| RMSE (for regression) | | | |

#### Infrastructure Performance
- **Latency:** [Target response time in ms]
- **Throughput:** [Requests per second]
- **Availability:** [Uptime %]
- **Scalability:** [Max concurrent users/requests]

#### Data Quality
- **Completeness:** % of non-null values
- **Accuracy:** % of correct values
- **Consistency:** % of consistent records
- **Timeliness:** Data freshness requirements

### 3. Deployment Criteria

- [ ] Model passes all acceptance tests
- [ ] Performance metrics meet thresholds
- [ ] Infrastructure meets availability requirements
- [ ] Monitoring and alerting are configured
- [ ] Rollback plan is documented and tested
- [ ] Documentation is complete
- [ ] Compliance checks passed
- [ ] Stakeholder sign-off obtained

### 4. Adoption Metrics

- **User Adoption:** % of eligible users using the model
- **Feature Adoption:** % of features being used
- **User Satisfaction:** NPS or satisfaction score
- **Support Load:** Number of support tickets

### 5. Operational Metrics

- **Model Drift:** Monitoring data/model drift detection
- **Retraining Frequency:** How often model needs retraining
- **Incident Rate:** Production issues per month
- **Mean Time to Recovery:** Time to resolve issues

## Measurement Plan

### Data Collection
- **Where:** [Source systems, logs, monitoring tools]
- **Frequency:** [Daily, weekly, monthly]
- **Owner:** [Who is responsible]
- **Tools:** [Monitoring tools, dashboards, reporting]

### Reporting
- **Frequency:** [When metrics are reported]
- **Audience:** [Who receives reports]
- **Format:** [Dashboard, email, presentation]
- **Escalation:** [When to escalate issues]

### Targets Over Time

| Phase | Accuracy | Latency | Availability | Volume |
|---|---|---|---|---|
| Pilot | 85% | <500ms | 99% | <1K req/day |
| Early Prod | 88% | <300ms | 99.5% | <10K req/day |
| Full Scale | 90% | <200ms | 99.9% | >100K req/day |

## Success Thresholds

### Go/No-Go Criteria
- **Minimum acceptable performance:** [Threshold]
- **Target performance:** [Goal]
- **Excellent performance:** [Stretch goal]

### Decision Framework

If model achieves:
- ✅ **95%+ target metrics:** Full production deployment
- ✅ **80-95% target metrics:** Conditional deployment with monitoring
- ⚠️ **60-80% target metrics:** Pilot program or additional development
- ❌ **<60% target metrics:** Project reassessment required

## Post-Launch Success Measurement

### Week 1-4
- [ ] System stability and availability
- [ ] Data quality monitoring
- [ ] User feedback collection

### Month 1-3
- [ ] Business metric baseline establishment
- [ ] Model performance validation
- [ ] User adoption tracking

### Month 3-12
- [ ] Business impact realization
- [ ] ROI measurement
- [ ] Continuous improvement identification

## Success Celebration & Lessons Learned

When success criteria are met:
1. Document achievements
2. Share results with stakeholders
3. Conduct lessons learned session
4. Plan for scale/optimization
5. Archive for future reference

## Definition of Done

A feature is "done" when:
- ✅ Code is peer-reviewed
- ✅ Tests pass (unit, integration, system)
- ✅ Performance meets specifications
- ✅ Documentation is complete
- ✅ Monitoring is configured
- ✅ Stakeholder acceptance obtained

## Best Practices

1. ✅ Define metrics at project start
2. ✅ Make metrics SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
3. ✅ Establish baselines
4. ✅ Monitor metrics regularly
5. ✅ Be flexible - update criteria as understanding improves
6. ✅ Include both leading and lagging indicators
7. ✅ Align metrics with business goals

## Anti-Patterns to Avoid

- ❌ Too many metrics (focus on 5-10 key metrics)
- ❌ Vague or unmeasurable criteria
- ❌ Criteria that are too easy or too hard
- ❌ Not tracking metrics after launch
- ❌ Ignoring business metrics in favor of technical ones
- ❌ Moving goalposts mid-project

---

## Related Documents

- [Project Charter](./project-charter.md) - Project scope and objectives
- [Stakeholders](./stakeholders.md) - Who cares about success
- [Performance Metrics](../05-monitoring/performance-metrics.md) - Ongoing metric tracking
- [Model Evaluation](../03-modeling/model-evaluation.md) - Model assessment

---

*Success criteria should be defined early, communicated clearly, and tracked consistently*
