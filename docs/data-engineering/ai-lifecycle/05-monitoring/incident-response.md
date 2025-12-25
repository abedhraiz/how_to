# Incident Response

## Purpose

Quickly detect, respond to, and resolve production issues to minimize user impact.

## 1. Incident Severity Levels

### Severity Classification

```
SEVERITY 1 (CRITICAL)
- Model completely unavailable
- Service down or 100% failure rate
- Customer-facing impact
- Response time: Immediate (page on-call)
- Resolution time: < 30 minutes
- Example: API returning 500 errors

SEVERITY 2 (HIGH)
- Significant performance degradation
- 25-50% failure rate or accuracy drop > 10%
- Some customer impact
- Response time: < 30 minutes
- Resolution time: < 2 hours
- Example: Accuracy drops from 90% to 78%

SEVERITY 3 (MEDIUM)
- Minor issues affecting some users
- < 25% failure rate
- Workaround exists
- Response time: < 2 hours
- Resolution time: < 1 day
- Example: Slow response times

SEVERITY 4 (LOW)
- No user impact
- Feature request or non-critical bug
- Response time: < 1 day
- Resolution time: < 1 week
- Example: Logging format inconsistency
```

## 2. Incident Response Process

### Step 1: Detect

```
Alert triggered → Verify → Classify
         ↓
   - Automatic alert
   - Human report
   - Failed health check
```

### Step 2: Alert & Notify

```python
def notify_incident(incident):
    """Notify appropriate people based on severity"""
    
    if incident['severity'] == 'CRITICAL':
        # Page on-call engineer immediately
        page_on_call(incident)
        notify_slack('#critical-incidents', incident)
        notify_email('ml-team@company.com', incident)
    
    elif incident['severity'] == 'HIGH':
        # Notify team
        notify_slack('#ml-incidents', incident)
        notify_email('ml-lead@company.com', incident)
    
    else:
        # Create ticket
        create_jira_ticket(incident)
```

### Step 3: Investigate

```
Take ownership
    ↓
Gather logs → Analyze → Determine root cause
    ↓
- Check metrics dashboards
- Review recent deployments
- Check data quality
- Review error logs
- Run diagnostic queries
```

### Step 4: Mitigate

```
Temporary fix
    ↓
- Switch to fallback model
- Rollback to previous version
- Reroute traffic
- Increase resources
```

### Step 5: Resolve

```
Permanent fix
    ↓
- Deploy new version
- Retrain model
- Fix data pipeline
- Update configuration
```

### Step 6: Communicate

```
Keep stakeholders informed
    ↓
- Status updates every 15 minutes
- Post-incident notification
- Root cause explanation
```

### Step 7: Post-Mortem

```
Conduct review
    ↓
- What happened?
- Why did it happen?
- What could prevent it?
- What will we do differently?
```

## 3. Runbook Examples

### Model Unavailable

```
ALERT: Model Service Unavailable

SEVERITY: CRITICAL
RESPONSE: < 5 minutes

Steps:
1. Verify alert is real
   - Check /health endpoint
   - Check metrics dashboard
   
2. Check logs for errors
   - Pod crash logs
   - Application errors
   - Database connection errors
   
3. Quick fixes to try:
   - Restart service
   - Check memory/CPU
   - Verify database connection
   
4. If not fixed:
   - Rollback to last known good version
   - Page on-call engineer
   
5. Escalation:
   - If still broken after 10 min → page manager
   - If still broken after 20 min → page director
```

### High Error Rate

```
ALERT: Model Error Rate > 5%

SEVERITY: HIGH
RESPONSE: < 30 minutes

Steps:
1. Identify error type
   - Check error logs
   - Sample failing requests
   - Check for patterns
   
2. Determine scope
   - All users or specific segment?
   - All features or specific type?
   - Recent deployments?
   
3. Investigation
   - Check recent model deployment
   - Check recent data changes
   - Check input validation
   - Check resource utilization
   
4. Response options:
   a) Data issue?
      - Restart pipeline
      - Rollback data version
   
   b) Model issue?
      - Switch to fallback model
      - Rollback model version
   
   c) Infrastructure issue?
      - Increase replicas
      - Failover to other region
```

### Accuracy Degradation

```
ALERT: Accuracy < 85%

SEVERITY: HIGH
RESPONSE: < 1 hour

Steps:
1. Confirm issue
   - Run validation on test set
   - Check if data drift detected
   - Compare to baseline accuracy
   
2. Investigate
   - Analyze recent data changes
   - Check for input distribution changes
   - Review recent deployments
   
3. Root cause analysis
   - Data quality issues?
   - Concept drift?
   - Feature engineering change?
   - Bug in preprocessing?
   
4. Actions
   - Short-term: Rollback model
   - Medium-term: Investigate root cause
   - Long-term: Retrain with new data
```

## 4. Incident Tracking

### Incident Record

```python
incident = {
    'id': 'INC-2024-001',
    'timestamp': '2024-01-15T14:30:00Z',
    'severity': 'HIGH',
    'title': 'Model Accuracy Degradation',
    'description': 'Accuracy dropped from 92% to 78%',
    'status': 'investigating',
    'assigned_to': 'john_doe',
    'timeline': [
        {'time': '14:30', 'event': 'Alert triggered'},
        {'time': '14:33', 'event': 'On-call engineer paged'},
        {'time': '14:35', 'event': 'Investigation started'},
        {'time': '14:45', 'event': 'Root cause identified: data drift'},
        {'time': '15:15', 'event': 'Rolled back to v1.1.0'},
        {'time': '15:20', 'event': 'Accuracy restored to 91%'},
    ],
    'root_cause': 'Data distribution shift due to economic changes',
    'resolution': 'Rolled back model, scheduled retraining',
    'resolution_time_minutes': 50,
}
```

## 5. Communication Templates

### Incident Notification

```
🚨 INCIDENT: Model Service Degradation

Severity: HIGH
Time: 2024-01-15 14:30 UTC
Status: INVESTIGATING

Impact: 
- Accuracy dropped to 75% (target: 90%)
- Affecting all users
- Estimated resolution: 1 hour

Action:
- Team investigating root cause
- Preparing rollback if needed
- Will update in 15 minutes

Questions? Contact #ml-incidents on Slack
```

### Resolution Notification

```
✅ INCIDENT RESOLVED: Model Service Degradation

Severity: HIGH
Duration: 50 minutes
Start: 2024-01-15 14:30 UTC
End: 2024-01-15 15:20 UTC

Root Cause:
Data distribution shift due to seasonal changes
Not detected by drift monitoring

Resolution:
Rolled back to previous model version (v1.1.0)

Status:
- Accuracy restored to 91%
- All systems normal
- Scheduled post-mortem for tomorrow

Thank you for your patience!
```

## 6. Incident Prevention

### Monitoring & Alerts

```
Pre-incident detection:
- Drift monitoring alerts
- Performance degradation warnings
- Error rate anomalies
- Resource utilization alerts
- Data quality issues
```

### Testing & Validation

```
- Unit tests for model code
- Integration tests for pipelines
- Smoke tests before deployment
- A/B testing for new models
- Load testing for infrastructure
```

## Best Practices

1. ✅ Automate detection
2. ✅ Clear severity definitions
3. ✅ Quick response process
4. ✅ Documented runbooks
5. ✅ Regular drill exercises
6. ✅ Post-incident reviews
7. ✅ Preventive monitoring
8. ✅ Rapid rollback capability

---

## Related Documents

- [Performance Metrics](./performance-metrics.md) - Metric monitoring
- [Drift Detection](./drift-detection.md) - Issue detection
- [Deployment Strategy](../04-deployment/deployment-strategy.md) - Rollback procedures
