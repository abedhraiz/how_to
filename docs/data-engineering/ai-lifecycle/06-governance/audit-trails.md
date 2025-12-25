# Audit Trails

## Purpose

Maintain complete, tamper-proof records of all model-related activities for compliance, debugging, and accountability.

## 1. What to Log

### Model-Related Events

```
Model Actions:
- Model registered
- Model version created
- Model deployed
- Model rolled back
- Model deprecated
- Model archived

Training Actions:
- Training started
- Training completed
- Hyperparameters changed
- Training data version selected
- Model evaluated
- Results recorded

Deployment Actions:
- Deployment initiated
- Canary deployment started
- Traffic increased
- Deployment completed
- Rollback initiated
- Rollback completed

Access Actions:
- Model accessed
- Data downloaded
- Prediction made (sampled)
- Configuration changed
- Permissions modified
```

### Audit Log Fields

```python
audit_log_entry = {
    'timestamp': '2024-01-15T14:30:45.123Z',  # ISO 8601
    'event_type': 'model_deployed',            # Event type
    'entity_type': 'model',                    # What was affected
    'entity_id': 'credit-approval-v1.0.0',     # Which entity
    'actor': 'john_doe',                       # Who did it
    'action': 'transition_stage',              # What happened
    'from_value': 'staging',                   # Previous state
    'to_value': 'production',                  # New state
    'context': {                               # Additional info
        'approval_required': True,
        'approved_by': 'jane_smith',
        'reason': 'Performance improvement',
    },
    'result': 'success',                       # success/failure
    'error_message': None,                     # If failed
    'ip_address': '192.168.1.100',            # Source IP
    'user_agent': 'curl/7.68.0',              # User agent
}
```

## 2. Logging Implementation

### Structured Logging

```python
import logging
import json
from datetime import datetime

class AuditLogger:
    def __init__(self, log_file='audit.log'):
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def log_event(self, event_type, entity_id, actor, action, 
                  from_value=None, to_value=None, context=None):
        """Log an audit event"""
        
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'event_type': event_type,
            'entity_id': entity_id,
            'actor': actor,
            'action': action,
            'from_value': from_value,
            'to_value': to_value,
            'context': context or {},
        }
        
        self.logger.info(json.dumps(entry))
    
    def log_model_deployment(self, model_id, actor, approval):
        """Log model deployment"""
        self.log_event(
            event_type='deployment',
            entity_id=model_id,
            actor=actor,
            action='deployed_to_production',
            context={'approval': approval}
        )

# Usage
audit_logger = AuditLogger()
audit_logger.log_model_deployment(
    model_id='credit-approval-v1.0.0',
    actor='john_doe',
    approval='jane_smith'
)
```

### Centralized Logging

```
Application Logs
     ↓
Centralized Log Aggregator
     (ELK, Splunk, CloudWatch, etc.)
     ↓
Storage → Analysis → Alerting → Reporting
```

## 3. Immutable Audit Log

### Blockchain-Based Audit

```
For high-compliance scenarios:

Event 1
  ↓ (Hash)
Block 1 → Event 2
  ↓ (Hash)
Block 2 → Event 3
  ↓ (Hash)
Block 3

Each block references previous
Cannot be modified without detection
```

### Write-Once Storage

```
Configuration:
- Append-only storage
- No deletion capability
- Encryption at rest
- Replication for redundancy
- Regular verification
```

## 4. Access & Query

### Audit Log Query

```sql
-- Find all model deployments by user
SELECT * FROM audit_log 
WHERE event_type = 'deployment' 
  AND actor = 'john_doe'
  AND timestamp >= '2024-01-01'
ORDER BY timestamp DESC;

-- Find all changes to production models
SELECT * FROM audit_log 
WHERE entity_type = 'model'
  AND to_value = 'production'
  AND timestamp >= '2024-01-01'
ORDER BY timestamp DESC;

-- Find all failed actions
SELECT * FROM audit_log 
WHERE result = 'failure'
  AND timestamp >= '2024-01-01'
ORDER BY timestamp DESC;
```

### Audit Dashboard

```
Audit Trail Dashboard

Recent Events (Last 24h)
┌────────────────────────────────┐
│ john_doe - deployed model      │ 2024-01-15 14:30
│ jane_smith - approved training │ 2024-01-15 10:15
│ ml_pipeline - completed train  │ 2024-01-14 22:45
│ system - detected drift        │ 2024-01-14 18:20
└────────────────────────────────┘

Event Distribution
┌────────────────────────────────┐
│ Deployments:   15              │
│ Approvals:     10              │
│ Training Runs: 42              │
│ Errors:        2               │
└────────────────────────────────┘

User Activity
┌────────────────────────────────┐
│ john_doe:     28 actions       │
│ jane_smith:   15 actions       │
│ ml_pipeline:  150 actions      │
│ system:       45 actions       │
└────────────────────────────────┘
```

## 5. Data Lineage

### Track Data Sources

```
Original Data Source (Database)
    ↓ (ETL Pipeline v1.2)
Data Lake - Raw
    ↓ (Cleaning Process v2.1)
Data Lake - Cleaned
    ↓ (Feature Engineering v3.0)
Feature Store
    ↓
Training Data (v2024-01-15)
    ↓
Model Training (Commit abc123)
    ↓
Model v1.0.0
```

### Audit Questions Answered

```
- What data was used to train this model?
- Where did that data come from?
- What transformations were applied?
- Who approved these changes?
- When was each version created?
- What code was used?
```

## 6. Compliance & Retention

### Retention Policy

```
Audit logs retention:
- Recent logs (< 1 year): Hot storage (fast access)
- Archive logs (1-7 years): Cold storage (cheap, slow)
- Legal hold: Indefinite retention if needed

Deletion:
- Automatic deletion after retention period
- Manual deletion only with approval
- Deletion logged as event
```

### Regulatory Compliance

```
GDPR:
- Log all data access
- Document data minimization
- Audit deletion requests
- Track consent changes

HIPAA:
- Log all PHI access
- Document access reasons
- Keep audit logs for 6 years
- Alert on suspicious access

SOX:
- Track system access
- Document changes
- Maintain segregation of duties
- Regular audits
```

## 7. Anomaly Detection

### Detect Suspicious Activity

```python
def detect_audit_anomalies(audit_logs):
    """Detect unusual patterns in audit logs"""
    
    anomalies = []
    
    # Detect off-hours access
    for log in audit_logs:
        time = datetime.fromisoformat(log['timestamp'])
        if time.hour < 6 or time.hour > 22:
            anomalies.append({
                'type': 'off_hours_access',
                'log': log,
                'severity': 'medium'
            })
    
    # Detect mass data access
    access_counts = {}
    for log in audit_logs:
        if log['action'] == 'data_accessed':
            actor = log['actor']
            access_counts[actor] = access_counts.get(actor, 0) + 1
    
    for actor, count in access_counts.items():
        if count > 1000:
            anomalies.append({
                'type': 'mass_data_access',
                'actor': actor,
                'count': count,
                'severity': 'high'
            })
    
    return anomalies
```

## 8. Audit Report Template

```markdown
# Audit Report - Q1 2024

## Executive Summary
[High-level overview]

## Event Summary
- Total Events: 2,450
- Successful: 2,432 (99.3%)
- Failed: 18 (0.7%)

## Activity by Type
- Deployments: 24
- Training Runs: 145
- Approvals: 32
- Data Access: 1,800
- Configuration Changes: 45
- Other: 404

## User Activity
- Active Users: 12
- Top User: john_doe (450 actions)
- New Users: 2

## Compliance
- GDPR: ✓ Compliant
- HIPAA: ✓ Compliant
- Audit Integrity: ✓ Verified

## Issues
- No critical issues detected
- 2 warnings about data access patterns
- 5 informational notes

## Recommendations
- Implement additional data access monitoring
- Review access patterns for unusual activity
- Continue regular audits
```

## Best Practices

1. ✅ Log all critical actions
2. ✅ Use structured logging
3. ✅ Immutable storage
4. ✅ Timestamped entries
5. ✅ Include context
6. ✅ Regular verification
7. ✅ Secure access
8. ✅ Regular audits

---

## Related Documents

- [Model Registry](./model-registry.md) - Version tracking
- [Compliance Checklist](./compliance-checklist.md) - Regulatory requirements
- [Documentation Standards](./documentation-standards.md) - Writing guidelines
