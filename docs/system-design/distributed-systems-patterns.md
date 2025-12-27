# Distributed Systems Patterns

Patterns for building resilient, loosely coupled microservices and distributed applications.

## 1. Resilience Patterns

### Circuit Breaker
Prevents cascading failures by stopping calls to a failing service.

**States:**
- `Closed`: Normal operation.
- `Open`: Fails immediately (after error threshold reached).
- `Half-Open`: Test if service recovered.

```python
# Conceptual Python Example
class CircuitBreaker:
    def call(self, func):
        if self.state == 'OPEN':
            raise Exception("Circuit Open")
        try:
            return func()
        except Exception:
            self.record_failure()
```

### Bulkhead
Isolates resources (thread pools, connections) so one failing component doesn't sink the whole ship.

### Retry with Exponential Backoff
Retrying transient failures with increasing delays to avoid thundering herd problems.

## 2. Data Consistency Patterns

### Saga Pattern
Managing distributed transactions across multiple services.
- **Choreography**: Services emit events to trigger next steps.
- **Orchestration**: Central coordinator tells services what to do.

### CQRS (Command Query Responsibility Segregation)
Separating read and write models to optimize performance and scalability independently.

## 3. Communication Patterns

### Sidecar Pattern
Offloading infrastructure concerns (logging, monitoring, proxying) to a separate container (e.g., Envoy, Istio).

### Backends for Frontends (BFF)
Creating separate backend services for different frontend interfaces (Mobile vs. Web) to optimize data delivery.

## 4. Event-Driven Patterns

### Event Sourcing
Storing state as a sequence of events rather than just the current state. Allows for time-travel debugging and audit trails.

### Outbox Pattern
Ensuring reliable message publishing by writing the message to a database table in the same transaction as the state change, then relaying it to the message broker.
