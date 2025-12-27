# System Design & Architecture

Mastering the principles of building scalable, reliable, and maintainable distributed systems.

## Purpose

This section bridges the gap between code and architecture. It covers the patterns, trade-offs, and decision-making frameworks required to design systems that can handle growth and failure.

## Guides

- **[Distributed Systems Patterns](./distributed-systems-patterns.md)** - Circuit Breakers, Sagas, CQRS, and more.
- **[Scalability & Performance](./scalability-guide.md)** - Caching, Load Balancing, Sharding, and CAP Theorem.
- **[Diagramming & Documentation](./diagramming-guide.md)** - Communicating architecture with C4 and Mermaid.js.

## Key Concepts

### Scalability vs. Performance
- **Performance**: "How fast is it?" (Latency, Throughput)
- **Scalability**: "How much can it grow?" (Vertical vs. Horizontal scaling)

### Reliability
- **Availability**: System uptime (99.9% vs 99.99%).
- **Resiliency**: Recovering from failures (Retries, Backoff).
- **Fault Tolerance**: Operating despite component failure.

## Learning Path

1. **Basics**: Understand Load Balancing and Caching.
2. **Intermediate**: Database scaling (Sharding, Replication) and CAP Theorem.
3. **Advanced**: Microservices patterns (Saga, Sidecar) and Event-Driven Architecture.

## Related Categories

- ☁️ **[Cloud Platforms](../cloud-platforms/README.md)** - Implementation targets (AWS, Azure).
- 🏗️ **[Infrastructure & DevOps](../infrastructure-devops/README.md)** - Deploying these architectures.
- 💾 **[Databases](../databases/README.md)** - Storage engines and trade-offs.
