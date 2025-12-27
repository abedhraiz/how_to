# Scalability & Performance Guide

Strategies for handling growth, optimizing performance, and making architectural trade-offs.

## 1. Scaling Strategies

### Vertical Scaling (Scale Up)
Adding more power (CPU, RAM) to an existing machine.
- **Pros**: Simple, no code changes.
- **Cons**: Hardware limits, single point of failure, expensive.

### Horizontal Scaling (Scale Out)
Adding more machines to the pool.
- **Pros**: Unlimited theoretical scale, redundancy.
- **Cons**: Complexity (data consistency, load balancing).

## 2. Caching Strategies

### Caching Layers
- **Client-side**: Browser cache.
- **CDN**: Edge caching for static assets.
- **Load Balancer / Gateway**: API response caching.
- **Application**: In-memory (Redis/Memcached).
- **Database**: Buffer pool.

### Patterns
- **Cache-Aside (Lazy Loading)**: App checks cache; if miss, reads DB and updates cache.
- **Write-Through**: App writes to cache and DB synchronously.
- **Write-Back**: App writes to cache; cache writes to DB asynchronously (risk of data loss).

## 3. Database Scaling

### Replication (Read Scaling)
- **Master-Slave**: Writes go to Master, Reads go to Slaves.
- **Challenge**: Replication lag (Eventual Consistency).

### Sharding (Write Scaling)
- Partitioning data across multiple servers based on a key (e.g., UserID).
- **Challenge**: Cross-shard joins are difficult; rebalancing data is hard.

## 4. The CAP Theorem

In a distributed data store, you can only guarantee two of three:
1. **Consistency**: Every read receives the most recent write or an error.
2. **Availability**: Every request receives a (non-error) response, without the guarantee that it contains the most recent write.
3. **Partition Tolerance**: The system continues to operate despite an arbitrary number of messages being dropped or delayed by the network.

**Real world**: You usually choose between **CP** (Consistency/Partition Tolerance) and **AP** (Availability/Partition Tolerance) because network partitions *will* happen.

## 5. Load Balancing

### Algorithms
- **Round Robin**: Sequential distribution.
- **Least Connections**: Send to server with fewest active requests.
- **IP Hash**: Sticky sessions based on client IP.

### Layers
- **L4 (Transport)**: TCP/UDP level (faster, less context).
- **L7 (Application)**: HTTP level (can route based on URL, headers).
