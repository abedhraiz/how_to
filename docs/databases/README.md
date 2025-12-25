# Databases

## Purpose

Comprehensive guides for relational database systems, focusing on PostgreSQL - from basic setup to advanced optimization, replication, and production deployment patterns.

## Technologies Covered

### Relational Databases
- **[PostgreSQL](./postgresql/postgresql-guide.md)** - Advanced open-source relational database with rich feature set

## Prerequisites

### Basic Requirements
- SQL fundamentals (SELECT, INSERT, UPDATE, DELETE)
- Understanding of relational database concepts
- Basic command-line proficiency
- Understanding of data types and schemas

### Recommended Knowledge
- Database normalization (1NF, 2NF, 3NF)
- ACID properties
- Indexing strategies
- Transaction management
- Backup and recovery concepts

## Common Use Cases

### Application Development
- ✅ Store application data (users, products, orders)
- ✅ Implement complex business logic with SQL
- ✅ Ensure data integrity with constraints
- ✅ Handle concurrent transactions
- ✅ Full-text search capabilities

### Data Analytics
- ✅ Run complex analytical queries
- ✅ Aggregate and summarize data
- ✅ Build data warehouses
- ✅ Generate reports and dashboards
- ✅ Time-series data analysis

### High-Performance Systems
- ✅ Optimize query performance
- ✅ Implement replication for scalability
- ✅ Set up high availability
- ✅ Handle millions of transactions
- ✅ Partition large datasets

### Specialized Features
- ✅ Store and query JSON documents
- ✅ Geospatial data (PostGIS)
- ✅ Full-text search
- ✅ Time-series data (TimescaleDB)
- ✅ Graph queries (Apache AGE)

## Learning Path

### Beginner (1-2 months)
1. **SQL Basics**
   - Install PostgreSQL
   - Create databases and tables
   - Write CRUD queries (SELECT, INSERT, UPDATE, DELETE)
   - Understand data types
   - Use WHERE, ORDER BY, LIMIT

2. **Database Design**
   - Design schemas
   - Create relationships (foreign keys)
   - Implement constraints
   - Understand normalization
   - Basic indexing

3. **Essential Operations**
   - Joins (INNER, LEFT, RIGHT, FULL)
   - Aggregate functions (COUNT, SUM, AVG)
   - GROUP BY and HAVING
   - Subqueries
   - Basic views

### Intermediate (2-3 months)
4. **Advanced SQL**
   - Window functions (ROW_NUMBER, RANK, LAG, LEAD)
   - Common Table Expressions (CTEs)
   - Recursive queries
   - JSONB operations
   - Full-text search

5. **Performance Optimization**
   - Query optimization and EXPLAIN
   - Index types (B-tree, Hash, GiST, GIN)
   - Query planning and statistics
   - Vacuum and maintenance
   - Connection pooling

6. **Administration**
   - User management and permissions
   - Backup and restore
   - Monitoring and logging
   - Configuration tuning
   - Maintenance tasks

### Advanced (3+ months)
7. **High Availability**
   - Replication (streaming, logical)
   - Failover and recovery
   - Load balancing
   - Connection pooling (PgBouncer)
   - Monitoring and alerting

8. **Production Operations**
   - Partitioning strategies
   - Sharding considerations
   - Disaster recovery
   - Migration strategies
   - Performance at scale

## PostgreSQL Architecture

```
Client Application
       ↓
Connection (psql, pgAdmin, application)
       ↓
PostgreSQL Server
       ├─→ Shared Memory
       │   ├─ Shared Buffers
       │   ├─ WAL Buffers
       │   └─ Lock Tables
       ├─→ Background Processes
       │   ├─ Writer
       │   ├─ Checkpointer
       │   ├─ WAL Writer
       │   └─ Autovacuum
       └─→ Storage
           ├─ Data Files
           ├─ WAL (Write-Ahead Log)
           └─ Configuration Files
```

## Database Design Patterns

### Normalized Design (OLTP)
```
Users Table
├─ user_id (PK)
├─ username
└─ email

Orders Table
├─ order_id (PK)
├─ user_id (FK)
├─ order_date
└─ total_amount

Order_Items Table
├─ item_id (PK)
├─ order_id (FK)
├─ product_id (FK)
└─ quantity
```

### Denormalized Design (OLAP)
```
Sales_Fact Table
├─ sale_id
├─ customer_name (denormalized)
├─ product_name (denormalized)
├─ sale_date
├─ quantity
└─ total_amount
```

## Related Categories

- 🔧 **[Data Engineering](../data-engineering/README.md)** - Build data pipelines with PostgreSQL
- ☁️ **[Cloud Platforms](../cloud-platforms/README.md)** - Managed PostgreSQL services (RDS, CloudSQL)
- 🏗️ **[Infrastructure & DevOps](../infrastructure-devops/README.md)** - Deploy and manage databases
- 📊 **[Monitoring & Observability](../monitoring-observability/README.md)** - Monitor database performance
- 🔄 **[CI/CD Automation](../cicd-automation/README.md)** - Automate database migrations

## Quick Start Examples

### Create Database and Tables
```sql
-- Create database
CREATE DATABASE myapp;

-- Connect to database
\c myapp

-- Create users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'pending'
);

-- Create index
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
```

### Advanced Queries
```sql
-- Window function: Running total
SELECT 
    order_date,
    total_amount,
    SUM(total_amount) OVER (ORDER BY order_date) as running_total
FROM orders;

-- Common Table Expression (CTE)
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(total_amount) as total
    FROM orders
    GROUP BY month
)
SELECT 
    month,
    total,
    LAG(total) OVER (ORDER BY month) as prev_month,
    total - LAG(total) OVER (ORDER BY month) as growth
FROM monthly_sales;

-- JSONB operations
SELECT 
    user_id,
    metadata->>'country' as country,
    metadata->'preferences'->>'theme' as theme
FROM users
WHERE metadata @> '{"verified": true}';
```

### Performance Optimization
```sql
-- Analyze query performance
EXPLAIN ANALYZE
SELECT u.username, COUNT(o.order_id) as order_count
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.username;

-- Create partial index
CREATE INDEX idx_pending_orders 
ON orders(order_date) 
WHERE status = 'pending';

-- Create covering index
CREATE INDEX idx_orders_user_status 
ON orders(user_id, status) 
INCLUDE (total_amount);

-- Update statistics
ANALYZE users;
ANALYZE orders;
```

### Backup and Restore
```bash
# Backup database
pg_dump -h localhost -U postgres myapp > myapp_backup.sql

# Backup specific tables
pg_dump -h localhost -U postgres -t users -t orders myapp > tables_backup.sql

# Restore database
psql -h localhost -U postgres myapp < myapp_backup.sql

# Backup in custom format (compressed)
pg_dump -h localhost -U postgres -F c myapp > myapp_backup.dump

# Restore from custom format
pg_restore -h localhost -U postgres -d myapp myapp_backup.dump
```

### Replication Setup
```sql
-- On primary server: Create replication user
CREATE ROLE replication_user WITH REPLICATION LOGIN PASSWORD 'secure_password';

-- Configure pg_hba.conf
-- host replication replication_user replica_ip/32 md5

-- On replica: Create recovery configuration
-- standby_mode = 'on'
-- primary_conninfo = 'host=primary_ip port=5432 user=replication_user password=secure_password'
```

## Best Practices

### Schema Design
1. ✅ **Normalize Appropriately** - Balance normalization vs. performance
2. ✅ **Use Constraints** - Enforce data integrity at database level
3. ✅ **Choose Right Data Types** - Optimize storage and performance
4. ✅ **Plan for Growth** - Consider future scaling needs
5. ✅ **Document Schema** - Maintain clear documentation

### Query Optimization
1. ✅ **Use EXPLAIN ANALYZE** - Understand query execution plans
2. ✅ **Create Appropriate Indexes** - But avoid over-indexing
3. ✅ **Avoid SELECT *** - Query only needed columns
4. ✅ **Use Prepared Statements** - Improve performance and security
5. ✅ **Batch Operations** - Combine multiple inserts/updates

### Operations
1. ✅ **Regular Backups** - Automated, tested backup strategy
2. ✅ **Monitor Performance** - Track slow queries, connection pools
3. ✅ **Vacuum Regularly** - Prevent bloat and maintain statistics
4. ✅ **Update Statistics** - Keep query planner informed
5. ✅ **Connection Pooling** - Use PgBouncer for high-traffic apps

### Security
1. ✅ **Least Privilege** - Grant minimal necessary permissions
2. ✅ **Use SSL/TLS** - Encrypt connections
3. ✅ **Protect Credentials** - Never hardcode passwords
4. ✅ **Regular Updates** - Apply security patches
5. ✅ **Audit Logs** - Track database access

## Performance Tuning

### Key Configuration Parameters
```ini
# Memory settings
shared_buffers = 256MB              # 25% of RAM
effective_cache_size = 1GB          # 50-75% of RAM
work_mem = 8MB                      # Per operation
maintenance_work_mem = 128MB        # For VACUUM, CREATE INDEX

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
max_wal_size = 2GB

# Connection settings
max_connections = 100
```

## Common Patterns

### Soft Delete
```sql
ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP;
CREATE INDEX idx_users_deleted ON users(deleted_at) WHERE deleted_at IS NULL;

-- "Delete" user
UPDATE users SET deleted_at = NOW() WHERE user_id = 123;

-- Query active users
SELECT * FROM users WHERE deleted_at IS NULL;
```

### Audit Trail
```sql
CREATE TABLE audit_log (
    log_id SERIAL PRIMARY KEY,
    table_name VARCHAR(50),
    record_id INTEGER,
    action VARCHAR(10),
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(50),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger function
CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO audit_log (table_name, record_id, action, old_data, new_data, changed_by)
    VALUES (TG_TABLE_NAME, NEW.user_id, TG_OP, row_to_json(OLD), row_to_json(NEW), current_user);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

## Navigation

- [← Back to Main Documentation](../../README.md)
- [→ Next: Version Control](../version-control/README.md)
