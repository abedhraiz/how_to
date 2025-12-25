# PostgreSQL Guide

## What is PostgreSQL?

PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development. It's known for reliability, robustness, and performance.

**Key Features:**
- ACID compliance
- Complex queries and joins
- Foreign keys and constraints
- Views, stored procedures, triggers
- JSON/JSONB support
- Full-text search
- Geospatial data (PostGIS)
- Extensibility

## Prerequisites

- Basic SQL knowledge
- Command line familiarity
- Understanding of database concepts
- Linux/Unix system knowledge (for server setup)

## Installation

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Check status
sudo systemctl status postgresql

# PostgreSQL should start automatically
```

### macOS

```bash
# Using Homebrew
brew install postgresql@15

# Start service
brew services start postgresql@15

# Or start manually
pg_ctl -D /usr/local/var/postgres start
```

### Using Docker

```bash
# Run PostgreSQL container
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -e POSTGRES_USER=myuser \
  -e POSTGRES_DB=mydb \
  -p 5432:5432 \
  -v postgres-data:/var/lib/postgresql/data \
  postgres:15

# Connect to database
docker exec -it postgres psql -U myuser -d mydb
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: postgres
    environment:
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mysecretpassword
      POSTGRES_DB: mydb
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8"
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U myuser"]
      interval: 10s
      timeout: 5s
      retries: 5

  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@example.com
      PGADMIN_DEFAULT_PASSWORD: admin
    ports:
      - "5050:80"
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  postgres-data:
```

```bash
# Start services
docker-compose up -d

# Access pgAdmin at http://localhost:5050
```

## Basic Usage

### Connecting to PostgreSQL

```bash
# Connect as postgres user (default superuser)
sudo -u postgres psql

# Connect to specific database
psql -U myuser -d mydb -h localhost

# Connect with password prompt
psql -U myuser -d mydb -h localhost -W

# Connection string
psql postgresql://myuser:password@localhost:5432/mydb
```

### psql Commands

```sql
-- List databases
\l

-- Connect to database
\c mydb

-- List tables
\dt

-- Describe table
\d table_name

-- List schemas
\dn

-- List users/roles
\du

-- List functions
\df

-- Show current database
SELECT current_database();

-- Show current user
SELECT current_user;

-- Execute SQL from file
\i /path/to/file.sql

-- Output query results to file
\o /path/to/output.txt

-- Quit
\q

-- Help
\?
```

## Database Management

### Create Database

```sql
-- Create database
CREATE DATABASE myapp;

-- Create with options
CREATE DATABASE myapp
  WITH
  OWNER = myuser
  ENCODING = 'UTF8'
  LC_COLLATE = 'en_US.UTF-8'
  LC_CTYPE = 'en_US.UTF-8'
  TEMPLATE = template0;

-- Create from template
CREATE DATABASE test_db
  TEMPLATE myapp;
```

### Drop Database

```sql
-- Drop database
DROP DATABASE myapp;

-- Drop if exists
DROP DATABASE IF EXISTS myapp;

-- Force drop (disconnect users)
SELECT pg_terminate_backend(pg_stat_activity.pid)
FROM pg_stat_activity
WHERE pg_stat_activity.datname = 'myapp'
  AND pid <> pg_backend_pid();

DROP DATABASE myapp;
```

### User Management

```sql
-- Create user
CREATE USER myuser WITH PASSWORD 'mypassword';

-- Create user with options
CREATE USER admin_user
  WITH
  PASSWORD 'securepassword'
  CREATEDB
  CREATEROLE
  LOGIN;

-- Alter user
ALTER USER myuser WITH PASSWORD 'newpassword';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO myuser;
GRANT USAGE ON SCHEMA public TO myuser;

-- Revoke privileges
REVOKE ALL PRIVILEGES ON DATABASE mydb FROM myuser;

-- Drop user
DROP USER myuser;

-- List users
\du
SELECT * FROM pg_user;
```

## Table Operations

### Create Tables

```sql
-- Basic table
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  email VARCHAR(100) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table with constraints
CREATE TABLE orders (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL,
  total_amount DECIMAL(10, 2) NOT NULL CHECK (total_amount >= 0),
  status VARCHAR(20) DEFAULT 'pending',
  order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Table with composite key
CREATE TABLE order_items (
  order_id INTEGER,
  product_id INTEGER,
  quantity INTEGER NOT NULL CHECK (quantity > 0),
  price DECIMAL(10, 2) NOT NULL,
  PRIMARY KEY (order_id, product_id),
  FOREIGN KEY (order_id) REFERENCES orders(id) ON DELETE CASCADE,
  FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Table with JSON
CREATE TABLE user_preferences (
  user_id INTEGER PRIMARY KEY,
  settings JSONB DEFAULT '{}',
  FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Table with arrays
CREATE TABLE posts (
  id SERIAL PRIMARY KEY,
  title VARCHAR(200) NOT NULL,
  tags TEXT[],
  author_id INTEGER REFERENCES users(id)
);
```

### Alter Tables

```sql
-- Add column
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- Drop column
ALTER TABLE users DROP COLUMN phone;

-- Rename column
ALTER TABLE users RENAME COLUMN username TO user_name;

-- Change column type
ALTER TABLE users ALTER COLUMN email TYPE TEXT;

-- Add constraint
ALTER TABLE users ADD CONSTRAINT email_check CHECK (email LIKE '%@%');

-- Drop constraint
ALTER TABLE users DROP CONSTRAINT email_check;

-- Add foreign key
ALTER TABLE orders ADD FOREIGN KEY (user_id) REFERENCES users(id);

-- Rename table
ALTER TABLE orders RENAME TO customer_orders;
```

### Drop Tables

```sql
-- Drop table
DROP TABLE users;

-- Drop if exists
DROP TABLE IF EXISTS users;

-- Drop with cascade (removes dependent objects)
DROP TABLE users CASCADE;
```

## Querying Data

### SELECT Statements

```sql
-- Select all
SELECT * FROM users;

-- Select specific columns
SELECT username, email FROM users;

-- With WHERE clause
SELECT * FROM users WHERE id = 1;
SELECT * FROM users WHERE username LIKE 'john%';
SELECT * FROM users WHERE created_at > '2024-01-01';

-- With ORDER BY
SELECT * FROM users ORDER BY created_at DESC;
SELECT * FROM users ORDER BY username ASC, email DESC;

-- With LIMIT and OFFSET
SELECT * FROM users LIMIT 10;
SELECT * FROM users LIMIT 10 OFFSET 20;

-- With DISTINCT
SELECT DISTINCT status FROM orders;

-- With aggregate functions
SELECT COUNT(*) FROM users;
SELECT AVG(total_amount) FROM orders;
SELECT SUM(total_amount) FROM orders;
SELECT MAX(total_amount), MIN(total_amount) FROM orders;

-- With GROUP BY
SELECT status, COUNT(*) 
FROM orders 
GROUP BY status;

SELECT user_id, COUNT(*) as order_count, SUM(total_amount) as total_spent
FROM orders
GROUP BY user_id;

-- With HAVING
SELECT status, COUNT(*) as count
FROM orders
GROUP BY status
HAVING COUNT(*) > 10;
```

### JOINs

```sql
-- INNER JOIN
SELECT users.username, orders.id, orders.total_amount
FROM users
INNER JOIN orders ON users.id = orders.user_id;

-- LEFT JOIN
SELECT users.username, orders.id
FROM users
LEFT JOIN orders ON users.id = orders.user_id;

-- RIGHT JOIN
SELECT users.username, orders.id
FROM users
RIGHT JOIN orders ON users.id = orders.user_id;

-- FULL OUTER JOIN
SELECT users.username, orders.id
FROM users
FULL OUTER JOIN orders ON users.id = orders.user_id;

-- Multiple JOINs
SELECT 
  users.username,
  orders.id as order_id,
  order_items.product_id,
  order_items.quantity
FROM users
INNER JOIN orders ON users.id = orders.user_id
INNER JOIN order_items ON orders.id = order_items.order_id;

-- Self JOIN
SELECT 
  e.name as employee,
  m.name as manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

### Subqueries

```sql
-- Subquery in WHERE
SELECT * FROM users
WHERE id IN (SELECT user_id FROM orders WHERE total_amount > 100);

-- Subquery in SELECT
SELECT 
  username,
  (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) as order_count
FROM users;

-- Subquery in FROM
SELECT avg_amount
FROM (
  SELECT AVG(total_amount) as avg_amount
  FROM orders
  GROUP BY user_id
) as subquery;

-- EXISTS
SELECT * FROM users
WHERE EXISTS (
  SELECT 1 FROM orders WHERE orders.user_id = users.id
);
```

## Inserting Data

```sql
-- Insert single row
INSERT INTO users (username, email, password_hash)
VALUES ('john', 'john@example.com', 'hashed_password');

-- Insert multiple rows
INSERT INTO users (username, email, password_hash)
VALUES 
  ('alice', 'alice@example.com', 'hash1'),
  ('bob', 'bob@example.com', 'hash2'),
  ('charlie', 'charlie@example.com', 'hash3');

-- Insert with RETURNING
INSERT INTO users (username, email, password_hash)
VALUES ('dave', 'dave@example.com', 'hash4')
RETURNING id, username;

-- Insert from SELECT
INSERT INTO archive_orders
SELECT * FROM orders WHERE order_date < '2023-01-01';

-- Insert with ON CONFLICT (upsert)
INSERT INTO users (username, email, password_hash)
VALUES ('john', 'john@example.com', 'new_hash')
ON CONFLICT (username) 
DO UPDATE SET email = EXCLUDED.email, password_hash = EXCLUDED.password_hash;

-- Insert or do nothing
INSERT INTO users (username, email, password_hash)
VALUES ('john', 'john@example.com', 'hash')
ON CONFLICT (username) DO NOTHING;
```

## Updating Data

```sql
-- Update single column
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;

-- Update multiple columns
UPDATE users 
SET 
  email = 'updated@example.com',
  updated_at = CURRENT_TIMESTAMP
WHERE id = 1;

-- Update with calculation
UPDATE products 
SET price = price * 1.1
WHERE category = 'electronics';

-- Update from another table
UPDATE orders
SET total_amount = (
  SELECT SUM(quantity * price)
  FROM order_items
  WHERE order_items.order_id = orders.id
);

-- Update with JOIN
UPDATE users
SET status = 'active'
FROM orders
WHERE users.id = orders.user_id
  AND orders.order_date > '2024-01-01';

-- Update with RETURNING
UPDATE users
SET status = 'inactive'
WHERE last_login < '2023-01-01'
RETURNING id, username;
```

## Deleting Data

```sql
-- Delete specific rows
DELETE FROM users WHERE id = 1;

-- Delete with condition
DELETE FROM orders WHERE status = 'cancelled';

-- Delete all rows (keep structure)
DELETE FROM orders;

-- Delete with RETURNING
DELETE FROM users
WHERE status = 'deleted'
RETURNING id, username;

-- Delete with JOIN
DELETE FROM order_items
USING orders
WHERE order_items.order_id = orders.id
  AND orders.status = 'cancelled';

-- Truncate (faster for all rows)
TRUNCATE TABLE orders;
TRUNCATE TABLE orders RESTART IDENTITY CASCADE;
```

## Indexes

```sql
-- Create index
CREATE INDEX idx_users_email ON users(email);

-- Create unique index
CREATE UNIQUE INDEX idx_users_username ON users(username);

-- Create composite index
CREATE INDEX idx_orders_user_date ON orders(user_id, order_date);

-- Create partial index
CREATE INDEX idx_active_users ON users(username) WHERE status = 'active';

-- Create index with expression
CREATE INDEX idx_lower_email ON users(LOWER(email));

-- Create GIN index for JSON
CREATE INDEX idx_settings ON user_preferences USING GIN (settings);

-- Create GiST index for full-text search
CREATE INDEX idx_posts_fulltext ON posts USING GiST (to_tsvector('english', title));

-- List indexes
\di

-- Drop index
DROP INDEX idx_users_email;

-- Analyze index usage
SELECT 
  schemaname,
  tablename,
  indexname,
  idx_scan as index_scans,
  idx_tup_read as tuples_read,
  idx_tup_fetch as tuples_fetched
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## Views

```sql
-- Create view
CREATE VIEW active_users AS
SELECT id, username, email
FROM users
WHERE status = 'active';

-- Create or replace view
CREATE OR REPLACE VIEW user_orders AS
SELECT 
  users.id,
  users.username,
  COUNT(orders.id) as order_count,
  COALESCE(SUM(orders.total_amount), 0) as total_spent
FROM users
LEFT JOIN orders ON users.id = orders.user_id
GROUP BY users.id, users.username;

-- Materialized view (cached results)
CREATE MATERIALIZED VIEW daily_sales AS
SELECT 
  DATE(order_date) as sale_date,
  COUNT(*) as order_count,
  SUM(total_amount) as total_sales
FROM orders
GROUP BY DATE(order_date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW daily_sales;

-- Drop view
DROP VIEW active_users;
DROP MATERIALIZED VIEW daily_sales;
```

## Transactions

```sql
-- Basic transaction
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;

-- Rollback on error
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
-- Something goes wrong
ROLLBACK;

-- Savepoints
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
SAVEPOINT my_savepoint;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
-- Oops, rollback to savepoint
ROLLBACK TO SAVEPOINT my_savepoint;
-- Continue with different operation
UPDATE accounts SET balance = balance + 100 WHERE id = 3;
COMMIT;

-- Isolation levels
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- Your queries
COMMIT;
```

## Functions and Procedures

### Functions

```sql
-- Simple function
CREATE FUNCTION get_user_count() 
RETURNS INTEGER AS $$
BEGIN
  RETURN (SELECT COUNT(*) FROM users);
END;
$$ LANGUAGE plpgsql;

-- Use function
SELECT get_user_count();

-- Function with parameters
CREATE FUNCTION get_user_orders(user_id_param INTEGER)
RETURNS TABLE(order_id INTEGER, total DECIMAL) AS $$
BEGIN
  RETURN QUERY
  SELECT id, total_amount
  FROM orders
  WHERE user_id = user_id_param;
END;
$$ LANGUAGE plpgsql;

-- Use function
SELECT * FROM get_user_orders(1);

-- Function with default parameters
CREATE FUNCTION greet(name TEXT DEFAULT 'World')
RETURNS TEXT AS $$
BEGIN
  RETURN 'Hello, ' || name || '!';
END;
$$ LANGUAGE plpgsql;

-- Drop function
DROP FUNCTION get_user_count();
```

### Stored Procedures

```sql
-- Create procedure
CREATE PROCEDURE transfer_funds(
  sender_id INTEGER,
  receiver_id INTEGER,
  amount DECIMAL
)
LANGUAGE plpgsql
AS $$
BEGIN
  -- Deduct from sender
  UPDATE accounts SET balance = balance - amount WHERE id = sender_id;
  
  -- Add to receiver
  UPDATE accounts SET balance = balance + amount WHERE id = receiver_id;
  
  -- Log transaction
  INSERT INTO transaction_log (sender_id, receiver_id, amount, timestamp)
  VALUES (sender_id, receiver_id, amount, CURRENT_TIMESTAMP);
  
  COMMIT;
END;
$$;

-- Call procedure
CALL transfer_funds(1, 2, 100.00);
```

## Triggers

```sql
-- Create trigger function
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = CURRENT_TIMESTAMP;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger
CREATE TRIGGER update_users_modtime
BEFORE UPDATE ON users
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- Audit trigger
CREATE TABLE audit_log (
  id SERIAL PRIMARY KEY,
  table_name TEXT,
  operation TEXT,
  old_data JSONB,
  new_data JSONB,
  changed_by TEXT,
  changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION audit_trigger()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO audit_log (table_name, operation, old_data, new_data, changed_by)
  VALUES (
    TG_TABLE_NAME,
    TG_OP,
    CASE WHEN TG_OP IN ('UPDATE', 'DELETE') THEN row_to_json(OLD) ELSE NULL END,
    CASE WHEN TG_OP IN ('INSERT', 'UPDATE') THEN row_to_json(NEW) ELSE NULL END,
    current_user
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to table
CREATE TRIGGER users_audit
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW
EXECUTE FUNCTION audit_trigger();

-- Drop trigger
DROP TRIGGER update_users_modtime ON users;
```

## JSON Operations

```sql
-- Create table with JSONB
CREATE TABLE products (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100),
  attributes JSONB
);

-- Insert JSON data
INSERT INTO products (name, attributes)
VALUES 
  ('Laptop', '{"brand": "Dell", "ram": 16, "storage": 512}'),
  ('Phone', '{"brand": "Apple", "model": "iPhone 14", "color": "black"}');

-- Query JSON field
SELECT name, attributes->>'brand' as brand FROM products;

-- Query nested JSON
SELECT name, attributes->'specs'->>'cpu' as cpu FROM products;

-- Query with JSON operators
SELECT * FROM products WHERE attributes->>'brand' = 'Dell';
SELECT * FROM products WHERE attributes @> '{"brand": "Apple"}';

-- Update JSON field
UPDATE products
SET attributes = attributes || '{"warranty": "2 years"}'
WHERE id = 1;

-- Remove JSON key
UPDATE products
SET attributes = attributes - 'color'
WHERE id = 2;

-- JSON aggregation
SELECT jsonb_agg(jsonb_build_object('name', name, 'brand', attributes->>'brand'))
FROM products;

-- Create GIN index for JSON queries
CREATE INDEX idx_attributes ON products USING GIN (attributes);
```

## Full-Text Search

```sql
-- Create table
CREATE TABLE articles (
  id SERIAL PRIMARY KEY,
  title TEXT,
  content TEXT,
  search_vector tsvector
);

-- Generate search vector
UPDATE articles
SET search_vector = 
  to_tsvector('english', coalesce(title, '') || ' ' || coalesce(content, ''));

-- Create GIN index
CREATE INDEX idx_search ON articles USING GIN (search_vector);

-- Search
SELECT title, ts_rank(search_vector, query) as rank
FROM articles, to_tsquery('english', 'postgresql & database') query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Automatic update with trigger
CREATE TRIGGER articles_search_update
BEFORE INSERT OR UPDATE ON articles
FOR EACH ROW
EXECUTE FUNCTION
  tsvector_update_trigger(search_vector, 'pg_catalog.english', title, content);
```

## Backup and Restore

### pg_dump (Backup)

```bash
# Backup database
pg_dump mydb > mydb_backup.sql

# Backup with compression
pg_dump mydb | gzip > mydb_backup.sql.gz

# Backup specific tables
pg_dump -t users -t orders mydb > tables_backup.sql

# Backup in custom format
pg_dump -Fc mydb > mydb_backup.dump

# Backup all databases
pg_dumpall > all_databases.sql

# Backup only schema
pg_dump --schema-only mydb > schema.sql

# Backup only data
pg_dump --data-only mydb > data.sql
```

### pg_restore (Restore)

```bash
# Restore from SQL file
psql mydb < mydb_backup.sql

# Restore from compressed file
gunzip -c mydb_backup.sql.gz | psql mydb

# Restore from custom format
pg_restore -d mydb mydb_backup.dump

# Restore specific table
pg_restore -d mydb -t users mydb_backup.dump

# Create database and restore
createdb newdb
pg_restore -d newdb mydb_backup.dump
```

## Performance Optimization

### EXPLAIN

```sql
-- Show query plan
EXPLAIN SELECT * FROM users WHERE username = 'john';

-- Show query plan with execution
EXPLAIN ANALYZE SELECT * FROM users WHERE username = 'john';

-- Detailed output
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT * FROM users WHERE username = 'john';
```

### VACUUM and ANALYZE

```sql
-- Vacuum table (reclaim storage)
VACUUM users;

-- Vacuum with analyze (update statistics)
VACUUM ANALYZE users;

-- Full vacuum (more thorough but locks table)
VACUUM FULL users;

-- Analyze only (update statistics)
ANALYZE users;

-- Auto vacuum (configure in postgresql.conf)
-- autovacuum = on
```

### Query Optimization Tips

```sql
-- Use indexes
CREATE INDEX idx_users_email ON users(email);

-- Use LIMIT
SELECT * FROM users LIMIT 100;

-- Use proper JOIN types
-- INNER JOIN instead of WHERE EXISTS when possible

-- Avoid SELECT *
SELECT id, username FROM users; -- Better

-- Use prepared statements (prevents SQL injection and improves performance)

-- Partition large tables
CREATE TABLE orders_2024 PARTITION OF orders
FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

## Security Best Practices

```sql
-- Use strong passwords
ALTER USER myuser WITH PASSWORD 'str0ng!P@ssw0rd';

-- Limit privileges
GRANT SELECT ON users TO readonly_user;

-- Use SSL connections
-- In postgresql.conf: ssl = on

-- Restrict network access
-- In pg_hba.conf:
-- host    all    all    10.0.0.0/24    md5

-- Use row-level security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_policy ON users
FOR SELECT
USING (id = current_user_id());

-- Encrypt sensitive data
CREATE EXTENSION pgcrypto;

INSERT INTO users (username, password_hash)
VALUES ('john', crypt('password', gen_salt('bf')));

-- Verify password
SELECT (password_hash = crypt('entered_password', password_hash)) AS valid
FROM users WHERE username = 'john';
```

## Connection from Applications

### Python (psycopg2)

```python
import psycopg2
from psycopg2.extras import RealDictCursor

# Connect
conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="myuser",
    password="mypassword"
)

# Create cursor
cur = conn.cursor(cursor_factory=RealDictCursor)

# Execute query
cur.execute("SELECT * FROM users WHERE id = %s", (1,))
user = cur.fetchone()
print(user['username'])

# Insert data
cur.execute(
    "INSERT INTO users (username, email) VALUES (%s, %s) RETURNING id",
    ('john', 'john@example.com')
)
user_id = cur.fetchone()['id']

# Commit changes
conn.commit()

# Close connections
cur.close()
conn.close()

# Using context manager
with psycopg2.connect(database="mydb", user="myuser") as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users")
        users = cur.fetchall()
```

### Node.js (pg)

```javascript
const { Pool } = require('pg');

// Create connection pool
const pool = new Pool({
  host: 'localhost',
  database: 'mydb',
  user: 'myuser',
  password: 'mypassword',
  port: 5432,
});

// Query
async function getUsers() {
  const result = await pool.query('SELECT * FROM users');
  return result.rows;
}

// Parameterized query
async function getUser(id) {
  const result = await pool.query(
    'SELECT * FROM users WHERE id = $1',
    [id]
  );
  return result.rows[0];
}

// Insert
async function createUser(username, email) {
  const result = await pool.query(
    'INSERT INTO users (username, email) VALUES ($1, $2) RETURNING *',
    [username, email]
  );
  return result.rows[0];
}

// Transaction
async function transfer(senderId, receiverId, amount) {
  const client = await pool.connect();
  
  try {
    await client.query('BEGIN');
    
    await client.query(
      'UPDATE accounts SET balance = balance - $1 WHERE id = $2',
      [amount, senderId]
    );
    
    await client.query(
      'UPDATE accounts SET balance = balance + $1 WHERE id = $2',
      [amount, receiverId]
    );
    
    await client.query('COMMIT');
  } catch (e) {
    await client.query('ROLLBACK');
    throw e;
  } finally {
    client.release();
  }
}

// Close pool
pool.end();
```

## Real-World Examples

### Example 1: E-Commerce Database Schema

```sql
-- Complete e-commerce schema with relationships

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    email_verified BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    CONSTRAINT check_status CHECK (status IN ('active', 'inactive', 'suspended'))
);

-- User addresses
CREATE TABLE addresses (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    address_type VARCHAR(20) NOT NULL,
    street_address VARCHAR(255) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100) NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_address_type CHECK (address_type IN ('shipping', 'billing'))
);

-- Categories
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    parent_id INTEGER REFERENCES categories(id),
    description TEXT,
    image_url VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    compare_price DECIMAL(10, 2),
    cost_price DECIMAL(10, 2),
    sku VARCHAR(100) UNIQUE NOT NULL,
    barcode VARCHAR(100),
    quantity INTEGER DEFAULT 0,
    category_id INTEGER REFERENCES categories(id),
    brand VARCHAR(100),
    weight DECIMAL(10, 2),
    dimensions JSONB,
    images JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    is_featured BOOLEAN DEFAULT FALSE,
    seo_title VARCHAR(255),
    seo_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_price CHECK (price >= 0),
    CONSTRAINT check_quantity CHECK (quantity >= 0)
);

-- Product variants
CREATE TABLE product_variants (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    sku VARCHAR(100) UNIQUE NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    quantity INTEGER DEFAULT 0,
    attributes JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_number VARCHAR(50) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    subtotal DECIMAL(10, 2) NOT NULL,
    tax_amount DECIMAL(10, 2) DEFAULT 0,
    shipping_amount DECIMAL(10, 2) DEFAULT 0,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    total_amount DECIMAL(10, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    payment_method VARCHAR(50),
    payment_status VARCHAR(20) DEFAULT 'pending',
    shipping_address_id INTEGER REFERENCES addresses(id),
    billing_address_id INTEGER REFERENCES addresses(id),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    shipped_at TIMESTAMP,
    delivered_at TIMESTAMP,
    cancelled_at TIMESTAMP,
    CONSTRAINT check_status CHECK (status IN ('pending', 'processing', 'shipped', 'delivered', 'cancelled', 'refunded')),
    CONSTRAINT check_payment_status CHECK (payment_status IN ('pending', 'paid', 'failed', 'refunded'))
);

-- Order items
CREATE TABLE order_items (
    id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id),
    variant_id INTEGER REFERENCES product_variants(id),
    product_name VARCHAR(255) NOT NULL,
    sku VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    total DECIMAL(10, 2) NOT NULL,
    CONSTRAINT check_quantity CHECK (quantity > 0)
);

-- Reviews
CREATE TABLE product_reviews (
    id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER NOT NULL,
    title VARCHAR(255),
    comment TEXT,
    verified_purchase BOOLEAN DEFAULT FALSE,
    is_approved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_rating CHECK (rating BETWEEN 1 AND 5),
    UNIQUE(product_id, user_id)
);

-- Cart
CREATE TABLE cart_items (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(id) ON DELETE CASCADE,
    variant_id INTEGER REFERENCES product_variants(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_quantity CHECK (quantity > 0),
    UNIQUE(user_id, product_id, variant_id)
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_products_category ON products(category_id);
CREATE INDEX idx_products_slug ON products(slug);
CREATE INDEX idx_products_active ON products(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created ON orders(created_at DESC);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_reviews_product ON product_reviews(product_id);
CREATE INDEX idx_reviews_approved ON product_reviews(is_approved) WHERE is_approved = TRUE;

-- Full-text search index
ALTER TABLE products ADD COLUMN search_vector tsvector;

CREATE INDEX idx_products_search ON products USING GIN(search_vector);

CREATE OR REPLACE FUNCTION products_search_trigger() RETURNS trigger AS $$
BEGIN
  NEW.search_vector :=
    setweight(to_tsvector('english', COALESCE(NEW.name, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(NEW.description, '')), 'B');
  RETURN NEW;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER products_search_update 
BEFORE INSERT OR UPDATE ON products
FOR EACH ROW EXECUTE FUNCTION products_search_trigger();

-- Useful views
CREATE VIEW product_stats AS
SELECT 
    p.id,
    p.name,
    p.price,
    p.quantity,
    COUNT(DISTINCT r.id) as review_count,
    AVG(r.rating) as avg_rating,
    COUNT(DISTINCT oi.id) as total_sold
FROM products p
LEFT JOIN product_reviews r ON p.id = r.product_id AND r.is_approved = TRUE
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name, p.price, p.quantity;

CREATE VIEW order_summary AS
SELECT 
    o.id,
    o.order_number,
    u.username,
    u.email,
    o.status,
    o.total_amount,
    o.created_at,
    COUNT(oi.id) as item_count
FROM orders o
JOIN users u ON o.user_id = u.id
LEFT JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id, o.order_number, u.username, u.email, o.status, o.total_amount, o.created_at;

-- Sample data
INSERT INTO users (username, email, password_hash, first_name, last_name) VALUES
('john_doe', 'john@example.com', 'hashed_password_1', 'John', 'Doe'),
('jane_smith', 'jane@example.com', 'hashed_password_2', 'Jane', 'Smith'),
('bob_wilson', 'bob@example.com', 'hashed_password_3', 'Bob', 'Wilson');

INSERT INTO categories (name, slug, description) VALUES
('Electronics', 'electronics', 'Electronic devices and accessories'),
('Clothing', 'clothing', 'Apparel and fashion items'),
('Books', 'books', 'Books and magazines');

INSERT INTO products (name, slug, price, sku, quantity, category_id, description) VALUES
('Laptop Pro 15', 'laptop-pro-15', 1299.99, 'LAPTOP-001', 50, 1, 'Professional laptop with high performance'),
('Wireless Mouse', 'wireless-mouse', 29.99, 'MOUSE-001', 200, 1, 'Ergonomic wireless mouse'),
('Cotton T-Shirt', 'cotton-tshirt', 19.99, 'SHIRT-001', 100, 2, 'Comfortable cotton t-shirt');
```

### Example 2: Advanced Query Examples

```sql
-- Complex analytics query
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        SUM(total_amount) as revenue,
        COUNT(*) as order_count,
        AVG(total_amount) as avg_order_value
    FROM orders
    WHERE status IN ('delivered', 'shipped')
    GROUP BY DATE_TRUNC('month', created_at)
),
monthly_growth AS (
    SELECT 
        month,
        revenue,
        LAG(revenue) OVER (ORDER BY month) as previous_revenue,
        ((revenue - LAG(revenue) OVER (ORDER BY month)) / 
         LAG(revenue) OVER (ORDER BY month) * 100) as growth_rate
    FROM monthly_sales
)
SELECT 
    TO_CHAR(month, 'YYYY-MM') as month,
    ROUND(revenue, 2) as revenue,
    ROUND(previous_revenue, 2) as previous_revenue,
    ROUND(growth_rate, 2) as growth_percentage
FROM monthly_growth
ORDER BY month DESC;

-- Top selling products
SELECT 
    p.id,
    p.name,
    p.price,
    COUNT(oi.id) as times_sold,
    SUM(oi.quantity) as total_quantity,
    SUM(oi.total) as total_revenue,
    ROUND(AVG(r.rating), 2) as avg_rating,
    COUNT(DISTINCT r.id) as review_count
FROM products p
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
LEFT JOIN product_reviews r ON p.id = r.product_id AND r.is_approved = TRUE
WHERE o.status IN ('delivered', 'shipped')
    AND o.created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY p.id, p.name, p.price
ORDER BY total_revenue DESC
LIMIT 10;

-- Customer lifetime value
SELECT 
    u.id,
    u.username,
    u.email,
    COUNT(DISTINCT o.id) as total_orders,
    SUM(o.total_amount) as lifetime_value,
    AVG(o.total_amount) as avg_order_value,
    MAX(o.created_at) as last_order_date,
    EXTRACT(DAY FROM CURRENT_TIMESTAMP - MAX(o.created_at)) as days_since_last_order,
    CASE 
        WHEN EXTRACT(DAY FROM CURRENT_TIMESTAMP - MAX(o.created_at)) < 30 THEN 'Active'
        WHEN EXTRACT(DAY FROM CURRENT_TIMESTAMP - MAX(o.created_at)) < 90 THEN 'At Risk'
        ELSE 'Churned'
    END as customer_status
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
WHERE u.status = 'active'
GROUP BY u.id, u.username, u.email
ORDER BY lifetime_value DESC;

-- Inventory alert
SELECT 
    p.id,
    p.name,
    p.sku,
    p.quantity as current_stock,
    COALESCE(SUM(oi.quantity), 0) as sold_last_30_days,
    ROUND(COALESCE(SUM(oi.quantity), 0) / 30.0, 2) as daily_avg_sales,
    CASE 
        WHEN p.quantity = 0 THEN 'OUT_OF_STOCK'
        WHEN p.quantity < (COALESCE(SUM(oi.quantity), 0) / 30.0 * 7) THEN 'LOW_STOCK'
        ELSE 'OK'
    END as stock_status,
    ROUND(p.quantity / NULLIF(COALESCE(SUM(oi.quantity), 0) / 30.0, 0), 1) as days_of_stock
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.id 
    AND o.created_at >= CURRENT_DATE - INTERVAL '30 days'
    AND o.status IN ('delivered', 'shipped')
WHERE p.is_active = TRUE
GROUP BY p.id, p.name, p.sku, p.quantity
HAVING p.quantity < (COALESCE(SUM(oi.quantity), 0) / 30.0 * 7) OR p.quantity = 0
ORDER BY stock_status, days_of_stock;

-- Product recommendations (collaborative filtering)
WITH user_purchases AS (
    SELECT DISTINCT
        o.user_id,
        oi.product_id
    FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    WHERE o.status IN ('delivered', 'shipped')
),
similar_users AS (
    SELECT 
        up1.user_id as user1,
        up2.user_id as user2,
        COUNT(*) as common_products
    FROM user_purchases up1
    JOIN user_purchases up2 ON up1.product_id = up2.product_id
        AND up1.user_id != up2.user_id
    WHERE up1.user_id = 1  -- Target user
    GROUP BY up1.user_id, up2.user_id
    HAVING COUNT(*) >= 2
)
SELECT DISTINCT
    p.id,
    p.name,
    p.price,
    COUNT(DISTINCT su.user2) as recommended_by,
    AVG(r.rating) as avg_rating
FROM similar_users su
JOIN user_purchases up ON su.user2 = up.user_id
JOIN products p ON up.product_id = p.id
LEFT JOIN product_reviews r ON p.id = r.product_id AND r.is_approved = TRUE
WHERE NOT EXISTS (
    SELECT 1 FROM user_purchases 
    WHERE user_id = 1 AND product_id = p.id
)
AND p.is_active = TRUE
GROUP BY p.id, p.name, p.price
ORDER BY recommended_by DESC, avg_rating DESC
LIMIT 10;
```

### Example 3: Performance Optimization

```sql
-- Query performance analysis
EXPLAIN ANALYZE
SELECT 
    p.name,
    p.price,
    COUNT(oi.id) as sales_count
FROM products p
LEFT JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name, p.price
ORDER BY sales_count DESC;

-- Create covering index
CREATE INDEX idx_order_items_product_covering 
ON order_items(product_id, id);

-- Partition large table by date
CREATE TABLE orders_partitioned (
    LIKE orders INCLUDING ALL
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2023 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE orders_2024 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Materialized view for complex aggregations
CREATE MATERIALIZED VIEW daily_sales_summary AS
SELECT 
    DATE(created_at) as sale_date,
    COUNT(DISTINCT id) as order_count,
    SUM(total_amount) as total_revenue,
    AVG(total_amount) as avg_order_value,
    COUNT(DISTINCT user_id) as unique_customers
FROM orders
WHERE status IN ('delivered', 'shipped')
GROUP BY DATE(created_at);

CREATE UNIQUE INDEX ON daily_sales_summary (sale_date);

-- Refresh materialized view
REFRESH MATERIALIZED VIEW CONCURRENTLY daily_sales_summary;

-- Query optimization with CTEs
WITH RECURSIVE category_tree AS (
    -- Base case: root categories
    SELECT id, name, parent_id, 1 as level, ARRAY[id] as path
    FROM categories
    WHERE parent_id IS NULL
    
    UNION ALL
    
    -- Recursive case: child categories
    SELECT c.id, c.name, c.parent_id, ct.level + 1, ct.path || c.id
    FROM categories c
    JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT 
    REPEAT('  ', level - 1) || name as category_hierarchy,
    level,
    path
FROM category_tree
ORDER BY path;
```

### Example 4: Database Maintenance Scripts

```sql
-- Vacuum and analyze
VACUUM ANALYZE products;
VACUUM ANALYZE orders;

-- Reindex
REINDEX TABLE products;

-- Update statistics
ANALYZE products;

-- Check table bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as indexes_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Find missing indexes
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE schemaname = 'public'
    AND n_distinct > 100
    AND correlation < 0.1
ORDER BY n_distinct DESC;

-- Identify slow queries
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time,
    stddev_exec_time
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_%'
ORDER BY mean_exec_time DESC
LIMIT 20;

-- Connection monitoring
SELECT 
    datname,
    count(*) as connections,
    max(backend_start) as oldest_connection
FROM pg_stat_activity
WHERE state = 'active'
GROUP BY datname;

-- Lock monitoring
SELECT 
    l.locktype,
    l.database,
    l.relation::regclass,
    l.page,
    l.tuple,
    l.transactionid,
    l.mode,
    l.granted,
    a.usename,
    a.query,
    a.query_start
FROM pg_locks l
JOIN pg_stat_activity a ON l.pid = a.pid
WHERE NOT l.granted
ORDER BY a.query_start;
```

### Example 5: Backup and Restore

```bash
# Full database backup
pg_dump -U myuser -d mydb -F c -f backup_$(date +%Y%m%d).dump

# Backup with compression
pg_dump -U myuser -d mydb -F c -Z 9 -f backup_compressed.dump

# Backup specific tables
pg_dump -U myuser -d mydb -t products -t orders -F c -f tables_backup.dump

# Backup schema only
pg_dump -U myuser -d mydb --schema-only -f schema_backup.sql

# Backup data only
pg_dump -U myuser -d mydb --data-only -f data_backup.sql

# Restore database
pg_restore -U myuser -d mydb -c backup.dump

# Restore specific tables
pg_restore -U myuser -d mydb -t products backup.dump

# Continuous archiving (PITR)
# In postgresql.conf:
# wal_level = replica
# archive_mode = on
# archive_command = 'cp %p /path/to/archive/%f'

# Base backup
pg_basebackup -U myuser -D /path/to/backup -F tar -z -P

# Point-in-time recovery
# Create recovery.conf:
restore_command = 'cp /path/to/archive/%f %p'
recovery_target_time = '2024-01-15 10:00:00'
```

### Example 6: Replication Setup

```sql
-- Primary server setup (postgresql.conf)
-- wal_level = replica
-- max_wal_senders = 3
-- wal_keep_size = 64
-- hot_standby = on

-- Create replication user
CREATE ROLE replicator WITH REPLICATION PASSWORD 'replicator_password' LOGIN;

-- Configure pg_hba.conf
-- host replication replicator replica_ip/32 md5

-- On replica server:
-- 1. Stop PostgreSQL
-- 2. Clear data directory
-- 3. Create base backup
-- pg_basebackup -h primary_ip -D /var/lib/postgresql/data -U replicator -P -v -R -X stream -C -S replica_slot

-- 4. Start PostgreSQL on replica

-- Check replication status (on primary)
SELECT * FROM pg_stat_replication;

-- Check replica lag
SELECT 
    client_addr,
    state,
    sent_lsn,
    write_lsn,
    flush_lsn,
    replay_lsn,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) as lag_bytes
FROM pg_stat_replication;
```

### Example 7: Full-Text Search

```sql
-- Add full-text search column
ALTER TABLE products ADD COLUMN search_vector tsvector;

-- Update search vector
UPDATE products 
SET search_vector = 
    setweight(to_tsvector('english', COALESCE(name, '')), 'A') ||
    setweight(to_tsvector('english', COALESCE(description, '')), 'B');

-- Create index
CREATE INDEX idx_products_search ON products USING GIN(search_vector);

-- Search query
SELECT 
    id,
    name,
    description,
    ts_rank(search_vector, query) as rank
FROM products, 
     to_tsquery('english', 'laptop & wireless') as query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- Search with highlighting
SELECT 
    id,
    name,
    ts_headline('english', description, query, 'MaxWords=20, MinWords=10') as highlighted
FROM products,
     to_tsquery('english', 'laptop & gaming') as query
WHERE search_vector @@ query;

-- Fuzzy search
SELECT 
    id,
    name,
    similarity(name, 'lapton') as similarity_score
FROM products
WHERE similarity(name, 'lapton') > 0.3
ORDER BY similarity_score DESC;

-- Create GIN index for similarity
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_products_name_trgm ON products USING GIN(name gin_trgm_ops);
```

### Example 8: JSON Operations

```sql
-- Create table with JSONB
CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert JSON data
INSERT INTO user_preferences (user_id, settings) VALUES
(1, '{"theme": "dark", "notifications": {"email": true, "push": false}, "language": "en"}'),
(2, '{"theme": "light", "notifications": {"email": false, "push": true}, "language": "es"}');

-- Query JSON fields
SELECT 
    user_id,
    settings->>'theme' as theme,
    settings->'notifications'->>'email' as email_notifications
FROM user_preferences;

-- Query nested JSON
SELECT * FROM user_preferences
WHERE settings->'notifications'->>'email' = 'true';

-- Update JSON field
UPDATE user_preferences
SET settings = jsonb_set(settings, '{theme}', '"dark"')
WHERE user_id = 2;

-- Add new JSON field
UPDATE user_preferences
SET settings = settings || '{"timezone": "UTC"}'::jsonb
WHERE user_id = 1;

-- Remove JSON field
UPDATE user_preferences
SET settings = settings - 'timezone'
WHERE user_id = 1;

-- Query JSON array
CREATE TABLE products_with_tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    tags JSONB
);

INSERT INTO products_with_tags (name, tags) VALUES
('Product A', '["electronics", "gaming", "featured"]'),
('Product B', '["clothing", "sale"]');

-- Query if array contains element
SELECT * FROM products_with_tags
WHERE tags ? 'gaming';

-- Query if array contains any of elements
SELECT * FROM products_with_tags
WHERE tags ?| array['gaming', 'sale'];

-- Create GIN index for JSON
CREATE INDEX idx_user_preferences_settings 
ON user_preferences USING GIN(settings);

-- Aggregate JSON
SELECT 
    jsonb_agg(jsonb_build_object(
        'id', id,
        'name', name,
        'price', price
    )) as products
FROM products
WHERE category_id = 1;
```

### Example 9: Triggers and Functions

```sql
-- Audit log trigger
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    record_id INTEGER NOT NULL,
    action VARCHAR(10) NOT NULL,
    old_data JSONB,
    new_data JSONB,
    changed_by VARCHAR(50),
    changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, action, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'INSERT', row_to_json(NEW)::jsonb, current_user);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE', row_to_json(OLD)::jsonb, row_to_json(NEW)::jsonb, current_user);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, changed_by)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', row_to_json(OLD)::jsonb, current_user);
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Apply trigger to tables
CREATE TRIGGER products_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON products
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER users_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON users
FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

-- Auto-update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_products_updated_at
BEFORE UPDATE ON products
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Inventory management trigger
CREATE OR REPLACE FUNCTION update_product_inventory()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- Reduce inventory when order is created
        UPDATE products
        SET quantity = quantity - NEW.quantity
        WHERE id = NEW.product_id;
        
        IF NOT FOUND OR (SELECT quantity FROM products WHERE id = NEW.product_id) < 0 THEN
            RAISE EXCEPTION 'Insufficient inventory for product %', NEW.product_id;
        END IF;
    ELSIF TG_OP = 'DELETE' THEN
        -- Restore inventory when order is cancelled
        UPDATE products
        SET quantity = quantity + OLD.quantity
        WHERE id = OLD.product_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER order_items_inventory_trigger
AFTER INSERT OR DELETE ON order_items
FOR EACH ROW EXECUTE FUNCTION update_product_inventory();
```

### Example 10: Connection Pooling with PgBouncer

**pgbouncer.ini:**
```ini
[databases]
mydb = host=localhost port=5432 dbname=mydb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 10
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 100
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
```

**userlist.txt:**
```
"myuser" "md5_hashed_password"
```

**Start PgBouncer:**
```bash
pgbouncer -d /etc/pgbouncer/pgbouncer.ini
```

**Connect through PgBouncer:**
```bash
psql -h localhost -p 6432 -U myuser mydb
```

## Troubleshooting

### Common Issues

**Connection refused:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check if listening on correct port
sudo netstat -plnt | grep 5432

# Check pg_hba.conf for connection rules
sudo nano /etc/postgresql/15/main/pg_hba.conf
```

**Slow queries:**
```sql
-- Enable query logging
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1s

-- Find slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY correlation;
```

**High disk usage:**
```bash
# Check database sizes
SELECT pg_database.datname, pg_size_pretty(pg_database_size(pg_database.datname))
FROM pg_database;

# Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename))
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

# Vacuum to reclaim space
VACUUM FULL;
```

## Resources

- **Official Documentation**: https://www.postgresql.org/docs/
- **PostgreSQL Tutorial**: https://www.postgresqltutorial.com/
- **PostgreSQL Wiki**: https://wiki.postgresql.org/
- **psql Guide**: https://www.postgresql.org/docs/current/app-psql.html

## Quick Reference

### Data Types

| Type | Description | Example |
|------|-------------|---------|
| `INTEGER` | Whole numbers | 42 |
| `SERIAL` | Auto-increment integer | 1, 2, 3... |
| `BIGINT` | Large integers | 9223372036854775807 |
| `DECIMAL(p,s)` | Exact decimal | 123.45 |
| `REAL` | Floating point | 3.14159 |
| `VARCHAR(n)` | Variable length string | 'Hello' |
| `TEXT` | Unlimited string | 'Long text...' |
| `BOOLEAN` | True/false | TRUE |
| `DATE` | Date | '2024-01-15' |
| `TIMESTAMP` | Date and time | '2024-01-15 10:30:00' |
| `JSON/JSONB` | JSON data | '{"key": "value"}' |
| `ARRAY` | Array | '{1,2,3}' |
| `UUID` | Unique identifier | 'a0eebc99-9c0b-4ef8...' |

### psql Quick Commands

```bash
\l                  # List databases
\c dbname           # Connect to database
\dt                 # List tables
\d tablename        # Describe table
\du                 # List users
\dn                 # List schemas
\df                 # List functions
\dv                 # List views
\di                 # List indexes
\x                  # Toggle expanded output
\timing             # Toggle timing
\q                  # Quit
```

---

*This guide covers PostgreSQL fundamentals and advanced features. Practice with sample data to build your database skills.*
