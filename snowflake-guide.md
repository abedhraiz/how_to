# Snowflake Guide

## What is Snowflake?

Snowflake is a cloud-based data warehouse platform that provides data storage, processing, and analytics services. It's built for the cloud and offers unique features like instant elasticity, data sharing, and separation of storage and compute.

## Prerequisites

- Snowflake account (trial or paid)
- Basic understanding of SQL
- Understanding of data warehousing concepts
- Web browser (for Snowflake UI)
- Optional: Python, JDBC/ODBC drivers for programmatic access

## Getting Started

### Create a Snowflake Account

1. Visit https://signup.snowflake.com/
2. Choose cloud provider (AWS, Azure, or GCP)
3. Select region
4. Sign up for free trial (30 days, $400 credit)
5. Verify email and set password
6. Access Snowflake UI at `https://<account_name>.snowflakecomputing.com`

### First Login

```sql
-- Default credentials (change after first login)
Username: <your_email>
Password: <your_password>

-- First steps after login
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE SNOWFLAKE_SAMPLE_DATA;
USE SCHEMA TPCH_SF1;

-- Test query
SELECT * FROM CUSTOMER LIMIT 10;
```

## Core Concepts

### 1. **Virtual Warehouse**
Compute resources that execute queries. Can be started, stopped, and resized independently.

### 2. **Database**
Container for schemas and database objects.

### 3. **Schema**
Container for tables, views, and other database objects.

### 4. **Table**
Storage for structured data (permanent, temporary, transient, external).

### 5. **Stage**
Location for data files (internal or external) used for loading/unloading.

### 6. **File Format**
Definition of data file structure (CSV, JSON, Parquet, etc.).

### 7. **Role**
Set of privileges that can be granted to users.

### 8. **User**
Identity that can authenticate and access Snowflake.

## Snowflake Architecture

```
┌─────────────────────────────────────────┐
│         Cloud Services Layer            │
│  (Metadata, Security, Optimization)     │
└─────────────────────────────────────────┘
           ↓              ↓
┌──────────────────┐  ┌──────────────────┐
│  Virtual WH 1    │  │  Virtual WH 2    │
│  (Compute)       │  │  (Compute)       │
└──────────────────┘  └──────────────────┘
           ↓              ↓
┌─────────────────────────────────────────┐
│         Storage Layer                    │
│  (Centralized, Scalable, Durable)       │
└─────────────────────────────────────────┘
```

## Basic Operations

### Database Management

```sql
-- Create database
CREATE DATABASE my_database;

-- Use database
USE DATABASE my_database;

-- Show databases
SHOW DATABASES;

-- Describe database
DESCRIBE DATABASE my_database;

-- Drop database
DROP DATABASE my_database;

-- Clone database
CREATE DATABASE my_database_clone CLONE my_database;
```

### Schema Management

```sql
-- Create schema
CREATE SCHEMA my_schema;

-- Use schema
USE SCHEMA my_schema;

-- Show schemas
SHOW SCHEMAS IN DATABASE my_database;

-- Drop schema
DROP SCHEMA my_schema;

-- Create schema with options
CREATE SCHEMA my_schema
  WITH MANAGED ACCESS
  DATA_RETENTION_TIME_IN_DAYS = 7;
```

### Warehouse Management

```sql
-- Create warehouse
CREATE WAREHOUSE my_warehouse
  WITH WAREHOUSE_SIZE = 'MEDIUM'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE;

-- Show warehouses
SHOW WAREHOUSES;

-- Use warehouse
USE WAREHOUSE my_warehouse;

-- Alter warehouse size
ALTER WAREHOUSE my_warehouse SET WAREHOUSE_SIZE = 'LARGE';

-- Suspend warehouse
ALTER WAREHOUSE my_warehouse SUSPEND;

-- Resume warehouse
ALTER WAREHOUSE my_warehouse RESUME;

-- Drop warehouse
DROP WAREHOUSE my_warehouse;
```

### Warehouse Sizes

| Size | Credits/Hour | Typical Use Case |
|------|--------------|------------------|
| X-SMALL | 1 | Development, testing |
| SMALL | 2 | Light queries |
| MEDIUM | 4 | Standard workloads |
| LARGE | 8 | Heavy queries |
| X-LARGE | 16 | Complex analytics |
| 2X-LARGE | 32 | Very large datasets |
| 3X-LARGE | 64 | Massive computations |
| 4X-LARGE | 128 | Extreme workloads |

## Table Management

### Create Tables

```sql
-- Create permanent table
CREATE TABLE customers (
    customer_id NUMBER(38,0),
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    address VARCHAR(200),
    city VARCHAR(50),
    state VARCHAR(2),
    zip_code VARCHAR(10),
    created_date DATE,
    updated_timestamp TIMESTAMP_NTZ
);

-- Create table with constraints
CREATE TABLE orders (
    order_id NUMBER PRIMARY KEY,
    customer_id NUMBER NOT NULL,
    order_date DATE NOT NULL,
    total_amount NUMBER(10,2) DEFAULT 0.00,
    status VARCHAR(20) DEFAULT 'PENDING',
    CONSTRAINT fk_customer FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Create table from query (CTAS)
CREATE TABLE high_value_customers AS
SELECT 
    customer_id,
    first_name,
    last_name,
    SUM(total_amount) as lifetime_value
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
GROUP BY customer_id, first_name, last_name
HAVING SUM(total_amount) > 10000;

-- Create temporary table
CREATE TEMPORARY TABLE temp_results (
    id NUMBER,
    value VARCHAR
);

-- Create transient table (no fail-safe, lower cost)
CREATE TRANSIENT TABLE staging_data (
    id NUMBER,
    data VARIANT
);

-- Create external table
CREATE EXTERNAL TABLE external_customers
  WITH LOCATION = @my_s3_stage/customers/
  FILE_FORMAT = (TYPE = PARQUET)
  PATTERN = '.*customers.*[.]parquet';
```

### Alter Tables

```sql
-- Add column
ALTER TABLE customers ADD COLUMN loyalty_points NUMBER DEFAULT 0;

-- Drop column
ALTER TABLE customers DROP COLUMN loyalty_points;

-- Rename column
ALTER TABLE customers RENAME COLUMN phone TO phone_number;

-- Change column type
ALTER TABLE customers MODIFY COLUMN zip_code VARCHAR(15);

-- Rename table
ALTER TABLE customers RENAME TO customer_master;

-- Add constraint
ALTER TABLE orders ADD CONSTRAINT chk_amount CHECK (total_amount >= 0);

-- Drop constraint
ALTER TABLE orders DROP CONSTRAINT chk_amount;
```

### Table Types Comparison

| Type | Fail-Safe | Time Travel | Use Case |
|------|-----------|-------------|----------|
| Permanent | 7 days | 0-90 days | Production data |
| Transient | None | 0-1 day | Staging, temp data |
| Temporary | None | 0-1 day | Session data |
| External | None | None | Data lake integration |

## Loading Data

### Internal Stages

```sql
-- Create internal stage
CREATE STAGE my_stage;

-- Show stages
SHOW STAGES;

-- List files in stage
LIST @my_stage;

-- Put file into stage (SnowSQL)
PUT file:///path/to/local/file.csv @my_stage;

-- Remove file from stage
REMOVE @my_stage/file.csv;
```

### External Stages (S3 Example)

```sql
-- Create external stage (S3)
CREATE STAGE my_s3_stage
  URL = 's3://my-bucket/data/'
  CREDENTIALS = (AWS_KEY_ID = 'xxx' AWS_SECRET_KEY = 'yyy')
  FILE_FORMAT = (TYPE = CSV FIELD_DELIMITER = ',' SKIP_HEADER = 1);

-- Create external stage (Azure)
CREATE STAGE my_azure_stage
  URL = 'azure://myaccount.blob.core.windows.net/mycontainer/path/'
  CREDENTIALS = (AZURE_SAS_TOKEN = 'xxx');

-- Create external stage (GCS)
CREATE STAGE my_gcs_stage
  URL = 'gcs://my-bucket/path/'
  CREDENTIALS = (GCS_SERVICE_ACCOUNT = 'xxx');
```

### File Formats

```sql
-- Create CSV file format
CREATE FILE FORMAT my_csv_format
  TYPE = CSV
  FIELD_DELIMITER = ','
  SKIP_HEADER = 1
  NULL_IF = ('NULL', 'null', '')
  FIELD_OPTIONALLY_ENCLOSED_BY = '"'
  COMPRESSION = GZIP;

-- Create JSON file format
CREATE FILE FORMAT my_json_format
  TYPE = JSON
  COMPRESSION = AUTO;

-- Create Parquet file format
CREATE FILE FORMAT my_parquet_format
  TYPE = PARQUET
  COMPRESSION = SNAPPY;

-- Show file formats
SHOW FILE FORMATS;
```

### COPY INTO Command

```sql
-- Load from stage into table
COPY INTO customers
FROM @my_stage/customers.csv
FILE_FORMAT = (FORMAT_NAME = my_csv_format)
ON_ERROR = 'CONTINUE';

-- Load with pattern matching
COPY INTO orders
FROM @my_s3_stage
FILE_FORMAT = (TYPE = CSV)
PATTERN = '.*orders_2024.*[.]csv'
ON_ERROR = 'SKIP_FILE';

-- Load JSON data
COPY INTO json_table
FROM @my_stage/data.json
FILE_FORMAT = (TYPE = JSON)
MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- Load with transformation
COPY INTO customers (customer_id, full_name, email)
FROM (
    SELECT 
        $1::NUMBER as customer_id,
        CONCAT($2, ' ', $3) as full_name,
        $4::VARCHAR as email
    FROM @my_stage/data.csv
)
FILE_FORMAT = (TYPE = CSV);

-- Validate staged files before loading
COPY INTO customers
FROM @my_stage/customers.csv
FILE_FORMAT = (FORMAT_NAME = my_csv_format)
VALIDATION_MODE = 'RETURN_ERRORS';

-- Check copy history
SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
    TABLE_NAME => 'CUSTOMERS',
    START_TIME => DATEADD(hours, -1, CURRENT_TIMESTAMP())
));
```

### Snowpipe (Continuous Loading)

```sql
-- Create pipe for auto-ingestion
CREATE PIPE customer_pipe
  AUTO_INGEST = TRUE
  AWS_SNS_TOPIC = 'arn:aws:sns:us-east-1:xxx:snowpipe'
AS
  COPY INTO customers
  FROM @my_s3_stage
  FILE_FORMAT = (FORMAT_NAME = my_csv_format);

-- Show pipes
SHOW PIPES;

-- Check pipe status
SELECT SYSTEM$PIPE_STATUS('customer_pipe');

-- Refresh pipe (manual)
ALTER PIPE customer_pipe REFRESH;

-- Pause pipe
ALTER PIPE customer_pipe SET PIPE_EXECUTION_PAUSED = TRUE;

-- Resume pipe
ALTER PIPE customer_pipe SET PIPE_EXECUTION_PAUSED = FALSE;
```

## Querying Data

### Basic Queries

```sql
-- Simple SELECT
SELECT * FROM customers LIMIT 10;

-- SELECT with WHERE
SELECT first_name, last_name, email
FROM customers
WHERE state = 'CA'
  AND created_date >= '2024-01-01';

-- Aggregations
SELECT 
    state,
    COUNT(*) as customer_count,
    AVG(lifetime_value) as avg_value
FROM customers
GROUP BY state
ORDER BY customer_count DESC;

-- JOIN operations
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) as total_orders,
    SUM(o.total_amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING COUNT(o.order_id) > 5;

-- Window functions
SELECT 
    customer_id,
    order_date,
    total_amount,
    SUM(total_amount) OVER (
        PARTITION BY customer_id 
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) as running_total
FROM orders;

-- Common Table Expressions (CTE)
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        SUM(total_amount) as monthly_revenue
    FROM orders
    GROUP BY month
)
SELECT 
    month,
    monthly_revenue,
    LAG(monthly_revenue) OVER (ORDER BY month) as prev_month,
    monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY month) as growth
FROM monthly_sales
ORDER BY month;
```

### Semi-Structured Data

```sql
-- Create table for JSON data
CREATE TABLE json_data (
    id NUMBER,
    raw_data VARIANT
);

-- Insert JSON
INSERT INTO json_data
SELECT 
    1,
    PARSE_JSON('{"name": "John", "age": 30, "city": "NYC"}');

-- Query JSON data
SELECT 
    raw_data:name::STRING as name,
    raw_data:age::NUMBER as age,
    raw_data:city::STRING as city
FROM json_data;

-- Query nested JSON
SELECT 
    raw_data:user.name::STRING as name,
    raw_data:user.email::STRING as email,
    value:product::STRING as product
FROM json_data,
LATERAL FLATTEN(input => raw_data:orders);

-- Array operations
SELECT 
    raw_data:items[0]::STRING as first_item,
    ARRAY_SIZE(raw_data:items) as item_count
FROM json_data;
```

## Time Travel

```sql
-- Query historical data (by time)
SELECT * FROM customers
AT(TIMESTAMP => '2024-01-01 00:00:00'::TIMESTAMP);

-- Query by offset
SELECT * FROM customers
AT(OFFSET => -3600);  -- 1 hour ago

-- Query before statement
SELECT * FROM customers
BEFORE(STATEMENT => '01a12b34-5678-90cd-ef12-34567890abcd');

-- Restore dropped table
UNDROP TABLE customers;

-- Clone table at specific time
CREATE TABLE customers_backup CLONE customers
AT(TIMESTAMP => '2024-01-01 00:00:00'::TIMESTAMP);

-- Show retention period
SHOW PARAMETERS LIKE 'DATA_RETENTION_TIME_IN_DAYS' IN TABLE customers;

-- Set retention period
ALTER TABLE customers SET DATA_RETENTION_TIME_IN_DAYS = 30;
```

## Zero-Copy Cloning

```sql
-- Clone table (instant, no storage cost initially)
CREATE TABLE customers_dev CLONE customers;

-- Clone schema
CREATE SCHEMA dev_schema CLONE prod_schema;

-- Clone database
CREATE DATABASE dev_database CLONE prod_database;

-- Clone with time travel
CREATE TABLE customers_snapshot CLONE customers
AT(TIMESTAMP => '2024-01-01 00:00:00'::TIMESTAMP);

-- Clone from share
CREATE DATABASE shared_db FROM SHARE provider_account.share_name;
```

## Streams and Tasks

### Streams (Change Data Capture)

```sql
-- Create stream on table
CREATE STREAM customer_stream ON TABLE customers;

-- Show streams
SHOW STREAMS;

-- Query stream (shows changes)
SELECT * FROM customer_stream;

-- Columns in stream
-- METADATA$ACTION: INSERT, DELETE
-- METADATA$ISUPDATE: TRUE if UPDATE
-- METADATA$ROW_ID: Unique row identifier

-- Process stream data
BEGIN TRANSACTION;
  
  INSERT INTO customer_history
  SELECT * FROM customer_stream
  WHERE METADATA$ACTION = 'INSERT';
  
  UPDATE customer_summary
  SET update_count = update_count + 1
  FROM customer_stream
  WHERE customer_summary.id = customer_stream.customer_id
    AND customer_stream.METADATA$ACTION = 'UPDATE';

COMMIT;

-- Stream is automatically advanced after consumption
```

### Tasks (Scheduled Jobs)

```sql
-- Create simple task
CREATE TASK daily_aggregation
  WAREHOUSE = my_warehouse
  SCHEDULE = '1440 MINUTE'  -- Daily
AS
  INSERT INTO daily_summary
  SELECT 
    CURRENT_DATE() as report_date,
    COUNT(*) as order_count,
    SUM(total_amount) as total_revenue
  FROM orders
  WHERE order_date = CURRENT_DATE();

-- Create task with cron
CREATE TASK hourly_task
  WAREHOUSE = my_warehouse
  SCHEDULE = 'USING CRON 0 * * * * America/New_York'
AS
  CALL my_stored_procedure();

-- Create task tree (with dependencies)
CREATE TASK parent_task
  WAREHOUSE = my_warehouse
  SCHEDULE = '60 MINUTE'
AS
  INSERT INTO staging_table SELECT * FROM source_table;

CREATE TASK child_task
  WAREHOUSE = my_warehouse
  AFTER parent_task
AS
  INSERT INTO final_table SELECT * FROM staging_table;

-- Show tasks
SHOW TASKS;

-- Describe task
DESCRIBE TASK daily_aggregation;

-- Resume task (tasks are suspended by default)
ALTER TASK daily_aggregation RESUME;
ALTER TASK child_task RESUME;  -- Resume child first
ALTER TASK parent_task RESUME;  -- Then parent

-- Suspend task
ALTER TASK daily_aggregation SUSPEND;

-- Execute task manually
EXECUTE TASK daily_aggregation;

-- View task history
SELECT *
FROM TABLE(INFORMATION_SCHEMA.TASK_HISTORY(
    TASK_NAME => 'DAILY_AGGREGATION',
    SCHEDULED_TIME_RANGE_START => DATEADD('day', -7, CURRENT_TIMESTAMP())
));

-- Drop task
DROP TASK daily_aggregation;
```

## Views and Materialized Views

### Standard Views

```sql
-- Create view
CREATE VIEW customer_summary AS
SELECT 
    c.customer_id,
    c.first_name,
    c.last_name,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.first_name, c.last_name;

-- Create secure view (hides definition)
CREATE SECURE VIEW sensitive_customer_data AS
SELECT 
    customer_id,
    first_name,
    last_name,
    MASK(email, '*', 5) as masked_email
FROM customers;

-- Replace view
CREATE OR REPLACE VIEW customer_summary AS
SELECT * FROM customers WHERE status = 'ACTIVE';
```

### Materialized Views

```sql
-- Create materialized view (automatically maintained)
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT 
    DATE_TRUNC('day', order_date) as order_day,
    COUNT(*) as order_count,
    SUM(total_amount) as daily_revenue
FROM orders
GROUP BY order_day;

-- Show materialized views
SHOW MATERIALIZED VIEWS;

-- Refresh materialized view (usually automatic)
ALTER MATERIALIZED VIEW mv_daily_sales SUSPEND;
ALTER MATERIALIZED VIEW mv_daily_sales RESUME;

-- Drop materialized view
DROP MATERIALIZED VIEW mv_daily_sales;
```

## User Defined Functions (UDFs)

### SQL UDFs

```sql
-- Create SQL UDF
CREATE FUNCTION calculate_tax(amount FLOAT)
RETURNS FLOAT
AS
$$
    amount * 0.08
$$;

-- Use UDF
SELECT 
    order_id,
    total_amount,
    calculate_tax(total_amount) as tax,
    total_amount + calculate_tax(total_amount) as total_with_tax
FROM orders;

-- UDF with multiple statements
CREATE FUNCTION categorize_customer(order_count NUMBER)
RETURNS VARCHAR
AS
$$
    CASE
        WHEN order_count > 50 THEN 'VIP'
        WHEN order_count > 20 THEN 'Premium'
        WHEN order_count > 5 THEN 'Regular'
        ELSE 'New'
    END
$$;
```

### JavaScript UDFs

```sql
-- Create JavaScript UDF
CREATE FUNCTION parse_json_string(json_str VARCHAR)
RETURNS VARIANT
LANGUAGE JAVASCRIPT
AS
$$
    return JSON.parse(JSON_STR);
$$;

-- More complex JavaScript UDF
CREATE FUNCTION calculate_compound_interest(
    principal FLOAT,
    rate FLOAT,
    time FLOAT
)
RETURNS FLOAT
LANGUAGE JAVASCRIPT
AS
$$
    var amount = PRINCIPAL * Math.pow(1 + RATE, TIME);
    return amount;
$$;
```

### Python UDFs

```sql
-- Create Python UDF (requires Python UDF feature)
CREATE FUNCTION sentiment_score(text VARCHAR)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
PACKAGES = ('textblob')
HANDLER = 'analyze'
AS
$$
from textblob import TextBlob

def analyze(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
$$;
```

## Stored Procedures

```sql
-- Create stored procedure
CREATE PROCEDURE process_orders()
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    -- Create temp table
    CREATE OR REPLACE TEMPORARY TABLE temp_orders AS
    SELECT * FROM orders WHERE status = 'PENDING';
    
    -- Update orders
    UPDATE orders
    SET status = 'PROCESSING'
    WHERE order_id IN (SELECT order_id FROM temp_orders);
    
    -- Log the change
    INSERT INTO order_log (log_date, orders_processed)
    SELECT CURRENT_TIMESTAMP(), COUNT(*) FROM temp_orders;
    
    RETURN 'Processed ' || (SELECT COUNT(*) FROM temp_orders) || ' orders';
END;
$$;

-- Call procedure
CALL process_orders();

-- Procedure with parameters
CREATE PROCEDURE update_customer_status(
    customer_id_param NUMBER,
    new_status VARCHAR
)
RETURNS STRING
LANGUAGE SQL
AS
$$
BEGIN
    UPDATE customers
    SET status = :new_status,
        updated_timestamp = CURRENT_TIMESTAMP()
    WHERE customer_id = :customer_id_param;
    
    RETURN 'Updated customer ' || :customer_id_param;
END;
$$;

-- Call with parameters
CALL update_customer_status(123, 'ACTIVE');
```

## Data Sharing

### Creating a Share (Data Provider)

```sql
-- Create share
CREATE SHARE customer_share;

-- Grant usage on database
GRANT USAGE ON DATABASE my_database TO SHARE customer_share;

-- Grant usage on schema
GRANT USAGE ON SCHEMA my_database.public TO SHARE customer_share;

-- Grant select on table
GRANT SELECT ON TABLE my_database.public.customers TO SHARE customer_share;

-- Add account to share
ALTER SHARE customer_share ADD ACCOUNTS = account1, account2;

-- Show shares
SHOW SHARES;

-- Describe share
DESCRIBE SHARE customer_share;

-- Revoke access
ALTER SHARE customer_share REMOVE ACCOUNTS = account1;

-- Drop share
DROP SHARE customer_share;
```

### Consuming a Share (Data Consumer)

```sql
-- Show available shares
SHOW SHARES;

-- Create database from share
CREATE DATABASE shared_customer_data
FROM SHARE provider_account.customer_share;

-- Query shared data
SELECT * FROM shared_customer_data.public.customers LIMIT 10;

-- Create views on shared data
CREATE VIEW my_customer_view AS
SELECT * FROM shared_customer_data.public.customers
WHERE state = 'CA';
```

## Security and Access Control

### Role Management

```sql
-- Create role
CREATE ROLE analyst;

-- Grant privileges to role
GRANT USAGE ON WAREHOUSE my_warehouse TO ROLE analyst;
GRANT USAGE ON DATABASE my_database TO ROLE analyst;
GRANT USAGE ON SCHEMA my_database.public TO ROLE analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA my_database.public TO ROLE analyst;

-- Create role hierarchy
GRANT ROLE analyst TO ROLE manager;

-- Assign role to user
GRANT ROLE analyst TO USER john_doe;

-- Show grants
SHOW GRANTS TO ROLE analyst;
SHOW GRANTS ON TABLE customers;

-- Revoke privileges
REVOKE SELECT ON TABLE customers FROM ROLE analyst;
```

### User Management

```sql
-- Create user
CREATE USER john_doe
  PASSWORD = 'SecurePass123!'
  DEFAULT_ROLE = analyst
  DEFAULT_WAREHOUSE = my_warehouse
  DEFAULT_NAMESPACE = my_database.public
  MUST_CHANGE_PASSWORD = TRUE;

-- Alter user
ALTER USER john_doe SET PASSWORD = 'NewPass123!';
ALTER USER john_doe SET DEFAULT_ROLE = manager;

-- Show users
SHOW USERS;

-- Drop user
DROP USER john_doe;
```

### Row-Level Security

```sql
-- Create row access policy
CREATE ROW ACCESS POLICY customer_region_policy
AS (region VARCHAR) RETURNS BOOLEAN ->
    CASE
        WHEN CURRENT_ROLE() = 'ADMIN' THEN TRUE
        WHEN CURRENT_ROLE() = 'WEST_REGION' AND region = 'WEST' THEN TRUE
        WHEN CURRENT_ROLE() = 'EAST_REGION' AND region = 'EAST' THEN TRUE
        ELSE FALSE
    END;

-- Apply policy to table
ALTER TABLE customers
  ADD ROW ACCESS POLICY customer_region_policy ON (state);

-- Show policies
SHOW ROW ACCESS POLICIES;

-- Remove policy
ALTER TABLE customers DROP ROW ACCESS POLICY customer_region_policy;
```

### Column-Level Security (Masking)

```sql
-- Create masking policy
CREATE MASKING POLICY email_mask AS (val STRING) RETURNS STRING ->
    CASE
        WHEN CURRENT_ROLE() IN ('ADMIN', 'ANALYST') THEN val
        ELSE CONCAT(LEFT(val, 2), '****', RIGHT(val, 4))
    END;

-- Apply policy to column
ALTER TABLE customers MODIFY COLUMN email
  SET MASKING POLICY email_mask;

-- Show masking policies
SHOW MASKING POLICIES;

-- Remove policy
ALTER TABLE customers MODIFY COLUMN email
  UNSET MASKING POLICY;
```

## Working with Snowflake via Python

### Installation

```bash
pip install snowflake-connector-python
pip install snowflake-sqlalchemy  # For SQLAlchemy support
```

### Basic Connection

```python
import snowflake.connector

# Connect to Snowflake
conn = snowflake.connector.connect(
    user='your_username',
    password='your_password',
    account='your_account',
    warehouse='my_warehouse',
    database='my_database',
    schema='public'
)

# Create cursor
cur = conn.cursor()

# Execute query
cur.execute("SELECT * FROM customers LIMIT 10")

# Fetch results
results = cur.fetchall()
for row in results:
    print(row)

# Close connection
cur.close()
conn.close()
```

### Using Context Manager

```python
from snowflake.connector import connect

with connect(
    user='your_username',
    password='your_password',
    account='your_account',
    warehouse='my_warehouse',
    database='my_database',
    schema='public'
) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM customers")
        count = cur.fetchone()[0]
        print(f"Total customers: {count}")
```

### Pandas Integration

```python
import pandas as pd
from snowflake.connector.pandas_tools import write_pandas

# Read data into pandas
query = "SELECT * FROM customers WHERE state = 'CA'"
df = pd.read_sql(query, conn)

# Write pandas DataFrame to Snowflake
success, nchunks, nrows, _ = write_pandas(
    conn=conn,
    df=df,
    table_name='CUSTOMERS_CA',
    database='MY_DATABASE',
    schema='PUBLIC'
)
```

### Using SQLAlchemy

```python
from sqlalchemy import create_engine
import pandas as pd

# Create engine
engine = create_engine(
    'snowflake://{user}:{password}@{account}/{database}/{schema}?warehouse={warehouse}'.format(
        user='your_username',
        password='your_password',
        account='your_account',
        database='my_database',
        schema='public',
        warehouse='my_warehouse'
    )
)

# Query with pandas
df = pd.read_sql_query("SELECT * FROM customers LIMIT 10", engine)

# Write DataFrame
df.to_sql('new_table', engine, if_exists='replace', index=False)
```

## Performance Optimization

### Clustering

```sql
-- Add clustering key
ALTER TABLE large_table CLUSTER BY (date_column, category);

-- Show clustering information
SELECT SYSTEM$CLUSTERING_INFORMATION('large_table');

-- Re-cluster table (usually automatic)
ALTER TABLE large_table RECLUSTER;

-- Drop clustering
ALTER TABLE large_table DROP CLUSTERING KEY;
```

### Query Profiling

```sql
-- Enable query profiling
ALTER SESSION SET USE_CACHED_RESULT = FALSE;

-- View query history
SELECT *
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(
    DATEADD('hours', -1, CURRENT_TIMESTAMP()),
    CURRENT_TIMESTAMP()
))
WHERE EXECUTION_STATUS = 'SUCCESS'
ORDER BY TOTAL_ELAPSED_TIME DESC;

-- Get query profile
SELECT SYSTEM$QUERY_PROFILE('query_id');
```

### Result Caching

```sql
-- Disable result cache for session
ALTER SESSION SET USE_CACHED_RESULT = FALSE;

-- Enable result cache
ALTER SESSION SET USE_CACHED_RESULT = TRUE;
```

### Search Optimization

```sql
-- Add search optimization
ALTER TABLE customers ADD SEARCH OPTIMIZATION;

-- Show search optimization status
DESCRIBE SEARCH OPTIMIZATION ON customers;

-- Drop search optimization
ALTER TABLE customers DROP SEARCH OPTIMIZATION;
```

## Monitoring and Administration

### Resource Monitors

```sql
-- Create resource monitor
CREATE RESOURCE MONITOR monthly_limit
WITH CREDIT_QUOTA = 1000
TRIGGERS 
  ON 75 PERCENT DO NOTIFY
  ON 90 PERCENT DO SUSPEND
  ON 100 PERCENT DO SUSPEND_IMMEDIATE;

-- Assign to warehouse
ALTER WAREHOUSE my_warehouse SET RESOURCE_MONITOR = monthly_limit;

-- Show resource monitors
SHOW RESOURCE MONITORS;
```

### Monitoring Queries

```sql
-- Active queries
SELECT * FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
WHERE EXECUTION_STATUS = 'RUNNING'
ORDER BY START_TIME;

-- Long-running queries
SELECT 
    query_id,
    user_name,
    warehouse_name,
    execution_status,
    total_elapsed_time/1000 as elapsed_seconds,
    query_text
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
WHERE EXECUTION_STATUS = 'RUNNING'
  AND total_elapsed_time > 300000  -- > 5 minutes
ORDER BY total_elapsed_time DESC;

-- Warehouse usage
SELECT 
    warehouse_name,
    SUM(credits_used) as total_credits
FROM SNOWFLAKE.ACCOUNT_USAGE.WAREHOUSE_METERING_HISTORY
WHERE start_time >= DATEADD('day', -30, CURRENT_TIMESTAMP())
GROUP BY warehouse_name
ORDER BY total_credits DESC;

-- Storage usage
SELECT * FROM SNOWFLAKE.ACCOUNT_USAGE.STORAGE_USAGE
ORDER BY USAGE_DATE DESC
LIMIT 30;
```

## Best Practices

### 1. Warehouse Management
- Use appropriate warehouse sizes
- Enable auto-suspend (typically 5-10 minutes)
- Enable auto-resume
- Use multi-cluster warehouses for concurrent workloads
- Separate warehouses by workload type

### 2. Table Design
- Use clustering for large tables
- Consider transient tables for staging
- Set appropriate data retention periods
- Use external tables for data lake integration

### 3. Query Optimization
- Use LIMIT during development
- Avoid SELECT *
- Use clustering keys for large tables
- Leverage result caching
- Use appropriate data types

### 4. Data Loading
- Use COPY INTO for bulk loading
- Use Snowpipe for continuous loading
- Compress files before loading
- Use appropriate file formats (Parquet for structured data)
- Validate data before loading

### 5. Cost Management
- Monitor credit usage regularly
- Set up resource monitors
- Use appropriate warehouse sizes
- Suspend warehouses when not in use
- Use transient tables for temporary data

### 6. Security
- Use role-based access control
- Implement row-level security where needed
- Use column masking for sensitive data
- Enable MFA for users
- Rotate credentials regularly

### 7. Development
- Use separate warehouses for dev/test/prod
- Clone production data for testing
- Use version control for SQL scripts
- Document data models and processes

## SnowSQL CLI

### Installation and Setup

```bash
# Download from Snowflake
# https://docs.snowflake.com/en/user-guide/snowsql-install-config.html

# Configure connection
snowsql -a <account> -u <username>

# Or use config file (~/.snowsql/config)
[connections.myconnection]
accountname = myaccount
username = myuser
password = mypassword
dbname = mydatabase
schemaname = public
warehousename = mywarehouse
```

### Common SnowSQL Commands

```bash
# Connect
snowsql -c myconnection

# Execute query
snowsql -c myconnection -q "SELECT COUNT(*) FROM customers"

# Execute file
snowsql -c myconnection -f script.sql

# Output to file
snowsql -c myconnection -q "SELECT * FROM customers" -o output_file=results.csv

# Variables
snowsql -c myconnection -D my_var=value -q "SELECT * FROM &my_var"
```

## Common Issues and Solutions

### Issue: Warehouse Suspended
```sql
-- Resume warehouse
ALTER WAREHOUSE my_warehouse RESUME;
```

### Issue: Query Timeout
```sql
-- Increase statement timeout
ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = 7200;

-- Or use larger warehouse
ALTER WAREHOUSE my_warehouse SET WAREHOUSE_SIZE = 'LARGE';
```

### Issue: Permission Denied
```sql
-- Check current role
SELECT CURRENT_ROLE();

-- Switch role
USE ROLE ACCOUNTADMIN;

-- Grant required privileges
GRANT USAGE ON WAREHOUSE my_warehouse TO ROLE my_role;
```

### Issue: Data Not Loading
```sql
-- Check for errors
SELECT * FROM TABLE(VALIDATE(my_table, JOB_ID => '_last'));

-- Check copy history
SELECT * FROM TABLE(INFORMATION_SCHEMA.COPY_HISTORY(
    TABLE_NAME => 'MY_TABLE',
    START_TIME => DATEADD(hours, -1, CURRENT_TIMESTAMP())
));
```

## Useful Resources

- Official Documentation: https://docs.snowflake.com/
- Snowflake University: https://learn.snowflake.com/
- Community: https://community.snowflake.com/
- GitHub: https://github.com/snowflakedb
- Quick Starts: https://quickstarts.snowflake.com/

## Quick Reference

### Common Commands

| Command | Description |
|---------|-------------|
| `USE DATABASE db_name;` | Switch database |
| `USE WAREHOUSE wh_name;` | Switch warehouse |
| `SHOW TABLES;` | List tables |
| `DESCRIBE TABLE table_name;` | Table structure |
| `SELECT CURRENT_WAREHOUSE();` | Current warehouse |
| `SELECT CURRENT_DATABASE();` | Current database |
| `SELECT CURRENT_ROLE();` | Current role |
| `SHOW GRANTS TO ROLE role_name;` | Show role permissions |

### Data Types

| Type | Description | Example |
|------|-------------|---------|
| NUMBER | Fixed-point number | NUMBER(38,0) |
| FLOAT | Floating-point | FLOAT |
| VARCHAR | Variable-length string | VARCHAR(100) |
| BOOLEAN | True/False | BOOLEAN |
| DATE | Date | DATE |
| TIMESTAMP | Timestamp | TIMESTAMP_NTZ |
| VARIANT | Semi-structured | VARIANT |
| ARRAY | Array | ARRAY |
| OBJECT | Object | OBJECT |

---

*This guide covers Snowflake fundamentals. For production use, consider implementing proper security policies, cost monitoring, disaster recovery strategies, and integration with your data ecosystem.*
