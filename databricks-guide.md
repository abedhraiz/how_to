# Databricks Guide

## What is Databricks?

Databricks is a unified analytics platform built on Apache Spark that provides a collaborative environment for data engineering, data science, and machine learning. Founded by the creators of Apache Spark, it simplifies big data processing and machine learning workflows.

**Key Features:**
- Unified workspace for data teams
- Managed Apache Spark clusters
- Collaborative notebooks
- Delta Lake for reliable data lakes
- MLflow for ML lifecycle management
- Integration with major cloud providers
- Auto-scaling and optimization
- Built-in security and governance

## Prerequisites

- Basic understanding of SQL and Python
- Familiarity with distributed computing concepts
- Cloud platform account (AWS, Azure, or GCP)
- Understanding of data engineering principles
- Basic knowledge of Spark (helpful but not required)

## Core Concepts

### Workspace
Centralized environment for all Databricks resources including notebooks, libraries, clusters, and jobs.

### Clusters
Managed Spark compute resources that can be shared across multiple users and workloads.

### Notebooks
Interactive documents combining code, visualizations, and narrative text (supports Python, Scala, SQL, R).

### Jobs
Scheduled or on-demand execution of notebooks or JAR files.

### Delta Lake
Open-source storage layer providing ACID transactions, schema enforcement, and time travel.

### DBFS (Databricks File System)
Distributed file system mounted in Databricks workspaces.

### Unity Catalog
Unified governance solution for data and AI assets.

## Getting Started

### Sign Up for Databricks

1. **AWS**: Visit [databricks.com](https://databricks.com/)
2. **Azure**: Use Azure Databricks from Azure Portal
3. **GCP**: Use Databricks on Google Cloud Platform

```bash
# Community Edition (Free Tier)
# Visit: https://community.cloud.databricks.com/
# Sign up with email
# Limited resources but great for learning
```

### Workspace Setup

1. Log in to your Databricks workspace
2. Navigate to the workspace browser (left sidebar)
3. Create folders to organize your work
4. Set up version control integration (GitHub, GitLab, etc.)

## Clusters

### Create Cluster

**UI Method:**
1. Click "Compute" in left sidebar
2. Click "Create Cluster"
3. Configure:
   - Cluster name
   - Cluster mode (Standard, High Concurrency, Single Node)
   - Databricks Runtime version
   - Node type and number of workers
   - Auto-scaling settings
   - Auto-termination

**Python API:**
```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

cluster = w.clusters.create(
    cluster_name="my-cluster",
    spark_version="13.3.x-scala2.12",
    node_type_id="i3.xlarge",
    num_workers=2,
    autotermination_minutes=30,
    spark_conf={
        "spark.sql.adaptive.enabled": "true"
    }
)

print(f"Cluster ID: {cluster.cluster_id}")
```

### Cluster Types

**Standard Cluster:**
```python
# Best for single-user workloads
# Full access to Spark APIs
# Can run Python, Scala, R, SQL
```

**High Concurrency Cluster:**
```python
# Optimized for multiple concurrent users
# Shared compute resources
# Fine-grained resource allocation
# Table access control
```

**Single Node Cluster:**
```python
# No workers, driver only
# Lightweight for small workloads
# Development and testing
```

### Cluster Configuration

```python
# Cluster configuration example
{
    "cluster_name": "production-cluster",
    "spark_version": "13.3.x-scala2.12",
    "node_type_id": "i3.xlarge",
    "driver_node_type_id": "i3.xlarge",
    "num_workers": 2,
    "autoscale": {
        "min_workers": 2,
        "max_workers": 8
    },
    "autotermination_minutes": 30,
    "spark_conf": {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.databricks.delta.preview.enabled": "true"
    },
    "aws_attributes": {
        "availability": "SPOT_WITH_FALLBACK",
        "zone_id": "us-west-2a",
        "spot_bid_price_percent": 100,
        "ebs_volume_type": "GENERAL_PURPOSE_SSD",
        "ebs_volume_count": 1,
        "ebs_volume_size": 100
    }
}
```

## Notebooks

### Create Notebook

1. Click "Workspace" in left sidebar
2. Right-click folder → Create → Notebook
3. Name your notebook
4. Select default language (Python, Scala, SQL, R)
5. Attach to cluster

### Notebook Basics

```python
# Python cell
print("Hello Databricks!")

# Display dataframe
df = spark.range(10)
display(df)

# Magic commands
%python  # Switch to Python
%sql     # Switch to SQL
%scala   # Switch to Scala
%r       # Switch to R
%md      # Markdown cell
%sh      # Shell commands
%fs      # Databricks filesystem commands
%run     # Run another notebook
```

### Data Visualization

```python
# Create sample data
data = [
    ("Alice", 25, 50000),
    ("Bob", 30, 60000),
    ("Charlie", 35, 70000),
    ("Diana", 28, 55000)
]

df = spark.createDataFrame(data, ["name", "age", "salary"])

# Display with automatic visualization
display(df)

# Matplotlib visualization
import matplotlib.pyplot as plt

df_pandas = df.toPandas()
plt.figure(figsize=(10, 6))
plt.bar(df_pandas['name'], df_pandas['salary'])
plt.title('Salary by Name')
plt.xlabel('Name')
plt.ylabel('Salary')
plt.show()

# Plotly for interactive charts
import plotly.express as px

fig = px.scatter(df_pandas, x='age', y='salary', text='name', 
                 title='Age vs Salary')
fig.show()
```

### Widgets (Parameters)

```python
# Create text widget
dbutils.widgets.text("environment", "dev", "Environment")

# Create dropdown widget
dbutils.widgets.dropdown("region", "us-west-2", 
                        ["us-west-2", "us-east-1", "eu-west-1"])

# Create multiselect widget
dbutils.widgets.multiselect("datasets", "sales", 
                           ["sales", "customers", "products"])

# Get widget value
env = dbutils.widgets.get("environment")
print(f"Running in {env} environment")

# Remove widget
dbutils.widgets.remove("environment")

# Remove all widgets
dbutils.widgets.removeAll()
```

## Working with Data

### Read Data

```python
# Read CSV
df = spark.read.csv("/databricks-datasets/samples/population-vs-price/data_geo.csv", 
                    header=True, inferSchema=True)

# Read JSON
df = spark.read.json("dbfs:/path/to/file.json")

# Read Parquet
df = spark.read.parquet("dbfs:/path/to/file.parquet")

# Read Delta Lake
df = spark.read.format("delta").load("/delta/table")

# Read from S3
df = spark.read.parquet("s3://bucket/path/to/data")

# Read with options
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", ",") \
    .option("quote", '"') \
    .load("/path/to/file.csv")

# Read multiple files
df = spark.read.parquet("dbfs:/path/to/data/*.parquet")
```

### Write Data

```python
# Write CSV
df.write.csv("/path/to/output", header=True, mode="overwrite")

# Write Parquet
df.write.parquet("/path/to/output", mode="overwrite")

# Write Delta Lake
df.write.format("delta").mode("overwrite").save("/delta/table")

# Write to table
df.write.saveAsTable("database.table_name", mode="overwrite")

# Partitioned write
df.write.partitionBy("year", "month").parquet("/path/to/output")

# Write modes
df.write.mode("overwrite").parquet("/path")  # Overwrite
df.write.mode("append").parquet("/path")     # Append
df.write.mode("ignore").parquet("/path")     # Ignore if exists
df.write.mode("error").parquet("/path")      # Error if exists
```

### SQL Queries

```sql
-- Create database
CREATE DATABASE IF NOT EXISTS my_database;

-- Use database
USE my_database;

-- Create table
CREATE TABLE employees (
  id INT,
  name STRING,
  department STRING,
  salary DOUBLE,
  hire_date DATE
) USING DELTA;

-- Insert data
INSERT INTO employees VALUES
  (1, 'Alice', 'Engineering', 95000, '2020-01-15'),
  (2, 'Bob', 'Sales', 75000, '2019-06-20'),
  (3, 'Charlie', 'Engineering', 105000, '2018-03-10');

-- Query data
SELECT department, AVG(salary) as avg_salary
FROM employees
GROUP BY department
ORDER BY avg_salary DESC;

-- Create view
CREATE OR REPLACE VIEW high_earners AS
SELECT * FROM employees WHERE salary > 90000;

-- Describe table
DESCRIBE EXTENDED employees;

-- Show tables
SHOW TABLES;
```

### PySpark DataFrame Operations

```python
# Create DataFrame
data = [
    (1, "Alice", "Engineering", 95000),
    (2, "Bob", "Sales", 75000),
    (3, "Charlie", "Engineering", 105000),
    (4, "Diana", "Marketing", 85000)
]

df = spark.createDataFrame(data, ["id", "name", "department", "salary"])

# Select columns
df.select("name", "salary").show()

# Filter
df.filter(df.salary > 80000).show()
df.where(df.department == "Engineering").show()

# Group by and aggregate
df.groupBy("department").agg({"salary": "avg"}).show()

# Multiple aggregations
from pyspark.sql.functions import avg, max, min, count

df.groupBy("department").agg(
    avg("salary").alias("avg_salary"),
    max("salary").alias("max_salary"),
    min("salary").alias("min_salary"),
    count("*").alias("employee_count")
).show()

# Join DataFrames
departments = spark.createDataFrame([
    ("Engineering", "Building A"),
    ("Sales", "Building B"),
    ("Marketing", "Building C")
], ["department", "location"])

df.join(departments, "department", "left").show()

# Add column
from pyspark.sql.functions import col, when

df = df.withColumn("bonus", 
                   when(col("salary") > 90000, col("salary") * 0.1)
                   .otherwise(col("salary") * 0.05))

# Rename column
df = df.withColumnRenamed("name", "employee_name")

# Drop column
df = df.drop("id")

# Sort
df.orderBy("salary", ascending=False).show()

# Window functions
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank

window = Window.partitionBy("department").orderBy(col("salary").desc())
df.withColumn("rank", rank().over(window)).show()
```

## Delta Lake

### Create Delta Table

```python
# Write DataFrame as Delta table
df.write.format("delta").save("/delta/my_table")

# Create managed table
df.write.format("delta").saveAsTable("my_table")

# SQL
%sql
CREATE TABLE my_delta_table (
  id INT,
  name STRING,
  value DOUBLE
) USING DELTA
LOCATION '/delta/my_delta_table';
```

### Delta Operations

```python
from delta.tables import DeltaTable

# Read Delta table
df = spark.read.format("delta").load("/delta/my_table")

# Update
deltaTable = DeltaTable.forPath(spark, "/delta/my_table")
deltaTable.update(
    condition="id = 1",
    set={"name": "'Updated Name'"}
)

# Delete
deltaTable.delete("id < 10")

# Merge (Upsert)
deltaTable.alias("target").merge(
    source_df.alias("source"),
    "target.id = source.id"
).whenMatchedUpdate(set={
    "name": "source.name",
    "value": "source.value"
}).whenNotMatchedInsert(values={
    "id": "source.id",
    "name": "source.name",
    "value": "source.value"
}).execute()

# Optimize (compaction)
deltaTable.optimize().executeCompaction()

# Z-order optimization
deltaTable.optimize().executeZOrderBy("date", "country")

# Vacuum (remove old files)
deltaTable.vacuum(168)  # Remove files older than 7 days
```

### Time Travel

```python
# Read specific version
df = spark.read.format("delta") \
    .option("versionAsOf", 5) \
    .load("/delta/my_table")

# Read as of timestamp
df = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-01") \
    .load("/delta/my_table")

# SQL
%sql
SELECT * FROM my_table VERSION AS OF 5;
SELECT * FROM my_table TIMESTAMP AS OF '2024-01-01';

# View history
%sql
DESCRIBE HISTORY my_table;

# Restore to previous version
%sql
RESTORE TABLE my_table TO VERSION AS OF 5;
```

### Schema Evolution

```python
# Merge schema on write
df.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save("/delta/my_table")

# Overwrite schema
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save("/delta/my_table")

# SQL
%sql
ALTER TABLE my_table ADD COLUMNS (new_column STRING);
```

## Machine Learning with MLflow

### Setup MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Set experiment
mlflow.set_experiment("/Users/your-email@company.com/my-ml-experiment")

# Enable autologging
mlflow.sklearn.autolog()
```

### Training with MLflow

```python
# Load data
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow run
with mlflow.start_run(run_name="random_forest_model"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
```

### Model Registry

```python
# Register model
model_uri = f"runs:/{run.info.run_id}/model"
model_details = mlflow.register_model(model_uri, "iris_classifier")

# Transition model stage
from mlflow.tracking import MlflowClient

client = MlflowClient()
client.transition_model_version_stage(
    name="iris_classifier",
    version=1,
    stage="Production"
)

# Load model from registry
model = mlflow.pyfunc.load_model("models:/iris_classifier/Production")

# Make predictions
predictions = model.predict(X_test)
```

### Hyperparameter Tuning

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Set experiment
mlflow.set_experiment("/my-experiment")

# Define parameter search space
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

best_score = 0
best_params = None

# Grid search with MLflow tracking
for n_est in param_grid['n_estimators']:
    for max_d in param_grid['max_depth']:
        for min_split in param_grid['min_samples_split']:
            with mlflow.start_run(nested=True):
                # Log parameters
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", max_d)
                mlflow.log_param("min_samples_split", min_split)
                
                # Train and evaluate
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    random_state=42
                )
                
                scores = cross_val_score(model, X_train, y_train, cv=5)
                mean_score = np.mean(scores)
                
                # Log metrics
                mlflow.log_metric("cv_mean_score", mean_score)
                mlflow.log_metric("cv_std_score", np.std(scores))
                
                # Track best model
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split
                    }

print(f"Best score: {best_score:.4f}")
print(f"Best params: {best_params}")
```

## Databricks Utilities (dbutils)

### File System Commands

```python
# List files
dbutils.fs.ls("dbfs:/")
dbutils.fs.ls("/databricks-datasets/")

# Create directory
dbutils.fs.mkdirs("/tmp/my_directory")

# Copy files
dbutils.fs.cp("/source/path", "/destination/path", recurse=True)

# Move files
dbutils.fs.mv("/source/path", "/destination/path")

# Delete files
dbutils.fs.rm("/path/to/file", recurse=True)

# Read file
contents = dbutils.fs.head("/path/to/file.txt")
print(contents)

# Put file
dbutils.fs.put("/path/to/file.txt", "File contents here", overwrite=True)

# Mount external storage (S3)
dbutils.fs.mount(
    source="s3a://my-bucket",
    mount_point="/mnt/my-bucket",
    extra_configs={
        "fs.s3a.access.key": "YOUR_ACCESS_KEY",
        "fs.s3a.secret.key": "YOUR_SECRET_KEY"
    }
)

# Unmount
dbutils.fs.unmount("/mnt/my-bucket")

# List mounts
dbutils.fs.mounts()
```

### Secrets

```python
# Get secret from secret scope
api_key = dbutils.secrets.get(scope="my-scope", key="api-key")

# List secret scopes
dbutils.secrets.listScopes()

# List secrets in scope
dbutils.secrets.list("my-scope")
```

### Notebook Workflows

```python
# Run another notebook
result = dbutils.notebook.run(
    "/path/to/notebook",
    timeout_seconds=600,
    arguments={"param1": "value1", "param2": "value2"}
)

# Exit notebook with value
dbutils.notebook.exit("Success")
```

## Jobs and Workflows

### Create Job via UI

1. Click "Workflows" in left sidebar
2. Click "Create Job"
3. Configure:
   - Job name
   - Task type (Notebook, JAR, Python script)
   - Cluster configuration
   - Schedule (optional)
   - Parameters
   - Email alerts

### Create Job via API

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# Create job
job = w.jobs.create(
    name="my-data-pipeline",
    tasks=[
        {
            "task_key": "extract",
            "notebook_task": {
                "notebook_path": "/Users/user@company.com/extract_data",
                "base_parameters": {"date": "2024-01-01"}
            },
            "new_cluster": {
                "spark_version": "13.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 2
            }
        },
        {
            "task_key": "transform",
            "depends_on": [{"task_key": "extract"}],
            "notebook_task": {
                "notebook_path": "/Users/user@company.com/transform_data"
            },
            "existing_cluster_id": "cluster-id"
        },
        {
            "task_key": "load",
            "depends_on": [{"task_key": "transform"}],
            "notebook_task": {
                "notebook_path": "/Users/user@company.com/load_data"
            },
            "existing_cluster_id": "cluster-id"
        }
    ],
    schedule={
        "quartz_cron_expression": "0 0 2 * * ?",  # Daily at 2 AM
        "timezone_id": "America/Los_Angeles"
    },
    email_notifications={
        "on_failure": ["team@company.com"],
        "on_success": ["success@company.com"]
    }
)

print(f"Job created with ID: {job.job_id}")

# Run job
run = w.jobs.run_now(job_id=job.job_id)
print(f"Run ID: {run.run_id}")
```

### Multi-Task Workflow

```python
# Notebook 1: Extract
# Save data to temp location
df = spark.read.format("parquet").load("s3://source/data")
df.write.format("delta").mode("overwrite").save("/tmp/extracted_data")

# Notebook 2: Transform (depends on Notebook 1)
# Read from temp location
df = spark.read.format("delta").load("/tmp/extracted_data")
transformed_df = df.filter(df.value > 100).withColumn("processed_date", current_date())
transformed_df.write.format("delta").mode("overwrite").save("/tmp/transformed_data")

# Notebook 3: Load (depends on Notebook 2)
# Read and load to final location
df = spark.read.format("delta").load("/tmp/transformed_data")
df.write.format("delta").mode("overwrite").saveAsTable("production.final_table")
```

## Real-World Examples

### ETL Pipeline

```python
# Extract: Read from various sources
customers_df = spark.read.format("delta").load("/mnt/raw/customers")
orders_df = spark.read.format("delta").load("/mnt/raw/orders")
products_df = spark.read.format("delta").load("/mnt/raw/products")

# Transform: Clean and join data
from pyspark.sql.functions import col, when, lit, current_timestamp

# Clean customers
customers_clean = customers_df \
    .filter(col("email").isNotNull()) \
    .withColumn("full_name", 
                when(col("last_name").isNotNull(), 
                     col("first_name") + " " + col("last_name"))
                .otherwise(col("first_name"))) \
    .select("customer_id", "full_name", "email", "country")

# Enrich orders with customer and product info
orders_enriched = orders_df \
    .join(customers_clean, "customer_id", "left") \
    .join(products_df, "product_id", "left") \
    .withColumn("total_amount", col("quantity") * col("unit_price")) \
    .withColumn("processed_timestamp", current_timestamp()) \
    .select(
        "order_id", "customer_id", "full_name", "email",
        "product_id", "product_name", "quantity", 
        "unit_price", "total_amount", "order_date",
        "processed_timestamp"
    )

# Load: Write to Delta Lake with partitioning
orders_enriched.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("order_date") \
    .saveAsTable("gold.orders_enriched")

# Create aggregate view
%sql
CREATE OR REPLACE VIEW gold.customer_summary AS
SELECT 
    customer_id,
    full_name,
    email,
    COUNT(DISTINCT order_id) as total_orders,
    SUM(total_amount) as lifetime_value,
    MAX(order_date) as last_order_date
FROM gold.orders_enriched
GROUP BY customer_id, full_name, email;
```

### Streaming Pipeline

```python
from pyspark.sql.functions import from_json, col, window
from pyspark.sql.types import StructType, StringType, TimestampType, DoubleType

# Define schema
schema = StructType() \
    .add("event_id", StringType()) \
    .add("user_id", StringType()) \
    .add("event_type", StringType()) \
    .add("timestamp", TimestampType()) \
    .add("value", DoubleType())

# Read streaming data from Kafka
streaming_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka-server:9092") \
    .option("subscribe", "events") \
    .load()

# Parse JSON and process
parsed_df = streaming_df \
    .select(from_json(col("value").cast("string"), schema).alias("data")) \
    .select("data.*")

# Aggregate by window
windowed_counts = parsed_df \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(
        window("timestamp", "5 minutes"),
        "event_type"
    ) \
    .count()

# Write to Delta Lake
query = windowed_counts.writeStream \
    .format("delta") \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .table("streaming_events")

# Display streaming data
display(parsed_df)
```

### Data Quality Checks

```python
from pyspark.sql.functions import col, count, when, isnan

def data_quality_report(df, table_name):
    """Generate data quality report for a DataFrame"""
    
    # Row count
    total_rows = df.count()
    
    # Column statistics
    quality_metrics = []
    
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        null_percentage = (null_count / total_rows) * 100
        
        distinct_count = df.select(column).distinct().count()
        
        quality_metrics.append({
            "table": table_name,
            "column": column,
            "total_rows": total_rows,
            "null_count": null_count,
            "null_percentage": round(null_percentage, 2),
            "distinct_count": distinct_count
        })
    
    quality_df = spark.createDataFrame(quality_metrics)
    
    # Write report
    quality_df.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("data_quality.reports")
    
    return quality_df

# Run quality check
df = spark.table("gold.orders_enriched")
report = data_quality_report(df, "orders_enriched")
display(report)

# Alert on quality issues
failed_checks = report.filter(col("null_percentage") > 10)
if failed_checks.count() > 0:
    dbutils.notebook.exit("QUALITY_CHECK_FAILED")
```

## Performance Optimization

### Caching

```python
# Cache DataFrame
df = spark.table("large_table")
df.cache()
df.count()  # Materialize cache

# Persist with storage level
from pyspark import StorageLevel
df.persist(StorageLevel.MEMORY_AND_DISK)

# Unpersist
df.unpersist()
```

### Partitioning

```python
# Write with partitioning
df.write \
    .format("delta") \
    .partitionBy("year", "month") \
    .save("/delta/partitioned_table")

# Read specific partitions
df = spark.read \
    .format("delta") \
    .load("/delta/partitioned_table") \
    .filter("year = 2024 AND month = 1")

# Repartition DataFrame
df = df.repartition(200, "customer_id")

# Coalesce (reduce partitions)
df = df.coalesce(10)
```

### Broadcast Joins

```python
from pyspark.sql.functions import broadcast

# Broadcast small table
large_df = spark.table("large_orders")
small_df = spark.table("small_products")

result = large_df.join(broadcast(small_df), "product_id")
```

### Z-Order Optimization

```sql
-- Z-order by frequently filtered columns
OPTIMIZE my_table ZORDER BY (date, customer_id);
```

### Adaptive Query Execution

```python
# Enable AQE (enabled by default in recent versions)
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```

## Unity Catalog

### Create Catalog and Schema

```sql
-- Create catalog
CREATE CATALOG IF NOT EXISTS my_catalog;

-- Use catalog
USE CATALOG my_catalog;

-- Create schema
CREATE SCHEMA IF NOT EXISTS my_schema;

-- Use schema
USE SCHEMA my_schema;
```

### Grant Permissions

```sql
-- Grant catalog access
GRANT USE CATALOG ON CATALOG my_catalog TO `user@company.com`;

-- Grant schema access
GRANT USE SCHEMA ON SCHEMA my_catalog.my_schema TO `user@company.com`;

-- Grant table permissions
GRANT SELECT ON TABLE my_catalog.my_schema.my_table TO `user@company.com`;
GRANT MODIFY ON TABLE my_catalog.my_schema.my_table TO `data_engineers`;

-- Grant all privileges
GRANT ALL PRIVILEGES ON TABLE my_catalog.my_schema.my_table TO `admin@company.com`;
```

### External Locations

```sql
-- Create external location
CREATE EXTERNAL LOCATION my_s3_location
URL 's3://my-bucket/path/'
WITH (STORAGE CREDENTIAL my_aws_credential);

-- Create table in external location
CREATE TABLE my_catalog.my_schema.external_table (
  id INT,
  name STRING
)
LOCATION 's3://my-bucket/path/external_table';
```

## Best Practices

### Notebook Organization

```python
# Use clear section headers
# ==============================================
# SECTION 1: Configuration and Setup
# ==============================================

# Use widgets for parameters
dbutils.widgets.text("environment", "dev")
dbutils.widgets.text("date", "2024-01-01")

# ==============================================
# SECTION 2: Data Ingestion
# ==============================================

# Load data with error handling
try:
    df = spark.read.format("delta").load("/path/to/data")
    print(f"Loaded {df.count()} rows")
except Exception as e:
    dbutils.notebook.exit(f"ERROR: {str(e)}")

# ==============================================
# SECTION 3: Data Transformation
# ==============================================

# Clear transformations with comments
df_transformed = df \
    .filter(col("status") == "active")  # Filter active records
    .select("id", "name", "value")      # Select required columns
    .withColumn("processed_date", current_date())  # Add processing date
```

### Delta Lake Best Practices

```python
# 1. Optimize regularly
%sql
OPTIMIZE my_table;

# 2. Use Z-ordering for common filters
%sql
OPTIMIZE my_table ZORDER BY (date, customer_id);

# 3. Vacuum old files
%sql
VACUUM my_table RETAIN 168 HOURS;  -- 7 days

# 4. Enable auto-optimize
%sql
ALTER TABLE my_table SET TBLPROPERTIES (
  'delta.autoOptimize.optimizeWrite' = 'true',
  'delta.autoOptimize.autoCompact' = 'true'
);

# 5. Use merge for upserts instead of overwrite
from delta.tables import DeltaTable

deltaTable = DeltaTable.forPath(spark, "/delta/my_table")
deltaTable.alias("target").merge(
    new_data.alias("source"),
    "target.id = source.id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()
```

### Performance Best Practices

```python
# 1. Avoid collect() on large datasets
# Bad: large_list = df.collect()
# Good: Process in distributed manner

# 2. Use column pruning
df.select("needed_col1", "needed_col2")  # Only select needed columns

# 3. Filter early
df.filter(col("date") >= "2024-01-01")  # Filter before joins

# 4. Use appropriate file formats
# Parquet/Delta for analytics
# Avro for streaming

# 5. Repartition before expensive operations
df = df.repartition(200)

# 6. Use broadcast for small tables
result = large_df.join(broadcast(small_df), "key")
```

## Troubleshooting

### Common Issues

**Out of Memory Errors:**
```python
# Increase driver/executor memory in cluster config
# Use .persist() strategically
# Reduce data with filters before operations
# Increase number of partitions

df = df.repartition(500)  # More partitions = less memory per partition
```

**Slow Performance:**
```python
# Check query plan
df.explain()

# Enable AQE
spark.conf.set("spark.sql.adaptive.enabled", "true")

# Optimize Delta tables
%sql
OPTIMIZE my_table ZORDER BY (commonly_filtered_column);

# Check for data skew
df.groupBy("partition_key").count().show()
```

**Connection Errors:**
```python
# Check cluster status
# Verify mount points
dbutils.fs.mounts()

# Test connectivity
dbutils.fs.ls("dbfs:/")
```

## Resources

- **Official Documentation**: https://docs.databricks.com/
- **Community Edition**: https://community.cloud.databricks.com/
- **Delta Lake**: https://delta.io/
- **MLflow**: https://mlflow.org/
- **Academy**: https://academy.databricks.com/
- **Community Forums**: https://community.databricks.com/

## Quick Reference

### Essential Commands

```python
# Databricks utilities
dbutils.fs.ls("/path")
dbutils.secrets.get(scope="scope", key="key")
dbutils.notebook.run("/path", timeout, arguments)

# Spark DataFrame
df = spark.read.format("delta").load("/path")
df.write.format("delta").mode("overwrite").save("/path")
df.createOrReplaceTempView("temp_view")

# Delta Lake
from delta.tables import DeltaTable
deltaTable = DeltaTable.forPath(spark, "/path")
deltaTable.update(condition, set)
deltaTable.delete(condition)
deltaTable.optimize().executeCompaction()

# MLflow
import mlflow
mlflow.start_run()
mlflow.log_param("param", value)
mlflow.log_metric("metric", value)
mlflow.log_model(model, "model")
mlflow.end_run()
```

### SQL Quick Reference

```sql
-- Create table
CREATE TABLE my_table (id INT, name STRING) USING DELTA;

-- Query
SELECT * FROM my_table WHERE condition;

-- Update
UPDATE my_table SET column = value WHERE condition;

-- Delete
DELETE FROM my_table WHERE condition;

-- Merge
MERGE INTO target USING source ON condition
WHEN MATCHED THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *;

-- Optimize
OPTIMIZE my_table ZORDER BY (column1, column2);

-- Vacuum
VACUUM my_table RETAIN 168 HOURS;

-- Time travel
SELECT * FROM my_table VERSION AS OF 5;
```

---

*This guide covers Databricks fundamentals and advanced features for building data and ML pipelines at scale.*
