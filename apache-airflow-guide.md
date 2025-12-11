# Apache Airflow Guide

## What is Apache Airflow?

Apache Airflow is an open-source platform for programmatically authoring, scheduling, and monitoring workflows. It allows you to define workflows as Directed Acyclic Graphs (DAGs) of tasks, with rich scheduling capabilities and an intuitive UI for monitoring and troubleshooting.

## Prerequisites

- Python 3.8+ installed
- Basic understanding of Python
- Understanding of workflows and ETL concepts
- Database (SQLite for development, PostgreSQL/MySQL for production)

## Installation

### Using pip

```bash
# Set Airflow home (optional, defaults to ~/airflow)
export AIRFLOW_HOME=~/airflow

# Install Airflow with constraints
AIRFLOW_VERSION=2.8.0
PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"

# Initialize the database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start the web server (default port 8080)
airflow webserver --port 8080

# In another terminal, start the scheduler
airflow scheduler
```

### Using Docker

```yaml
# docker-compose.yml
version: '3.8'

x-airflow-common:
  &airflow-common
  image: apache/airflow:2.8.0
  environment:
    &airflow-common-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
  volumes:
    - ./dags:/opt/airflow/dags
    - ./logs:/opt/airflow/logs
    - ./plugins:/opt/airflow/plugins
  user: "${AIRFLOW_UID:-50000}:0"
  depends_on:
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-db-volume:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 5s
      retries: 5

  airflow-webserver:
    <<: *airflow-common
    command: webserver
    ports:
      - 8080:8080
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 10s
      timeout: 10s
      retries: 5

  airflow-scheduler:
    <<: *airflow-common
    command: scheduler
    healthcheck:
      test: ["CMD-SHELL", 'airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"']
      interval: 10s
      timeout: 10s
      retries: 5

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        mkdir -p /sources/logs /sources/dags /sources/plugins
        chown -R "${AIRFLOW_UID}:0" /sources/{logs,dags,plugins}
        exec /entrypoint airflow version

volumes:
  postgres-db-volume:
```

```bash
# Initialize and start
docker-compose up airflow-init
docker-compose up -d

# Access at http://localhost:8080
# Default credentials: admin/admin
```

## Core Concepts

### 1. **DAG (Directed Acyclic Graph)**
A collection of tasks with defined dependencies and execution order.

### 2. **Task**
A single unit of work defined by an operator.

### 3. **Operator**
A template for a specific type of task (PythonOperator, BashOperator, etc.).

### 4. **Task Instance**
A specific run of a task for a particular execution date.

### 5. **Executor**
Determines how tasks are executed (Sequential, Local, Celery, Kubernetes).

### 6. **Scheduler**
Monitors DAGs and triggers task instances when dependencies are met.

### 7. **XCom**
Cross-communication mechanism for passing data between tasks.

### 8. **Hook**
Interface to external platforms and databases.

### 9. **Connection**
Credentials and connection information for external systems.

### 10. **Pool**
Limit concurrent execution of tasks.

## Creating Your First DAG

### Basic DAG Structure

```python
# dags/my_first_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# Default arguments for tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['admin@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'my_first_dag',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['example'],
)

# Define tasks
task1 = BashOperator(
    task_id='print_date',
    bash_command='date',
    dag=dag,
)

task2 = BashOperator(
    task_id='sleep',
    bash_command='sleep 5',
    dag=dag,
)

task3 = BashOperator(
    task_id='print_hello',
    bash_command='echo "Hello World"',
    dag=dag,
)

# Set task dependencies
task1 >> task2 >> task3
# Or equivalently:
# task1.set_downstream(task2)
# task2.set_downstream(task3)
```

### Using Context Manager (Recommended)

```python
from datetime import datetime, timedelta
from airflow.decorators import dag, task

@dag(
    schedule_interval='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['example'],
)
def my_etl_dag():
    """
    ### Simple ETL DAG
    This is a simple ETL pipeline using TaskFlow API
    """
    
    @task()
    def extract():
        """Extract data from source"""
        data = {'value': 42, 'name': 'example'}
        return data
    
    @task()
    def transform(data: dict):
        """Transform the data"""
        data['value'] = data['value'] * 2
        return data
    
    @task()
    def load(data: dict):
        """Load data to destination"""
        print(f"Loading data: {data}")
        return "Success"
    
    # Define the pipeline
    data = extract()
    transformed = transform(data)
    load(transformed)

# Instantiate the DAG
dag = my_etl_dag()
```

## Operators

### BashOperator

```python
from airflow.operators.bash import BashOperator

# Simple command
bash_task = BashOperator(
    task_id='bash_task',
    bash_command='echo "Hello World"',
)

# Multiple commands
bash_task = BashOperator(
    task_id='bash_multi',
    bash_command='''
        cd /tmp
        echo "Working directory: $(pwd)"
        date
    ''',
)

# With environment variables
bash_task = BashOperator(
    task_id='bash_env',
    bash_command='echo "Value: $MY_VAR"',
    env={'MY_VAR': 'Hello'},
)
```

### PythonOperator

```python
from airflow.operators.python import PythonOperator

def my_python_function(name, age):
    print(f"Hello {name}, you are {age} years old")
    return f"Processed {name}"

python_task = PythonOperator(
    task_id='python_task',
    python_callable=my_python_function,
    op_kwargs={'name': 'John', 'age': 30},
)

# With context
def print_context(**context):
    print(f"Execution date: {context['execution_date']}")
    print(f"Task instance: {context['task_instance']}")
    print(f"DAG: {context['dag']}")

python_context_task = PythonOperator(
    task_id='python_context',
    python_callable=print_context,
    provide_context=True,
)
```

### TaskFlow API (Decorator)

```python
from airflow.decorators import task

@task()
def extract_data():
    return {'id': 1, 'value': 100}

@task()
def transform_data(data: dict):
    data['value'] = data['value'] * 2
    return data

@task()
def load_data(data: dict):
    print(f"Loading: {data}")

# Use in DAG
data = extract_data()
transformed = transform_data(data)
load_data(transformed)
```

### EmailOperator

```python
from airflow.operators.email import EmailOperator

email_task = EmailOperator(
    task_id='send_email',
    to='recipient@example.com',
    subject='Airflow Alert',
    html_content='<h3>Task completed successfully</h3>',
)
```

### SqliteOperator

```python
from airflow.providers.sqlite.operators.sqlite import SqliteOperator

create_table = SqliteOperator(
    task_id='create_table',
    sqlite_conn_id='sqlite_default',
    sql='''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT
        );
    ''',
)

insert_data = SqliteOperator(
    task_id='insert_data',
    sqlite_conn_id='sqlite_default',
    sql='''
        INSERT INTO users (name, email)
        VALUES ('John Doe', 'john@example.com');
    ''',
)
```

### PostgreSQL Operator

```python
from airflow.providers.postgres.operators.postgres import PostgresOperator

postgres_task = PostgresOperator(
    task_id='postgres_task',
    postgres_conn_id='postgres_default',
    sql='''
        SELECT * FROM users
        WHERE created_at > NOW() - INTERVAL '1 day';
    ''',
)
```

### HTTP Operator

```python
from airflow.providers.http.operators.http import SimpleHttpOperator

http_task = SimpleHttpOperator(
    task_id='http_request',
    http_conn_id='http_default',
    endpoint='api/data',
    method='GET',
    headers={'Content-Type': 'application/json'},
    response_check=lambda response: response.status_code == 200,
)
```

## DAG Scheduling

### Schedule Expressions

```python
from datetime import datetime, timedelta
from airflow import DAG

# Cron expressions
dag = DAG(
    'scheduled_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval='0 0 * * *',  # Daily at midnight
)

# Preset intervals
dag = DAG(
    'preset_intervals',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',  # Same as '0 0 * * *'
)

# Available presets:
# @once - Run once
# @hourly - 0 * * * *
# @daily - 0 0 * * *
# @weekly - 0 0 * * 0
# @monthly - 0 0 1 * *
# @yearly - 0 0 1 1 *

# Timedelta
dag = DAG(
    'timedelta_schedule',
    start_date=datetime(2024, 1, 1),
    schedule_interval=timedelta(hours=6),  # Every 6 hours
)

# No schedule (manual only)
dag = DAG(
    'manual_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
)
```

### Common Cron Patterns

```python
# Every 15 minutes
schedule_interval='*/15 * * * *'

# Every hour at minute 30
schedule_interval='30 * * * *'

# Every day at 2:30 AM
schedule_interval='30 2 * * *'

# Every Monday at 9 AM
schedule_interval='0 9 * * 1'

# First day of every month at midnight
schedule_interval='0 0 1 * *'

# Every weekday at 6 PM
schedule_interval='0 18 * * 1-5'

# Every 6 hours
schedule_interval='0 */6 * * *'
```

## XCom (Cross-Communication)

### Pushing and Pulling Data

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2024, 1, 1), schedule_interval='@daily', catchup=False)
def xcom_example():
    
    @task()
    def push_data():
        """Push data to XCom"""
        return {'key': 'value', 'number': 42}
    
    @task()
    def pull_data(data: dict):
        """Pull data from XCom (automatic with TaskFlow)"""
        print(f"Received data: {data}")
        return data['number'] * 2
    
    @task()
    def manual_xcom(**context):
        """Manually push/pull XCom"""
        ti = context['task_instance']
        
        # Push to XCom
        ti.xcom_push(key='my_key', value='my_value')
        
        # Pull from XCom
        value = ti.xcom_pull(task_ids='push_data', key='return_value')
        print(f"Pulled value: {value}")
    
    data = push_data()
    result = pull_data(data)
    manual_xcom()

dag = xcom_example()
```

## Branching and Conditionals

### BranchPythonOperator

```python
from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

@dag(start_date=datetime(2024, 1, 1), schedule_interval='@daily', catchup=False)
def branching_dag():
    
    def choose_branch(**context):
        """Decide which branch to take"""
        execution_date = context['execution_date']
        if execution_date.day % 2 == 0:
            return 'even_day_task'
        else:
            return 'odd_day_task'
    
    branch_task = BranchPythonOperator(
        task_id='branch_task',
        python_callable=choose_branch,
    )
    
    even_task = EmptyOperator(task_id='even_day_task')
    odd_task = EmptyOperator(task_id='odd_day_task')
    join_task = EmptyOperator(
        task_id='join_task',
        trigger_rule='none_failed_min_one_success'
    )
    
    branch_task >> [even_task, odd_task] >> join_task

dag = branching_dag()
```

### Using TaskFlow API

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2024, 1, 1), schedule_interval='@daily', catchup=False)
def conditional_dag():
    
    @task.branch()
    def branch_func(value: int):
        if value > 50:
            return 'high_value_task'
        else:
            return 'low_value_task'
    
    @task()
    def get_value():
        return 75
    
    @task()
    def high_value_task():
        print("Processing high value")
    
    @task()
    def low_value_task():
        print("Processing low value")
    
    value = get_value()
    branch = branch_func(value)
    high_value_task() << branch
    low_value_task() << branch

dag = conditional_dag()
```

## Dynamic Task Generation

```python
from airflow.decorators import dag, task
from datetime import datetime

@dag(start_date=datetime(2024, 1, 1), schedule_interval='@daily', catchup=False)
def dynamic_tasks_dag():
    
    @task()
    def get_items():
        """Return list of items to process"""
        return ['item1', 'item2', 'item3', 'item4']
    
    @task()
    def process_item(item: str):
        """Process individual item"""
        print(f"Processing {item}")
        return f"Processed {item}"
    
    items = get_items()
    # Dynamically create tasks
    process_item.expand(item=items)

dag = dynamic_tasks_dag()
```

## Task Groups

```python
from airflow.decorators import dag, task
from airflow.utils.task_group import TaskGroup
from datetime import datetime

@dag(start_date=datetime(2024, 1, 1), schedule_interval='@daily', catchup=False)
def task_group_dag():
    
    @task()
    def start():
        print("Starting pipeline")
    
    with TaskGroup("extract_group", tooltip="Extract tasks") as extract_group:
        @task()
        def extract_a():
            return "data_a"
        
        @task()
        def extract_b():
            return "data_b"
        
        extract_a()
        extract_b()
    
    with TaskGroup("transform_group", tooltip="Transform tasks") as transform_group:
        @task()
        def transform_a():
            print("Transform A")
        
        @task()
        def transform_b():
            print("Transform B")
        
        transform_a()
        transform_b()
    
    @task()
    def end():
        print("Pipeline complete")
    
    start() >> extract_group >> transform_group >> end()

dag = task_group_dag()
```

## Complete ETL Example

```python
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.http.hooks.http import HttpHook
import pandas as pd

default_args = {
    'owner': 'data_team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email': ['data-team@example.com'],
    'email_on_failure': True,
}

@dag(
    dag_id='etl_pipeline',
    default_args=default_args,
    description='Complete ETL pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['etl', 'production'],
)
def etl_pipeline():
    """
    ### ETL Pipeline
    Daily ETL pipeline to extract data from API,
    transform it, and load into database.
    """
    
    @task()
    def extract_from_api():
        """Extract data from REST API"""
        http_hook = HttpHook(method='GET', http_conn_id='api_default')
        response = http_hook.run('api/v1/data')
        
        if response.status_code == 200:
            data = response.json()
            print(f"Extracted {len(data)} records")
            return data
        else:
            raise Exception(f"API call failed: {response.status_code}")
    
    @task()
    def transform_data(raw_data: list):
        """Transform and clean data"""
        df = pd.DataFrame(raw_data)
        
        # Data cleaning
        df = df.dropna()
        df = df.drop_duplicates()
        
        # Data transformation
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['value'] = df['value'].astype(float)
        df['processed_date'] = datetime.now()
        
        # Business logic
        df['category'] = df['value'].apply(
            lambda x: 'high' if x > 100 else 'low'
        )
        
        print(f"Transformed {len(df)} records")
        return df.to_dict('records')
    
    @task()
    def validate_data(data: list):
        """Validate transformed data"""
        df = pd.DataFrame(data)
        
        # Validation checks
        assert len(df) > 0, "No data to load"
        assert df['value'].notna().all(), "Null values found"
        assert df['value'].min() >= 0, "Negative values found"
        
        print("Data validation passed")
        return data
    
    @task()
    def load_to_database(data: list):
        """Load data to PostgreSQL database"""
        postgres_hook = PostgresHook(postgres_conn_id='postgres_default')
        
        # Create table if not exists
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS processed_data (
            id SERIAL PRIMARY KEY,
            value FLOAT,
            category VARCHAR(50),
            created_at TIMESTAMP,
            processed_date TIMESTAMP
        );
        """
        postgres_hook.run(create_table_sql)
        
        # Insert data
        df = pd.DataFrame(data)
        records = df.to_records(index=False)
        
        postgres_hook.insert_rows(
            table='processed_data',
            rows=records,
            target_fields=df.columns.tolist()
        )
        
        print(f"Loaded {len(df)} records to database")
        return len(df)
    
    @task()
    def send_notification(record_count: int):
        """Send completion notification"""
        print(f"ETL pipeline completed successfully")
        print(f"Total records processed: {record_count}")
        # Could send email, Slack message, etc.
    
    # Define the pipeline
    raw_data = extract_from_api()
    transformed = transform_data(raw_data)
    validated = validate_data(transformed)
    count = load_to_database(validated)
    send_notification(count)

# Instantiate the DAG
dag = etl_pipeline()
```

## Sensors

### FileSensor

```python
from airflow.sensors.filesystem import FileSensor

wait_for_file = FileSensor(
    task_id='wait_for_file',
    filepath='/tmp/data.csv',
    poke_interval=30,  # Check every 30 seconds
    timeout=600,  # Timeout after 10 minutes
    mode='poke',  # or 'reschedule'
)
```

### S3KeySensor

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor

wait_for_s3_file = S3KeySensor(
    task_id='wait_for_s3_file',
    bucket_name='my-bucket',
    bucket_key='data/input.csv',
    aws_conn_id='aws_default',
    poke_interval=60,
    timeout=3600,
)
```

### ExternalTaskSensor

```python
from airflow.sensors.external_task import ExternalTaskSensor

wait_for_upstream_dag = ExternalTaskSensor(
    task_id='wait_for_upstream',
    external_dag_id='upstream_dag',
    external_task_id='final_task',
    timeout=600,
)
```

## Connections

### Managing Connections via UI

1. Go to Admin → Connections
2. Click "+" to add new connection
3. Fill in connection details
4. Test connection
5. Save

### Managing Connections via CLI

```bash
# Add connection
airflow connections add 'my_postgres' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-schema 'mydatabase' \
    --conn-login 'user' \
    --conn-password 'password' \
    --conn-port 5432

# List connections
airflow connections list

# Get connection
airflow connections get my_postgres

# Delete connection
airflow connections delete my_postgres
```

### Using Connections in Code

```python
from airflow.providers.postgres.hooks.postgres import PostgresHook

def query_database():
    postgres_hook = PostgresHook(postgres_conn_id='my_postgres')
    sql = "SELECT * FROM users LIMIT 10"
    results = postgres_hook.get_records(sql)
    return results
```

## Variables

### Managing Variables via UI

1. Go to Admin → Variables
2. Click "+" to add new variable
3. Enter key and value
4. Save

### Managing Variables via CLI

```bash
# Set variable
airflow variables set my_var my_value

# Get variable
airflow variables get my_var

# List variables
airflow variables list

# Delete variable
airflow variables delete my_var

# Import from JSON
airflow variables import variables.json

# Export to JSON
airflow variables export variables.json
```

### Using Variables in Code

```python
from airflow.models import Variable

# Get variable
my_var = Variable.get("my_var")

# Get with default value
my_var = Variable.get("my_var", default_var="default_value")

# Get JSON variable
config = Variable.get("config", deserialize_json=True)

# Set variable
Variable.set("my_var", "new_value")

# Use in task
@task()
def use_variable():
    api_key = Variable.get("api_key")
    print(f"Using API key: {api_key[:5]}...")
```

## Airflow CLI Commands

```bash
# DAG commands
airflow dags list
airflow dags show my_dag
airflow dags trigger my_dag
airflow dags pause my_dag
airflow dags unpause my_dag
airflow dags delete my_dag
airflow dags backfill my_dag -s 2024-01-01 -e 2024-01-31

# Task commands
airflow tasks list my_dag
airflow tasks test my_dag my_task 2024-01-01
airflow tasks run my_dag my_task 2024-01-01
airflow tasks clear my_dag -s 2024-01-01

# Database commands
airflow db init
airflow db upgrade
airflow db reset

# User commands
airflow users create -u admin -p admin -f Admin -l User -r Admin -e admin@example.com
airflow users list
airflow users delete -u username

# Scheduler
airflow scheduler

# Webserver
airflow webserver -p 8080

# Worker (for Celery executor)
airflow celery worker

# Info
airflow version
airflow config list
```

## Best Practices

### 1. Use the TaskFlow API

```python
# Good - TaskFlow API
@dag(schedule_interval='@daily', start_date=datetime(2024, 1, 1))
def my_dag():
    @task()
    def extract():
        return data
    
    @task()
    def transform(data):
        return transformed
    
    data = extract()
    transform(data)

dag = my_dag()
```

### 2. Idempotent Tasks

Tasks should produce the same result when run multiple times.

```python
@task()
def idempotent_load(data):
    # Use UPSERT instead of INSERT
    # DELETE before INSERT
    # Use transaction
    pass
```

### 3. Avoid Top-Level Code

```python
# Bad - Executes on every scheduler loop
data = expensive_operation()

# Good - Executes only when task runs
@task()
def extract():
    data = expensive_operation()
    return data
```

### 4. Use Appropriate Executors

```python
# Development: SequentialExecutor (default)
# Production with Postgres: LocalExecutor
# Distributed: CeleryExecutor or KubernetesExecutor
```

### 5. Set Proper Timeouts

```python
@task(execution_timeout=timedelta(hours=1))
def long_running_task():
    # Task will fail if exceeds 1 hour
    pass
```

### 6. Use Pools for Resource Management

```python
task = PythonOperator(
    task_id='db_task',
    python_callable=query_db,
    pool='database_pool',
    pool_slots=2,
)
```

### 7. Handle Failures Gracefully

```python
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': send_failure_alert,
}

def send_failure_alert(context):
    # Send notification on failure
    pass
```

### 8. Use Templating

```python
from airflow.operators.bash import BashOperator

bash_task = BashOperator(
    task_id='templated_task',
    bash_command='echo "Execution date: {{ ds }}"',
)
```

### 9. Document Your DAGs

```python
@dag(
    dag_id='well_documented_dag',
    description='Clear description of what this DAG does',
    doc_md="""
    ## Documentation
    
    ### Purpose
    This DAG processes daily sales data
    
    ### Schedule
    Runs daily at 2 AM
    
    ### Dependencies
    - Requires sales_api to be available
    - Writes to postgres database
    """,
    tags=['sales', 'etl', 'production'],
)
def documented_dag():
    pass
```

### 10. Use Separate Environments

```bash
# Development
AIRFLOW__CORE__DAGS_FOLDER=/dev/dags
AIRFLOW__CORE__LOAD_EXAMPLES=True

# Production
AIRFLOW__CORE__DAGS_FOLDER=/prod/dags
AIRFLOW__CORE__LOAD_EXAMPLES=False
```

## Configuration

### airflow.cfg Key Settings

```ini
[core]
# DAG folder
dags_folder = /opt/airflow/dags

# Executor
executor = LocalExecutor

# Timezone
default_timezone = UTC

# Parallelism
parallelism = 32
dag_concurrency = 16
max_active_runs_per_dag = 16

[scheduler]
# DAG processing interval
dag_dir_list_interval = 300

# Catchup
catchup_by_default = False

[webserver]
# Web server host/port
web_server_host = 0.0.0.0
web_server_port = 8080

# Authentication
authenticate = True
auth_backend = airflow.api.auth.backend.basic_auth

[email]
# Email settings
email_backend = airflow.utils.email.send_email_smtp

[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = your-email@gmail.com
smtp_password = your-password
smtp_port = 587
smtp_mail_from = airflow@example.com
```

## Monitoring and Troubleshooting

### Check DAG Status

```bash
# View DAG runs
airflow dags list-runs -d my_dag

# View task instances
airflow tasks list my_dag
airflow tasks states-for-dag-run my_dag 2024-01-01
```

### Logs

```bash
# Task logs location
$AIRFLOW_HOME/logs/dag_id/task_id/execution_date/

# View logs via CLI
airflow tasks logs my_dag my_task 2024-01-01

# Tail logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/scheduler.log
```

### Common Issues

**DAG not appearing:**
- Check DAG file for Python errors
- Ensure file is in dags_folder
- Check scheduler logs
- Verify DAG is not paused

**Tasks stuck in queue:**
- Check executor capacity
- Review pool slots
- Check scheduler status

**Task failures:**
- Review task logs
- Check XCom data
- Verify connections
- Test task independently

## Useful Resources

- Official Documentation: https://airflow.apache.org/docs/
- GitHub: https://github.com/apache/airflow
- Provider Packages: https://airflow.apache.org/docs/apache-airflow-providers/
- Community: https://airflow.apache.org/community/

## Quick Reference

| Command | Description |
|---------|-------------|
| `airflow dags list` | List all DAGs |
| `airflow dags trigger <dag_id>` | Trigger DAG run |
| `airflow tasks test <dag> <task> <date>` | Test single task |
| `airflow db init` | Initialize database |
| `airflow webserver` | Start web UI |
| `airflow scheduler` | Start scheduler |
| `airflow users create` | Create user |
| `airflow connections add` | Add connection |
| `airflow variables set` | Set variable |

### Common Template Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{{ ds }}` | Execution date (YYYY-MM-DD) | 2024-01-01 |
| `{{ ds_nodash }}` | Execution date (YYYYMMDD) | 20240101 |
| `{{ ts }}` | Timestamp (ISO format) | 2024-01-01T00:00:00+00:00 |
| `{{ dag }}` | DAG object | - |
| `{{ task }}` | Task object | - |
| `{{ params }}` | User-defined params | - |
| `{{ var.value.my_var }}` | Airflow variable | - |
| `{{ conn.my_conn.host }}` | Connection property | - |

---

*This guide covers Apache Airflow fundamentals. For production use, consider implementing monitoring (statsd, Prometheus), proper authentication (LDAP, OAuth), scaling strategies (CeleryExecutor, KubernetesExecutor), and security best practices.*
