# Data Collection

## Purpose

Efficiently and systematically gather data from all required sources while ensuring quality, completeness, and compliance.

## 1. Data Collection Planning

### Data Requirements Review
- **Feature Requirements:** [List required features]
- **Target Variable:** [What are we predicting?]
- **Time Period:** [Historical data needed]
- **Granularity:** [Record level: customer, transaction, etc.]
- **Volume:** [Expected data volume]
- **Frequency:** [Update frequency]

### Source Prioritization

| Source | Priority | Data Type | Volume | Quality | Complexity |
|--------|----------|-----------|--------|---------|-----------|
| [Source 1] | High | [Type] | [Size] | [Rating] | [Level] |
| [Source 2] | Medium | [Type] | [Size] | [Rating] | [Level] |
| [Source 3] | Low | [Type] | [Size] | [Rating] | [Level] |

## 2. Data Sources

### Internal Sources

#### Databases
```sql
-- Example: Customer database
SELECT customer_id, age, tenure, credit_score, 
       signup_date, usage_frequency
FROM customers
WHERE signup_date >= '2020-01-01'
```

- **Source Name:** [Database name]
- **Owner:** [Data owner]
- **Connection:** [Connection details]
- **Frequency:** [Update frequency]
- **SLA:** [Availability guarantee]

#### Data Warehouse/Lake
- **Platform:** [Snowflake, BigQuery, S3, etc.]
- **Existing Tables:** [Relevant tables]
- **Data Freshness:** [How recent is data?]
- **Quality Level:** [Known issues?]

#### Application Logs
- **Source:** [Application/system]
- **Log Format:** [JSON, CSV, structured]
- **Volume:** [Logs per day]
- **Retention:** [How long logs kept]

#### Business Systems
- **CRM/ERP Systems:** [System name and data]
- **Transactional Systems:** [Relevant transactions]
- **Operational Systems:** [Operations data]

### External Sources

#### Public Datasets
- **Kaggle:** [Dataset name, link]
- **Government Data:** [Source, dataset]
- **Academic Data:** [University, dataset]
- **Industry Benchmarks:** [Available benchmarks]

#### Third-Party Data
- **Vendor Name:** [Company]
- **Data Type:** [What data provided]
- **Cost:** [Cost per year]
- **Contract:** [Agreement terms]
- **Quality:** [Vendor's quality level]

#### APIs
```python
# Example: Public API call
import requests

response = requests.get(
    'https://api.example.com/data',
    params={'date': '2023-01-01', 'limit': 1000},
    headers={'Authorization': 'Bearer TOKEN'}
)
data = response.json()
```

- **Endpoint:** [API endpoint]
- **Rate Limits:** [Requests per hour]
- **Authentication:** [How to authenticate]
- **Pagination:** [How to handle large datasets]

## 3. Data Extraction

### Batch Extraction

```python
import pandas as pd
from datetime import datetime, timedelta

# Extract from SQL database
def extract_customer_data(start_date, end_date):
    query = f"""
    SELECT *
    FROM customers
    WHERE created_date BETWEEN '{start_date}' AND '{end_date}'
    """
    df = pd.read_sql(query, connection)
    return df

# Execute extraction
start = datetime(2020, 1, 1)
end = datetime.now()
df = extract_customer_data(start, end)
```

**Extraction Steps:**
1. Connect to data source
2. Query/filter data
3. Extract to temporary storage
4. Validate extraction
5. Load to staging area

### Real-Time Extraction

- **Streaming Technology:** [Kafka, Kinesis, Pub/Sub]
- **Message Format:** [JSON, Avro, Protobuf]
- **Consumer Group:** [Application group]
- **Lag Monitoring:** [Track ingestion delay]

### API Data Collection

```python
import pandas as pd
from datetime import datetime, timedelta
import time

class APIDataCollector:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
    
    def collect_data(self, start_date, end_date):
        all_data = []
        current_date = start_date
        
        while current_date <= end_date:
            response = self._fetch_day(current_date)
            all_data.extend(response['data'])
            current_date += timedelta(days=1)
            time.sleep(1)  # Respect rate limits
        
        return pd.DataFrame(all_data)
    
    def _fetch_day(self, date):
        # Implementation
        pass
```

## 4. Data Staging & Storage

### Staging Area
```
Raw Data Layer
    ↓
Data Warehouse/Lake
    ↓
Processed Data Layer
    ↓
Feature Store
```

### Storage Options

#### CSV/Parquet Files
```bash
# Save extracted data
df.to_parquet('data/raw/customers_20240101.parquet')
```
- Format: [CSV, Parquet, ORC]
- Compression: [gzip, snappy, lz4]
- Location: [S3, ADLS, GCS]

#### Database
```sql
CREATE TABLE staging.customers (
    customer_id INT,
    age INT,
    credit_score INT,
    created_date DATE,
    PRIMARY KEY (customer_id)
);
```

#### Data Lake
```
s3://company-datalake/
├── raw/
│   ├── customers/
│   │   ├── 2024-01-01/
│   │   └── 2024-01-02/
│   └── transactions/
├── staging/
└── processed/
```

## 5. Data Collection Pipeline

### Automated Pipeline

```python
# Example: Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

dag = DAG(
    'daily_data_collection',
    default_args=default_args,
    schedule_interval='0 2 * * *',  # 2 AM daily
)

def extract_customers():
    # Extraction logic
    pass

def extract_transactions():
    # Extraction logic
    pass

def validate_data():
    # Validation logic
    pass

extract_task = PythonOperator(
    task_id='extract_customers',
    python_callable=extract_customers,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

extract_task >> validate_task
```

### Error Handling

```python
def collect_with_retry(source, max_retries=3):
    for attempt in range(max_retries):
        try:
            data = source.fetch_data()
            return data
        except TemporaryError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
        except PermanentError as e:
            # Log and notify
            notify_admin(f"Permanent error: {e}")
            raise
```

## 6. Data Collection Schedule

### Timeline

| Phase | Timeline | Owner | Deliverable |
|-------|----------|-------|-------------|
| **Planning** | Week 1 | Data Engineer | Collection plan |
| **Setup** | Week 2-3 | Data Engineer | Pipeline infrastructure |
| **Initial Load** | Week 3-4 | Data Engineer | Historical data |
| **Validation** | Week 4 | Data Scientist | Quality report |
| **Automation** | Week 5 | Data Engineer | Automated pipeline |

### Collection Frequency
- **Initial:** Full historical data for modeling
- **Ongoing:** Daily, hourly, or real-time depending on use case
- **Refresh:** Full refresh weekly or as needed

## 7. Quality Checks During Collection

### Pre-Collection Validation
- [ ] Data source accessible
- [ ] Credentials valid
- [ ] Expected tables/files exist
- [ ] No schema changes

### Post-Collection Validation
```python
def validate_extracted_data(df):
    checks = {
        'row_count': len(df) > 0,
        'no_duplicates': len(df) == len(df.drop_duplicates()),
        'expected_columns': all(col in df.columns for col in EXPECTED_COLS),
        'no_all_nulls': not df.isnull().all().any(),
    }
    return all(checks.values()), checks
```

**Checks:**
- [ ] Row count reasonable
- [ ] No unexpected duplicates
- [ ] Expected columns present
- [ ] Expected data types
- [ ] No unexpected nulls
- [ ] Values in expected ranges

## 8. Data Collection Monitoring

### Metrics to Track
- **Extraction Volume:** Rows extracted per run
- **Extraction Time:** How long extraction takes
- **Data Freshness:** How old is the data?
- **Failure Rate:** How often does extraction fail?
- **Data Quality Score:** Overall quality metric

### Alerts
```
If extraction fails: Notify data team immediately
If volume drops >50%: Investigate source
If data >24 hours old: Alert on SLA miss
If quality score <80%: Stop data usage
```

## 9. Documentation

### Data Source Documentation
- **Source Name:** [Name]
- **Owner:** [Owner/Contact]
- **Description:** [What data is available]
- **Fields:** [List of available fields]
- **Update Frequency:** [How often updated]
- **Quality:** [Known issues]
- **Access:** [How to access]
- **Cost:** [Any costs]

### Collection Script Documentation
```python
"""
Script: daily_customer_extraction.py
Purpose: Extract daily customer data for ML pipeline
Schedule: 2 AM UTC daily
Owner: Data Team
Inputs: Customer database connection string
Outputs: s3://datalake/raw/customers/{date}/data.parquet
Dependencies: boto3, pandas, sqlalchemy
Last Updated: 2024-01-15
"""
```

## Best Practices

1. ✅ Automate everything possible
2. ✅ Validate data immediately after collection
3. ✅ Monitor collection pipeline continuously
4. ✅ Document data sources thoroughly
5. ✅ Version data extraction logic
6. ✅ Implement error handling and retries
7. ✅ Store raw data (don't overwrite)
8. ✅ Track data lineage

## Common Pitfalls

- ❌ Manual data collection (not scalable)
- ❌ No error handling (brittle pipelines)
- ❌ Missing quality checks (garbage in, garbage out)
- ❌ No monitoring (silent failures)
- ❌ Overwriting raw data (can't reprocess)
- ❌ Poor documentation
- ❌ No backups

---

## Related Documents

- [Data Strategy](../01-planning/data-strategy.md) - Data requirements planning
- [Data Exploration](./data-exploration.md) - EDA on collected data
- [Data Validation](./data-validation.md) - Quality assurance
- [Dataset Card](../templates/dataset-card.md) - Dataset documentation

---

*Quality data collection is the foundation for model success*
