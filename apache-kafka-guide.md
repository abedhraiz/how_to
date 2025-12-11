# Apache Kafka Guide

## What is Apache Kafka?

Apache Kafka is a distributed event streaming platform capable of handling trillions of events a day. Originally developed at LinkedIn, it's now used by thousands of companies for high-performance data pipelines, streaming analytics, data integration, and mission-critical applications.

**Key Features:**
- High throughput and low latency
- Distributed and fault-tolerant
- Horizontal scalability
- Persistent storage
- Real-time processing
- Strong ordering guarantees

## Core Concepts

### Topics
Logical channels where records are published. Topics are partitioned for scalability.

### Partitions
Each topic is divided into partitions for parallel processing and load distribution.

### Producers
Applications that publish records to Kafka topics.

### Consumers
Applications that subscribe to topics and process records.

### Consumer Groups
Multiple consumers working together to consume a topic in parallel.

### Brokers
Kafka servers that store data and serve clients.

### ZooKeeper / KRaft
Coordination service (ZooKeeper) or built-in consensus protocol (KRaft in newer versions).

## Prerequisites

- Java 11 or higher
- Basic understanding of distributed systems
- Command line familiarity
- Understanding of messaging patterns

## Installation

### Using Docker

```bash
# Create network
docker network create kafka-network

# Start ZooKeeper
docker run -d \
  --name zookeeper \
  --network kafka-network \
  -p 2181:2181 \
  -e ZOOKEEPER_CLIENT_PORT=2181 \
  -e ZOOKEEPER_TICK_TIME=2000 \
  confluentinc/cp-zookeeper:latest

# Start Kafka broker
docker run -d \
  --name kafka \
  --network kafka-network \
  -p 9092:9092 \
  -e KAFKA_BROKER_ID=1 \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  confluentinc/cp-kafka:latest
```

### Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    container_name: zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"
    networks:
      - kafka-network

  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: kafka
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
      - "9093:9093"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092,PLAINTEXT_INTERNAL://kafka:9093
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,PLAINTEXT_INTERNAL:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT_INTERNAL
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
    networks:
      - kafka-network

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9093
      KAFKA_CLUSTERS_0_ZOOKEEPER: zookeeper:2181
    networks:
      - kafka-network

networks:
  kafka-network:
    driver: bridge
```

```bash
# Start Kafka stack
docker-compose up -d

# View logs
docker-compose logs -f kafka

# Access Kafka UI at http://localhost:8080
```

### Manual Installation (Linux)

```bash
# Download Kafka
wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz

# Extract
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Start ZooKeeper
bin/zookeeper-server-start.sh config/zookeeper.properties

# In another terminal, start Kafka
bin/kafka-server-start.sh config/server.properties
```

## Basic Operations

### Topic Management

```bash
# Create topic
kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 3 \
  --replication-factor 1

# List topics
kafka-topics --list --bootstrap-server localhost:9092

# Describe topic
kafka-topics --describe \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# Alter topic (add partitions)
kafka-topics --alter \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 5

# Delete topic
kafka-topics --delete \
  --bootstrap-server localhost:9092 \
  --topic my-topic
```

### Using Docker

```bash
# Create topic
docker exec kafka kafka-topics --create \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --partitions 3 \
  --replication-factor 1

# List topics
docker exec kafka kafka-topics --list \
  --bootstrap-server localhost:9092

# Describe topic
docker exec kafka kafka-topics --describe \
  --bootstrap-server localhost:9092 \
  --topic my-topic
```

### Console Producer and Consumer

```bash
# Start console producer
kafka-console-producer \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# Type messages (one per line)
> Hello Kafka
> This is a test message
> ^C (Ctrl+C to exit)

# Start console consumer (from beginning)
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --from-beginning

# Start console consumer (latest messages)
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic my-topic

# Consumer with group
kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic my-topic \
  --group my-consumer-group
```

## Producers

### Python Producer

```python
from kafka import KafkaProducer
import json
from datetime import datetime

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8') if k else None
)

# Send message
message = {
    'user_id': 123,
    'action': 'page_view',
    'timestamp': datetime.now().isoformat()
}

future = producer.send('user-events', value=message, key='user-123')

# Wait for acknowledgment
try:
    record_metadata = future.get(timeout=10)
    print(f"Message sent to {record_metadata.topic} partition {record_metadata.partition} offset {record_metadata.offset}")
except Exception as e:
    print(f"Failed to send message: {e}")

# Send multiple messages
for i in range(100):
    message = {
        'id': i,
        'data': f'Message {i}',
        'timestamp': datetime.now().isoformat()
    }
    producer.send('my-topic', value=message)

# Flush and close
producer.flush()
producer.close()

# Producer with callback
def on_send_success(record_metadata):
    print(f"Sent to {record_metadata.topic}[{record_metadata.partition}] @ offset {record_metadata.offset}")

def on_send_error(excp):
    print(f"Error: {excp}")

future = producer.send('my-topic', value=message)
future.add_callback(on_send_success)
future.add_errback(on_send_error)
```

### Java Producer

```java
import org.apache.kafka.clients.producer.*;
import org.apache.kafka.common.serialization.StringSerializer;
import java.util.Properties;

public class SimpleProducer {
    public static void main(String[] args) {
        // Configure producer
        Properties props = new Properties();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, StringSerializer.class.getName());
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.RETRIES_CONFIG, 3);
        props.put(ProducerConfig.LINGER_MS_CONFIG, 1);
        
        // Create producer
        Producer<String, String> producer = new KafkaProducer<>(props);
        
        // Send messages
        for (int i = 0; i < 100; i++) {
            ProducerRecord<String, String> record = 
                new ProducerRecord<>("my-topic", "key-" + i, "value-" + i);
            
            // Async send with callback
            producer.send(record, new Callback() {
                @Override
                public void onCompletion(RecordMetadata metadata, Exception exception) {
                    if (exception != null) {
                        exception.printStackTrace();
                    } else {
                        System.out.println("Sent: " + metadata.topic() + 
                                         " [" + metadata.partition() + "] " +
                                         " @ offset " + metadata.offset());
                    }
                }
            });
        }
        
        // Flush and close
        producer.flush();
        producer.close();
    }
}
```

### Node.js Producer

```javascript
const { Kafka } = require('kafkajs');

// Create Kafka instance
const kafka = new Kafka({
  clientId: 'my-app',
  brokers: ['localhost:9092']
});

// Create producer
const producer = kafka.producer();

async function run() {
  // Connect
  await producer.connect();
  
  // Send single message
  await producer.send({
    topic: 'my-topic',
    messages: [
      { key: 'key1', value: 'Hello Kafka' }
    ]
  });
  
  // Send multiple messages
  await producer.send({
    topic: 'user-events',
    messages: [
      { key: 'user-1', value: JSON.stringify({ action: 'login', timestamp: Date.now() }) },
      { key: 'user-2', value: JSON.stringify({ action: 'purchase', timestamp: Date.now() }) },
      { key: 'user-3', value: JSON.stringify({ action: 'logout', timestamp: Date.now() }) }
    ]
  });
  
  // Send batch
  await producer.sendBatch({
    topicMessages: [
      {
        topic: 'topic-1',
        messages: [{ key: 'key1', value: 'message1' }]
      },
      {
        topic: 'topic-2',
        messages: [{ key: 'key2', value: 'message2' }]
      }
    ]
  });
  
  // Disconnect
  await producer.disconnect();
}

run().catch(console.error);
```

## Consumers

### Python Consumer

```python
from kafka import KafkaConsumer
import json

# Create consumer
consumer = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  # 'earliest' or 'latest'
    enable_auto_commit=True,
    group_id='my-consumer-group',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

# Consume messages
print("Waiting for messages...")
for message in consumer:
    print(f"Topic: {message.topic}")
    print(f"Partition: {message.partition}")
    print(f"Offset: {message.offset}")
    print(f"Key: {message.key}")
    print(f"Value: {message.value}")
    print("---")

# Subscribe to multiple topics
consumer.subscribe(['topic-1', 'topic-2'])

# Subscribe with pattern
consumer.subscribe(pattern='user-.*')

# Manual commit
consumer = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    enable_auto_commit=False,
    group_id='my-group'
)

for message in consumer:
    process_message(message.value)
    # Commit after processing
    consumer.commit()

# Manual partition assignment
from kafka import TopicPartition

consumer = KafkaConsumer(bootstrap_servers=['localhost:9092'])
consumer.assign([TopicPartition('my-topic', 0)])

# Seek to specific offset
consumer.seek(TopicPartition('my-topic', 0), 100)

# Close consumer
consumer.close()
```

### Java Consumer

```java
import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.serialization.StringDeserializer;
import java.time.Duration;
import java.util.*;

public class SimpleConsumer {
    public static void main(String[] args) {
        // Configure consumer
        Properties props = new Properties();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "my-consumer-group");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class.getName());
        props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "true");
        
        // Create consumer
        KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
        
        // Subscribe to topic
        consumer.subscribe(Arrays.asList("my-topic"));
        
        try {
            while (true) {
                ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
                
                for (ConsumerRecord<String, String> record : records) {
                    System.out.printf("Topic: %s, Partition: %d, Offset: %d, Key: %s, Value: %s%n",
                        record.topic(), record.partition(), record.offset(), 
                        record.key(), record.value());
                }
            }
        } finally {
            consumer.close();
        }
    }
}
```

### Node.js Consumer

```javascript
const { Kafka } = require('kafkajs');

const kafka = new Kafka({
  clientId: 'my-app',
  brokers: ['localhost:9092']
});

const consumer = kafka.consumer({ groupId: 'my-consumer-group' });

async function run() {
  // Connect
  await consumer.connect();
  
  // Subscribe to topic
  await consumer.subscribe({ topic: 'my-topic', fromBeginning: true });
  
  // Process messages
  await consumer.run({
    eachMessage: async ({ topic, partition, message }) => {
      console.log({
        topic,
        partition,
        offset: message.offset,
        key: message.key?.toString(),
        value: message.value.toString()
      });
    }
  });
}

run().catch(console.error);

// Graceful shutdown
process.on('SIGTERM', async () => {
  await consumer.disconnect();
});
```

## Consumer Groups

### How They Work

```python
# Consumer 1 in group
consumer1 = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    group_id='my-group',
    auto_offset_reset='earliest'
)

# Consumer 2 in same group (will share partitions)
consumer2 = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    group_id='my-group',
    auto_offset_reset='earliest'
)

# Consumer in different group (will get all messages)
consumer3 = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    group_id='another-group',
    auto_offset_reset='earliest'
)
```

### Managing Consumer Groups

```bash
# List consumer groups
kafka-consumer-groups --bootstrap-server localhost:9092 --list

# Describe consumer group
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group my-consumer-group

# Reset offsets to earliest
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-consumer-group \
  --reset-offsets --to-earliest \
  --topic my-topic \
  --execute

# Reset offsets to specific offset
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-consumer-group \
  --reset-offsets --to-offset 100 \
  --topic my-topic:0 \
  --execute

# Delete consumer group
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --delete --group my-consumer-group
```

## Kafka Streams

### Word Count Example (Java)

```java
import org.apache.kafka.streams.*;
import org.apache.kafka.streams.kstream.*;
import java.util.Properties;
import java.util.Arrays;

public class WordCountApp {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "wordcount-app");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        
        StreamsBuilder builder = new StreamsBuilder();
        
        // Read from input topic
        KStream<String, String> textLines = builder.stream("input-topic");
        
        // Process
        KTable<String, Long> wordCounts = textLines
            .flatMapValues(textLine -> Arrays.asList(textLine.toLowerCase().split("\\W+")))
            .groupBy((key, word) -> word)
            .count();
        
        // Write to output topic
        wordCounts.toStream().to("output-topic");
        
        // Build and start
        KafkaStreams streams = new KafkaStreams(builder.build(), props);
        streams.start();
        
        // Shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(streams::close));
    }
}
```

### Stateful Processing

```java
import org.apache.kafka.streams.kstream.*;
import org.apache.kafka.streams.state.*;
import java.time.Duration;

// Windowed aggregation
KTable<Windowed<String>, Long> windowedCounts = textLines
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .count();

// Join streams
KStream<String, String> joined = leftStream
    .join(rightStream,
        (leftValue, rightValue) -> leftValue + ", " + rightValue,
        JoinWindows.of(Duration.ofMinutes(5))
    );

// State store
StoreBuilder<KeyValueStore<String, Long>> storeBuilder =
    Stores.keyValueStoreBuilder(
        Stores.persistentKeyValueStore("my-state-store"),
        Serdes.String(),
        Serdes.Long()
    );

builder.addStateStore(storeBuilder);
```

## Kafka Connect

### Configuration

Create `connect-standalone.properties`:

```properties
bootstrap.servers=localhost:9092
key.converter=org.apache.kafka.connect.json.JsonConverter
value.converter=org.apache.kafka.connect.json.JsonConverter
key.converter.schemas.enable=false
value.converter.schemas.enable=false
offset.storage.file.filename=/tmp/connect.offsets
offset.flush.interval.ms=10000
```

### File Source Connector

Create `file-source.properties`:

```properties
name=file-source
connector.class=FileStreamSource
tasks.max=1
file=/tmp/test.txt
topic=file-topic
```

```bash
# Run connector
connect-standalone.sh connect-standalone.properties file-source.properties
```

### JDBC Source Connector

```properties
name=jdbc-source
connector.class=io.confluent.connect.jdbc.JdbcSourceConnector
tasks.max=1
connection.url=jdbc:postgresql://localhost:5432/mydb
connection.user=myuser
connection.password=mypassword
table.whitelist=users,orders
mode=incrementing
incrementing.column.name=id
topic.prefix=postgres-
```

### REST API

```bash
# List connectors
curl http://localhost:8083/connectors

# Get connector status
curl http://localhost:8083/connectors/my-connector/status

# Create connector
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-connector",
    "config": {
      "connector.class": "FileStreamSource",
      "tasks.max": "1",
      "file": "/tmp/test.txt",
      "topic": "my-topic"
    }
  }'

# Delete connector
curl -X DELETE http://localhost:8083/connectors/my-connector
```

## Real-World Examples

### Log Aggregation System

```python
# Producer: Application logging
from kafka import KafkaProducer
import logging
import json

class KafkaHandler(logging.Handler):
    def __init__(self, bootstrap_servers, topic):
        logging.Handler.__init__(self)
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = topic
    
    def emit(self, record):
        log_entry = {
            'timestamp': record.created,
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'host': record.hostname if hasattr(record, 'hostname') else 'unknown'
        }
        self.producer.send(self.topic, value=log_entry)

# Setup logger
logger = logging.getLogger('myapp')
logger.addHandler(KafkaHandler(['localhost:9092'], 'app-logs'))
logger.setLevel(logging.INFO)

# Use logger
logger.info("Application started")
logger.error("An error occurred", exc_info=True)
```

```python
# Consumer: Log aggregator
from kafka import KafkaConsumer
from elasticsearch import Elasticsearch
import json

consumer = KafkaConsumer(
    'app-logs',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

es = Elasticsearch(['http://localhost:9200'])

for message in consumer:
    log_entry = message.value
    
    # Index in Elasticsearch
    es.index(
        index='application-logs',
        document=log_entry
    )
    
    # Alert on errors
    if log_entry['level'] == 'ERROR':
        send_alert(log_entry)
```

### Event Sourcing

```python
# Event store
from kafka import KafkaProducer
import json
from datetime import datetime

class EventStore:
    def __init__(self, bootstrap_servers):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def store_event(self, aggregate_id, event_type, data):
        event = {
            'aggregate_id': aggregate_id,
            'event_type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.producer.send(
            f'events-{aggregate_id}',
            key=aggregate_id.encode('utf-8'),
            value=event
        )
        
        return event

# Usage
event_store = EventStore(['localhost:9092'])

# Store events
event_store.store_event('order-123', 'OrderCreated', {'amount': 100})
event_store.store_event('order-123', 'OrderPaid', {'payment_method': 'card'})
event_store.store_event('order-123', 'OrderShipped', {'tracking': 'ABC123'})

# Rebuild state
from kafka import KafkaConsumer

def rebuild_order_state(order_id):
    consumer = KafkaConsumer(
        f'events-{order_id}',
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest'
    )
    
    state = {}
    for message in consumer:
        event = message.value
        state = apply_event(state, event)
    
    return state
```

### Real-Time Analytics

```python
# Stream processing for analytics
from kafka import KafkaConsumer, KafkaProducer
import json
from collections import defaultdict
from datetime import datetime, timedelta

consumer = KafkaConsumer(
    'user-events',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Window state
window_size = timedelta(minutes=5)
user_sessions = defaultdict(lambda: {'events': [], 'start_time': None})

for message in consumer:
    event = message.value
    user_id = event['user_id']
    timestamp = datetime.fromisoformat(event['timestamp'])
    
    session = user_sessions[user_id]
    
    # Start new session if needed
    if not session['start_time'] or \
       timestamp - session['start_time'] > window_size:
        # Emit previous session stats
        if session['events']:
            stats = {
                'user_id': user_id,
                'session_duration': (timestamp - session['start_time']).seconds,
                'event_count': len(session['events']),
                'events': session['events']
            }
            producer.send('user-session-stats', value=stats)
        
        # Reset session
        session['events'] = []
        session['start_time'] = timestamp
    
    # Add event to session
    session['events'].append(event)
```

## Performance Tuning

### Producer Configuration

```python
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    # Batching
    batch_size=16384,  # 16KB
    linger_ms=10,      # Wait up to 10ms for batching
    
    # Compression
    compression_type='snappy',  # or 'gzip', 'lz4', 'zstd'
    
    # Retries
    retries=3,
    max_in_flight_requests_per_connection=5,
    
    # Acknowledgments
    acks='all',  # or '0', '1', 'all'
    
    # Buffer
    buffer_memory=33554432,  # 32MB
)
```

### Consumer Configuration

```python
consumer = KafkaConsumer(
    'my-topic',
    bootstrap_servers=['localhost:9092'],
    # Fetching
    fetch_min_bytes=1024,  # Wait for at least 1KB
    fetch_max_wait_ms=500,  # or wait 500ms
    max_partition_fetch_bytes=1048576,  # 1MB per partition
    
    # Polling
    max_poll_records=500,
    max_poll_interval_ms=300000,  # 5 minutes
    
    # Offsets
    enable_auto_commit=True,
    auto_commit_interval_ms=5000,
)
```

### Broker Configuration

Edit `server.properties`:

```properties
# Replication
num.replica.fetchers=4
replica.lag.time.max.ms=10000

# Log retention
log.retention.hours=168  # 7 days
log.retention.bytes=1073741824  # 1GB per partition

# Log segments
log.segment.bytes=1073741824  # 1GB
log.segment.ms=604800000  # 7 days

# Compression
compression.type=producer  # Use producer's compression

# Memory
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600  # 100MB
```

## Security

### SSL Configuration

Generate certificates:

```bash
# Create CA
openssl req -new -x509 -keyout ca-key -out ca-cert -days 365

# Create broker keystore
keytool -keystore kafka.server.keystore.jks -alias localhost -validity 365 -genkey

# Create certificate signing request
keytool -keystore kafka.server.keystore.jks -alias localhost -certreq -file cert-file

# Sign certificate
openssl x509 -req -CA ca-cert -CAkey ca-key -in cert-file -out cert-signed -days 365 -CAcreateserial

# Import certificates
keytool -keystore kafka.server.keystore.jks -alias CARoot -import -file ca-cert
keytool -keystore kafka.server.keystore.jks -alias localhost -import -file cert-signed
```

Broker configuration:

```properties
listeners=SSL://localhost:9093
ssl.keystore.location=/var/private/ssl/kafka.server.keystore.jks
ssl.keystore.password=password
ssl.key.password=password
ssl.truststore.location=/var/private/ssl/kafka.server.truststore.jks
ssl.truststore.password=password
ssl.client.auth=required
```

### SASL Authentication

```properties
# Broker
listeners=SASL_SSL://localhost:9093
security.inter.broker.protocol=SASL_SSL
sasl.mechanism.inter.broker.protocol=PLAIN
sasl.enabled.mechanisms=PLAIN

# JAAS config
listener.name.sasl_ssl.plain.sasl.jaas.config=org.apache.kafka.common.security.plain.PlainLoginModule required \
  username="admin" \
  password="admin-secret" \
  user_admin="admin-secret" \
  user_alice="alice-secret";
```

## Monitoring

### JMX Metrics

```bash
# Enable JMX in Kafka
export KAFKA_JMX_OPTS="-Dcom.sun.management.jmxremote \
  -Dcom.sun.management.jmxremote.port=9999 \
  -Dcom.sun.management.jmxremote.authenticate=false \
  -Dcom.sun.management.jmxremote.ssl=false"
```

### Key Metrics

- **Under-replicated partitions**: Should be 0
- **Active controller count**: Should be 1
- **Request rate**: Requests per second
- **Byte in/out rate**: Throughput
- **Consumer lag**: Messages behind
- **Failed fetch requests**: Consumer issues

### Prometheus + Grafana

```yaml
# docker-compose.yml addition
  kafka-exporter:
    image: danielqsj/kafka-exporter
    command: --kafka.server=kafka:9093
    ports:
      - "9308:9308"
    depends_on:
      - kafka
```

## Troubleshooting

**Consumer lag:**
```bash
# Check consumer group lag
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group my-group

# Increase consumers in group
# Increase partitions
# Optimize consumer processing
```

**Under-replicated partitions:**
```bash
# Check topic status
kafka-topics --describe --bootstrap-server localhost:9092 --under-replicated-partitions

# Check broker logs
# Ensure enough brokers are running
# Check network connectivity
```

**Disk space issues:**
```bash
# Check disk usage
df -h

# Reduce retention
kafka-configs --bootstrap-server localhost:9092 \
  --alter --entity-type topics --entity-name my-topic \
  --add-config retention.ms=86400000  # 1 day

# Enable log compaction
kafka-configs --bootstrap-server localhost:9092 \
  --alter --entity-type topics --entity-name my-topic \
  --add-config cleanup.policy=compact
```

## Resources

- **Official Documentation**: https://kafka.apache.org/documentation/
- **Confluent Documentation**: https://docs.confluent.io/
- **Kafka Streams**: https://kafka.apache.org/documentation/streams/
- **Kafka Connect**: https://kafka.apache.org/documentation/#connect

## Quick Reference

### CLI Commands

```bash
# Topics
kafka-topics --create --bootstrap-server localhost:9092 --topic my-topic --partitions 3 --replication-factor 1
kafka-topics --list --bootstrap-server localhost:9092
kafka-topics --describe --bootstrap-server localhost:9092 --topic my-topic

# Producer
kafka-console-producer --bootstrap-server localhost:9092 --topic my-topic

# Consumer
kafka-console-consumer --bootstrap-server localhost:9092 --topic my-topic --from-beginning

# Consumer Groups
kafka-consumer-groups --bootstrap-server localhost:9092 --list
kafka-consumer-groups --bootstrap-server localhost:9092 --describe --group my-group

# Configs
kafka-configs --bootstrap-server localhost:9092 --describe --entity-type topics --entity-name my-topic
```

---

*This guide covers Apache Kafka fundamentals and advanced patterns for building real-time streaming applications.*
