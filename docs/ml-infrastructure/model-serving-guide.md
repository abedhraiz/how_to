# Model Serving & Deployment Guide

## Introduction

Model serving is the critical infrastructure that takes trained ML models from development to production, handling inference at scale with low latency, high throughput, and reliability.

**What You'll Learn:**
- Model serving framework comparison
- Production deployment patterns
- Performance optimization techniques
- Auto-scaling strategies
- Monitoring and debugging

---

## Table of Contents

1. [Framework Comparison](#framework-comparison)
2. [Model Packaging](#model-packaging)
3. [Deployment Patterns](#deployment-patterns)
4. [Performance Optimization](#performance-optimization)
5. [Production Examples](#production-examples)
6. [Monitoring](#monitoring)

---

## Framework Comparison

### Overview Matrix

| Framework | Best For | GPU Support | Language | Auto-Scaling | Complexity |
|-----------|----------|-------------|----------|--------------|------------|
| **TorchServe** | PyTorch models | Excellent | Python | Yes | Medium |
| **TensorFlow Serving** | TensorFlow models | Excellent | C++ | Yes | Low |
| **Triton** | Multi-framework | Excellent | C++ | Yes | High |
| **BentoML** | Any framework | Good | Python | Yes | Low |
| **MLflow** | Experimentation | Limited | Python | Via Kubernetes | Low |
| **FastAPI + Model** | Simple REST | Good | Python | Manual | Very Low |
| **Seldon Core** | Kubernetes-native | Excellent | Python/Java | Yes | High |

---

## TorchServe

```python
# TorchServe - Official PyTorch Serving

✅ Pros:
• Official PyTorch solution
• Good documentation
• Built-in model versioning
• Metrics and logging included
• Multi-model serving

❌ Cons:
• PyTorch-only
• Steep learning curve
• Limited flexibility
• Complex configuration

🎯 Use When:
• Serving PyTorch models
• Need official support
• Multi-model management
• Enterprise deployment
```

### Setup and Configuration

```bash
# Install TorchServe
pip install torchserve torch-model-archiver torch-workflow-archiver

# Create model archive
torch-model-archiver \
  --model-name resnet50 \
  --version 1.0 \
  --model-file model.py \
  --serialized-file resnet50.pth \
  --handler image_classifier \
  --extra-files index_to_name.json \
  --export-path model_store

# Start TorchServe
torchserve \
  --start \
  --model-store model_store \
  --models resnet50=resnet50.mar \
  --ts-config config.properties
```

### Custom Handler Example

```python
# custom_handler.py - Advanced TorchServe Handler

import torch
import logging
from ts.torch_handler.base_handler import BaseHandler
from typing import List, Dict
import json
import time

logger = logging.getLogger(__name__)

class CustomTransformerHandler(BaseHandler):
    """
    Custom handler for transformer models with:
    - Batching
    - Caching
    - Custom preprocessing
    - Detailed logging
    """
    
    def __init__(self):
        super().__init__()
        self.initialized = False
        self.cache = {}
        self.model = None
        self.tokenizer = None
    
    def initialize(self, context):
        """Initialize model and tokenizer"""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        
        # Load model
        logger.info("Loading model from %s", model_dir)
        self.model = torch.load(
            f"{model_dir}/model.pth",
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.eval()
        
        # Load tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Configure batching
        self.batch_size = int(properties.get("batch_size", 32))
        self.max_length = int(properties.get("max_length", 512))
        
        logger.info("Model loaded successfully")
        self.initialized = True
    
    def preprocess(self, requests: List) -> torch.Tensor:
        """Preprocess batch of requests"""
        texts = []
        
        for request in requests:
            data = request.get("data") or request.get("body")
            
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            
            if isinstance(data, str):
                data = json.loads(data)
            
            text = data.get("text", "")
            texts.append(text)
        
        # Tokenize batch
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoded.to(self.model.device)
    
    def inference(self, inputs):
        """Run inference with timing"""
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        inference_time = time.time() - start_time
        logger.info(f"Inference took {inference_time:.3f}s")
        
        return outputs
    
    def postprocess(self, outputs) -> List[Dict]:
        """Post-process model outputs"""
        # Convert to probabilities
        probabilities = torch.softmax(outputs.logits, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
        
        results = []
        for probs, indices in zip(top_probs, top_indices):
            predictions = [
                {
                    "label": self.mapping[str(idx.item())],
                    "score": prob.item()
                }
                for prob, idx in zip(probs, indices)
            ]
            results.append({"predictions": predictions})
        
        return results
    
    def handle(self, data, context):
        """Main handler method"""
        try:
            # Preprocess
            inputs = self.preprocess(data)
            
            # Inference
            outputs = self.inference(inputs)
            
            # Postprocess
            results = self.postprocess(outputs)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in handler: {str(e)}")
            raise
```

### Configuration File

```properties
# config.properties - TorchServe Configuration

# Inference address
inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082

# Model store
model_store=/models

# Logging
install_py_dep_per_model=true
enable_envvars_config=true

# Performance
number_of_netty_threads=32
job_queue_size=100
default_workers_per_model=4

# GPU
number_of_gpu=1
default_response_timeout=120

# Metrics
metrics_mode=prometheus
```

---

## TensorFlow Serving

```python
# TensorFlow Serving - High Performance Serving

✅ Pros:
• Excellent performance
• Battle-tested at scale
• gRPC and REST APIs
• Version management
• GPU optimization

❌ Cons:
• TensorFlow-only
• C++ codebase (harder to customize)
• Limited preprocessing
• Docker-heavy setup

🎯 Use When:
• Serving TensorFlow models
• Need maximum performance
• Simple preprocessing
• Large-scale production
```

### Docker Setup

```bash
# Save TensorFlow model in SavedModel format
import tensorflow as tf

# Train/load your model
model = tf.keras.applications.ResNet50(weights='imagenet')

# Export
export_path = '/models/resnet/1'  # Version 1
model.save(export_path, save_format='tf')

# Directory structure:
# /models/
#   resnet/
#     1/
#       saved_model.pb
#       variables/
```

```bash
# Run TensorFlow Serving with Docker
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/models,target=/models \
  -e MODEL_NAME=resnet \
  -t tensorflow/serving
```

### Client Code

```python
# TensorFlow Serving Client

import requests
import numpy as np
import json
from typing import List, Dict
from PIL import Image
import io

class TFServingClient:
    """Client for TensorFlow Serving"""
    
    def __init__(self, host: str = "localhost", port: int = 8501):
        self.base_url = f"http://{host}:{port}"
        self.model_name = None
    
    def set_model(self, model_name: str, version: int = None):
        """Set active model and version"""
        self.model_name = model_name
        self.version = version or "latest"
    
    def predict(self, inputs: np.ndarray) -> Dict:
        """Send prediction request"""
        
        url = f"{self.base_url}/v1/models/{self.model_name}/versions/{self.version}:predict"
        
        # Prepare payload
        payload = {
            "signature_name": "serving_default",
            "instances": inputs.tolist()
        }
        
        # Send request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 32
    ) -> List[Dict]:
        """Batch prediction with chunking"""
        
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_array = np.stack(batch)
            
            result = self.predict(batch_array)
            results.extend(result["predictions"])
        
        return results
    
    def get_model_metadata(self) -> Dict:
        """Get model metadata"""
        
        url = f"{self.base_url}/v1/models/{self.model_name}/metadata"
        response = requests.get(url)
        return response.json()
    
    def health_check(self) -> bool:
        """Check server health"""
        
        try:
            url = f"{self.base_url}/v1/models/{self.model_name}"
            response = requests.get(url)
            return response.status_code == 200
        except:
            return False

# Usage
client = TFServingClient(host="localhost", port=8501)
client.set_model("resnet", version=1)

# Single prediction
image = np.random.rand(1, 224, 224, 3)
result = client.predict(image)

# Batch prediction
images = [np.random.rand(224, 224, 3) for _ in range(100)]
results = client.predict_batch(images, batch_size=32)
```

---

## NVIDIA Triton Inference Server

```python
# Triton - Multi-Framework, High Performance

✅ Pros:
• Multi-framework (PyTorch, TF, ONNX)
• Excellent GPU optimization
• Dynamic batching
• Model ensemble support
• Concurrent model execution

❌ Cons:
• Complex setup
• Steep learning curve
• Heavy resource usage
• NVIDIA hardware preferred

🎯 Use When:
• Multiple frameworks
• Maximum GPU utilization
• Model ensembles
• High-throughput requirements
```

### Model Repository Setup

```bash
# Triton model repository structure
model_repository/
  bert_classifier/
    config.pbtxt
    1/
      model.plan  # TensorRT
  resnet50/
    config.pbtxt
    1/
      model.onnx  # ONNX
  ensemble_model/
    config.pbtxt
    1/
```

### Configuration Example

```python
# config.pbtxt - Triton Model Configuration

name: "bert_classifier"
platform: "pytorch_libtorch"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT32
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }
]

# Dynamic batching configuration
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}

# Instance groups (GPU configuration)
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0, 1 ]
  }
]

# Model optimization
optimization {
  cuda {
    graphs: true
    graph_spec {
      batch_size: 1
      input {
        key: "input_ids"
        value {
          dim: [ 128 ]
        }
      }
    }
  }
}
```

### Python Client

```python
# Triton Python Client

import tritonclient.http as httpclient
import numpy as np
from typing import List, Dict

class TritonClient:
    """High-performance Triton client"""
    
    def __init__(self, url: str = "localhost:8000"):
        self.client = httpclient.InferenceServerClient(url=url)
    
    def predict(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray],
        model_version: str = ""
    ) -> Dict[str, np.ndarray]:
        """Send inference request"""
        
        # Prepare inputs
        triton_inputs = []
        for name, data in inputs.items():
            input_obj = httpclient.InferInput(
                name,
                data.shape,
                datatype="FP32"  # or INT32, etc.
            )
            input_obj.set_data_from_numpy(data)
            triton_inputs.append(input_obj)
        
        # Prepare outputs
        triton_outputs = [
            httpclient.InferRequestedOutput("logits")
        ]
        
        # Execute
        response = self.client.infer(
            model_name,
            triton_inputs,
            model_version=model_version,
            outputs=triton_outputs
        )
        
        # Parse results
        results = {}
        for output in triton_outputs:
            results[output.name()] = response.as_numpy(output.name())
        
        return results
    
    def async_predict(
        self,
        model_name: str,
        inputs: Dict[str, np.ndarray]
    ):
        """Asynchronous inference"""
        
        triton_inputs = self._prepare_inputs(inputs)
        triton_outputs = [httpclient.InferRequestedOutput("logits")]
        
        # Async request with callback
        self.client.async_infer(
            model_name,
            triton_inputs,
            callback=self._callback,
            outputs=triton_outputs
        )
    
    def _callback(self, result, error):
        """Callback for async inference"""
        if error:
            print(f"Error: {error}")
        else:
            output = result.as_numpy("logits")
            print(f"Result: {output}")
    
    def get_model_metadata(self, model_name: str) -> Dict:
        """Get model configuration"""
        return self.client.get_model_metadata(model_name)
    
    def get_server_metrics(self) -> str:
        """Get Prometheus metrics"""
        return self.client.get_inference_statistics(model_name="")

# Usage
client = TritonClient("localhost:8000")

# Synchronous prediction
inputs = {
    "input_ids": np.array([[101, 2023, 2003, 1037, 3231, 102]], dtype=np.int32),
    "attention_mask": np.array([[1, 1, 1, 1, 1, 1]], dtype=np.int32)
}
results = client.predict("bert_classifier", inputs)
```

---

## BentoML

```python
# BentoML - Simple, Flexible Model Serving

✅ Pros:
• Framework-agnostic
• Easy to use
• Good documentation
• Built-in containerization
• Flexible preprocessing

❌ Cons:
• Newer/less mature
• Limited enterprise features
• Performance overhead
• Smaller community

🎯 Use When:
• Quick deployment
• Multiple frameworks
• Custom preprocessing
• Prototyping
```

### Service Definition

```python
# service.py - BentoML Service

import bentoml
from bentoml.io import JSON, NumpyNdarray
import numpy as np
from typing import Dict, List
from pydantic import BaseModel

# Define input schema
class PredictionInput(BaseModel):
    text: str
    max_length: int = 512

class PredictionOutput(BaseModel):
    label: str
    score: float
    probabilities: Dict[str, float]

# Load model
model_ref = bentoml.sklearn.get("sentiment_model:latest")
model_runner = model_ref.to_runner()

# Create service
svc = bentoml.Service("sentiment_classifier", runners=[model_runner])

@svc.api(
    input=JSON(pydantic_model=PredictionInput),
    output=JSON(pydantic_model=PredictionOutput),
    route="/predict"
)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """Predict sentiment of text"""
    
    # Preprocess
    processed = preprocess_text(input_data.text, input_data.max_length)
    
    # Run inference
    result = await model_runner.predict.async_run(processed)
    
    # Postprocess
    probabilities = {
        "positive": float(result[0]),
        "negative": float(result[1])
    }
    
    label = "positive" if result[0] > result[1] else "negative"
    score = max(result)
    
    return PredictionOutput(
        label=label,
        score=score,
        probabilities=probabilities
    )

@svc.api(
    input=JSON(),
    output=JSON(),
    route="/batch_predict"
)
async def batch_predict(input_data: List[Dict]) -> List[Dict]:
    """Batch prediction endpoint"""
    
    texts = [item["text"] for item in input_data]
    processed = [preprocess_text(text, 512) for text in texts]
    
    # Batch inference
    results = await model_runner.predict.async_run(np.array(processed))
    
    outputs = []
    for result in results:
        outputs.append({
            "label": "positive" if result[0] > result[1] else "negative",
            "score": float(max(result))
        })
    
    return outputs

def preprocess_text(text: str, max_length: int) -> np.ndarray:
    """Preprocess text for model"""
    # Your preprocessing logic
    return np.array([text[:max_length]])
```

### Deployment

```bash
# Build Bento
bentoml build

# Serve locally
bentoml serve service:svc --reload

# Containerize
bentoml containerize sentiment_classifier:latest

# Deploy to cloud
bentoml deploy sentiment_classifier:latest --platform=aws
```

---

## Performance Optimization

### Dynamic Batching

```python
# Dynamic Batching Implementation

import asyncio
from typing import List, Dict, Callable
import time
from dataclasses import dataclass
from queue import Queue
import numpy as np

@dataclass
class BatchItem:
    """Single item in batch"""
    id: str
    input_data: np.ndarray
    future: asyncio.Future
    timestamp: float

class DynamicBatcher:
    """
    Dynamic batching for model serving
    Accumulates requests and processes in optimized batches
    """
    
    def __init__(
        self,
        model_fn: Callable,
        max_batch_size: int = 32,
        max_wait_time: float = 0.1,  # 100ms
        preferred_batch_sizes: List[int] = None
    ):
        self.model_fn = model_fn
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.preferred_batch_sizes = preferred_batch_sizes or [8, 16, 32]
        
        self.queue: List[BatchItem] = []
        self.processing = False
        self.lock = asyncio.Lock()
    
    async def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Add request to batch queue"""
        
        # Create batch item
        item = BatchItem(
            id=str(time.time()),
            input_data=input_data,
            future=asyncio.Future(),
            timestamp=time.time()
        )
        
        # Add to queue
        async with self.lock:
            self.queue.append(item)
            
            # Trigger batch processing if needed
            if not self.processing:
                asyncio.create_task(self._process_batches())
        
        # Wait for result
        return await item.future
    
    async def _process_batches(self):
        """Process batches from queue"""
        
        self.processing = True
        
        try:
            while True:
                # Wait for queue to fill or timeout
                await asyncio.sleep(0.01)  # 10ms check interval
                
                async with self.lock:
                    if not self.queue:
                        break
                    
                    # Determine batch size
                    batch_size = self._get_optimal_batch_size(len(self.queue))
                    
                    # Check if should wait longer
                    oldest_item = self.queue[0]
                    wait_time = time.time() - oldest_item.timestamp
                    
                    if len(self.queue) < batch_size and wait_time < self.max_wait_time:
                        continue  # Wait for more items
                    
                    # Extract batch
                    batch = self.queue[:batch_size]
                    self.queue = self.queue[batch_size:]
                
                # Process batch
                await self._execute_batch(batch)
                
        finally:
            self.processing = False
    
    def _get_optimal_batch_size(self, queue_size: int) -> int:
        """Select optimal batch size"""
        
        for size in sorted(self.preferred_batch_sizes, reverse=True):
            if queue_size >= size:
                return size
        
        return min(queue_size, self.max_batch_size)
    
    async def _execute_batch(self, batch: List[BatchItem]):
        """Execute model inference on batch"""
        
        try:
            # Stack inputs
            inputs = np.stack([item.input_data for item in batch])
            
            # Run model
            start_time = time.time()
            outputs = await self.model_fn(inputs)
            latency = time.time() - start_time
            
            # Distribute results
            for item, output in zip(batch, outputs):
                item.future.set_result(output)
            
            # Log metrics
            print(f"Processed batch of {len(batch)} in {latency:.3f}s "
                  f"({len(batch)/latency:.1f} req/s)")
            
        except Exception as e:
            # Propagate error to all items
            for item in batch:
                item.future.set_exception(e)

# Usage example
async def model_inference(inputs: np.ndarray) -> np.ndarray:
    """Simulated model inference"""
    await asyncio.sleep(0.05)  # Simulate GPU time
    return inputs * 2  # Dummy output

batcher = DynamicBatcher(
    model_fn=model_inference,
    max_batch_size=32,
    max_wait_time=0.1
)

# Make predictions
async def make_prediction(data):
    result = await batcher.predict(data)
    return result

# Concurrent requests
async def main():
    tasks = [
        make_prediction(np.random.rand(10))
        for _ in range(100)
    ]
    results = await asyncio.gather(*tasks)
    print(f"Completed {len(results)} predictions")

asyncio.run(main())
```

### Model Optimization

```python
# Model Optimization Techniques

import torch
import onnx
from torch import nn

class ModelOptimizer:
    """Optimize models for inference"""
    
    @staticmethod
    def quantize_pytorch(model: nn.Module, calibration_data=None):
        """Dynamic quantization (INT8)"""
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},  # Layers to quantize
            dtype=torch.qint8
        )
        
        return quantized_model
    
    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str
    ):
        """Export to ONNX format"""
        
        model.eval()
        
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to {output_path}")
    
    @staticmethod
    def optimize_onnx(model_path: str, output_path: str):
        """Optimize ONNX model"""
        
        import onnx
        from onnxruntime.transformers import optimizer
        
        # Load model
        model = onnx.load(model_path)
        
        # Optimize
        optimized_model = optimizer.optimize_model(
            model_path,
            model_type='bert',
            num_heads=12,
            hidden_size=768
        )
        
        # Save
        optimized_model.save_model_to_file(output_path)
    
    @staticmethod
    def compile_to_tensorrt(
        onnx_path: str,
        output_path: str,
        max_batch_size: int = 32
    ):
        """Compile to TensorRT for NVIDIA GPUs"""
        
        import tensorrt as trt
        
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            parser.parse(f.read())
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        # Optimization profile
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input",
            min=(1, 128),
            opt=(16, 128),
            max=(max_batch_size, 128)
        )
        config.add_optimization_profile(profile)
        
        # Serialize
        engine = builder.build_engine(network, config)
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {output_path}")

# Usage
optimizer = ModelOptimizer()

# Quantize model
quantized = optimizer.quantize_pytorch(original_model)

# Export to ONNX
optimizer.export_to_onnx(
    model=original_model,
    dummy_input=torch.randn(1, 3, 224, 224),
    output_path="model.onnx"
)

# Compile to TensorRT
optimizer.compile_to_tensorrt(
    onnx_path="model.onnx",
    output_path="model.trt",
    max_batch_size=32
)
```

---

## Auto-Scaling

### Kubernetes HPA with Custom Metrics

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: server
        image: your-model-server:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
# HPA based on custom metrics
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: inference_queue_length
      target:
        type: AverageValue
        averageValue: "10"
  - type: Pods
    pods:
      metric:
        name: p95_latency_ms
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

---

## Monitoring

```python
# Production Monitoring Setup

from prometheus_client import Counter, Histogram, Gauge
import time
from functools import wraps
from typing import Callable

# Define metrics
INFERENCE_COUNTER = Counter(
    'model_inference_total',
    'Total number of inference requests',
    ['model_name', 'model_version', 'status']
)

INFERENCE_LATENCY = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency',
    ['model_name', 'model_version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

BATCH_SIZE = Histogram(
    'model_batch_size',
    'Batch size distribution',
    ['model_name'],
    buckets=[1, 2, 4, 8, 16, 32, 64, 128]
)

ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Number of active inference requests',
    ['model_name']
)

def monitor_inference(model_name: str, model_version: str):
    """Decorator to monitor inference calls"""
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Track active requests
            ACTIVE_REQUESTS.labels(model_name=model_name).inc()
            
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                # Record metrics
                latency = time.time() - start_time
                
                INFERENCE_COUNTER.labels(
                    model_name=model_name,
                    model_version=model_version,
                    status=status
                ).inc()
                
                INFERENCE_LATENCY.labels(
                    model_name=model_name,
                    model_version=model_version
                ).observe(latency)
                
                ACTIVE_REQUESTS.labels(model_name=model_name).dec()
        
        return wrapper
    return decorator

# Usage
@monitor_inference(model_name="resnet50", model_version="v1")
async def predict(input_data):
    # Your inference code
    result = await model.predict(input_data)
    
    # Track batch size
    batch_size = len(input_data)
    BATCH_SIZE.labels(model_name="resnet50").observe(batch_size)
    
    return result
```

---

## Production Checklist

```markdown
✅ Model Serving Framework
  □ Framework selected based on requirements
  □ Performance benchmarked
  □ Resource requirements defined
  □ Cost calculated

✅ Model Packaging
  □ Model exported in serving format
  □ Dependencies documented
  □ Version strategy defined
  □ Rollback plan ready

✅ Performance
  □ Dynamic batching configured
  □ GPU utilization optimized
  □ Model quantized/optimized
  □ Load tested at scale

✅ Auto-Scaling
  □ HPA configured
  □ Metrics defined
  □ Scale limits set
  □ Scale-down policy tuned

✅ Monitoring
  □ Latency tracked
  □ Throughput monitored
  □ Error rate tracked
  □ Resource usage monitored
  □ Alerts configured

✅ Operations
  □ Health checks implemented
  □ Graceful shutdown configured
  □ Model warm-up strategy
  □ Deployment pipeline ready
  □ Rollback procedure tested
```

---

*This guide covers model serving. For ML lifecycle management, see [MLOps guides](../data-engineering/ml-ops/).*
