# AI/ML Model Formats - Comprehensive Guide

## Purpose

A comprehensive reference for understanding, comparing, and working with different AI/ML model formats. This guide focuses on interoperability, optimization, and production deployment considerations, with special emphasis on ONNX and TensorRT.

---

## Table of Contents

- [Overview](#overview)
- [Format Comparison Table](#format-comparison-table)
- [Detailed Format Analysis](#detailed-format-analysis)
- [ONNX Deep Dive](#onnx-deep-dive)
- [TensorRT Deep Dive](#tensorrt-deep-dive)
- [Conversion Pipelines](#conversion-pipelines)
- [Toolchain Comparison](#toolchain-comparison)
- [Use Case Scenarios](#use-case-scenarios)
- [Best Practices](#best-practices)
- [Code Examples](#code-examples)

---

## Overview

### Why Model Formats Matter

Machine learning models can be represented in various formats, each optimized for different purposes:

- **Training Formats** - Preserve training state, optimizers, gradients
- **Inference Formats** - Optimized for deployment, smaller size
- **Interoperability Formats** - Transfer models between frameworks
- **Hardware-Specific Formats** - Optimized for particular hardware (GPUs, TPUs, Edge devices)

### Key Considerations

When choosing a model format, consider:

1. **Target Hardware** - CPU, GPU, Mobile, Edge devices
2. **Framework Compatibility** - Single framework or cross-framework
3. **Performance Requirements** - Latency, throughput, memory
4. **Deployment Environment** - Cloud, edge, mobile, embedded
5. **Optimization Needs** - Quantization, pruning, hardware acceleration
6. **Maintenance** - Support, updates, community

---

## Format Comparison Table

| Format | Maintainer | Primary Use | Strengths | Limitations | Best For |
|--------|------------|-------------|-----------|-------------|----------|
| **ONNX** | Linux Foundation AI | Interoperability | • Cross-framework support<br>• Wide hardware support<br>• Active community<br>• Standardized operators | • Some custom ops missing<br>• May need optimization for specific hardware | • Multi-framework pipelines<br>• Framework migration<br>• Model sharing<br>• Cross-platform deployment |
| **TensorRT** | NVIDIA | GPU Inference Optimization | • Exceptional GPU performance<br>• Layer fusion optimization<br>• INT8/FP16 precision<br>• Dynamic tensor memory | • NVIDIA GPUs only<br>• Complex API<br>• Version compatibility issues | • High-performance GPU inference<br>• Real-time applications<br>• Batch processing<br>• NVIDIA hardware |
| **TensorFlow SavedModel** | Google | TensorFlow Production | • Complete graph + variables<br>• Version management<br>• Serving signatures<br>• TensorFlow ecosystem | • TensorFlow specific<br>• Larger file size<br>• Limited cross-framework | • TF production deployments<br>• TF Serving<br>• Cloud deployment<br>• Model versioning |
| **TorchScript** | Meta (PyTorch) | PyTorch Production | • Python-like syntax<br>• Dynamic + static graphs<br>• JIT compilation<br>• Mobile deployment | • PyTorch ecosystem only<br>• Learning curve<br>• Some dynamic features limited | • PyTorch production<br>• Research to production<br>• Mobile deployment<br>• C++ integration |
| **Core ML** | Apple | iOS/macOS Deployment | • Apple hardware optimization<br>• Neural Engine support<br>• On-device ML<br>• Privacy-focused | • Apple ecosystem only<br>• macOS/iOS versions<br>• Format complexity | • iPhone/iPad apps<br>• macOS applications<br>• Apple Silicon optimization<br>• On-device inference |
| **TensorFlow Lite** | Google | Mobile/Edge Deployment | • Small model size<br>• Quantization support<br>• Mobile optimized<br>• Cross-platform mobile | • Limited operator support<br>• Reduced accuracy<br>• Complex conversion | • Android applications<br>• IoT devices<br>• Edge computing<br>• Resource-constrained devices |
| **OpenVINO IR** | Intel | Intel Hardware Optimization | • Intel CPU/GPU/VPU support<br>• Inference optimization<br>• Model optimizer<br>• Heterogeneous execution | • Intel hardware focus<br>• Steep learning curve<br>• Limited mobile support | • Intel CPU deployment<br>• Edge AI on Intel<br>• Movidius VPU<br>• Mixed hardware inference |
| **MMdnn** | Microsoft | Cross-Framework Conversion | • Multi-framework support<br>• Conversion utilities<br>• Visualization tools | • Maintenance concerns<br>• Limited updates<br>• Complex dependencies | • One-time migrations<br>• Framework exploration<br>• Legacy model conversion |

---

## Detailed Format Analysis

### 1. ONNX (Open Neural Network Exchange)

#### Overview
ONNX is an open format for representing machine learning models, enabling interoperability between frameworks.

#### Architecture
```
Framework Model (PyTorch/TensorFlow/etc.)
            ↓
    ONNX Exporter
            ↓
    ONNX Graph (IR)
    ├── Nodes (Operators)
    ├── Initializers (Weights)
    ├── Inputs/Outputs
    └── Metadata
            ↓
    ONNX Runtime / TensorRT / etc.
            ↓
    Optimized Inference
```

#### Key Features
- **Operator Set**: Standardized set of operators versioned independently
- **Graph Representation**: Computational graph with nodes and edges
- **Opset Versioning**: Backward compatibility through operator set versions
- **Extensibility**: Custom operators support

#### Supported Frameworks
| Framework | Export | Import | Maturity |
|-----------|--------|--------|----------|
| PyTorch | ✅ Native | ✅ | Excellent |
| TensorFlow | ✅ via tf2onnx | ✅ | Good |
| Keras | ✅ via tf2onnx | ✅ | Good |
| Scikit-learn | ✅ via skl2onnx | ❌ | Good |
| XGBoost | ✅ via onnxmltools | ❌ | Fair |
| LightGBM | ✅ via onnxmltools | ❌ | Fair |

#### Inference Runtimes
- **ONNX Runtime** (Microsoft) - Cross-platform, CPU/GPU
- **TensorRT** (NVIDIA) - GPU optimization
- **OpenVINO** (Intel) - Intel hardware
- **Core ML** (Apple) - iOS/macOS
- **NNAPI** (Google) - Android

#### File Structure
```
model.onnx
├── IR Version: 7
├── Producer: pytorch v1.12
├── Opset: 13
├── Graph
│   ├── Input: [1, 3, 224, 224] (float32)
│   ├── Nodes: 150 operators
│   ├── Initializers: 50MB weights
│   └── Output: [1, 1000] (float32)
└── Metadata
    ├── model_version: 1
    └── description: "ResNet50 classifier"
```

---

### 2. TensorRT

#### Overview
NVIDIA's SDK for high-performance deep learning inference, optimizing models specifically for NVIDIA GPUs.

#### Architecture
```
Model (ONNX/TF/PyTorch)
        ↓
TensorRT Builder
        ↓
Optimization Pipeline
├── Layer Fusion
├── Precision Calibration
├── Kernel Auto-tuning
└── Memory Optimization
        ↓
Serialized Engine (.trt)
        ↓
TensorRT Runtime
        ↓
Optimized GPU Inference
```

#### Optimization Techniques

**1. Layer Fusion**
```
Before:
Conv → BatchNorm → ReLU (3 separate layers)

After (Fused):
Conv+BN+ReLU (1 optimized kernel)
```

**2. Precision Calibration**
- **FP32**: Full precision (baseline)
- **FP16**: Half precision (~2x speedup)
- **INT8**: Integer quantization (~4x speedup, requires calibration)

**3. Kernel Auto-tuning**
- Tests multiple implementations
- Selects fastest kernel for specific GPU
- Hardware-specific optimization

**4. Dynamic Tensor Memory**
- Reuses memory buffers
- Reduces memory footprint
- Faster memory allocation

#### TensorRT Workflow
```python
# High-level workflow
1. Parse model (ONNX/UFF/Caffe)
2. Build optimization profile
3. Build engine with optimizations
4. Serialize engine to .trt file
5. Deploy with TensorRT Runtime
```

#### Performance Characteristics

| Precision | Relative Speed | Accuracy Loss | Use Case |
|-----------|----------------|---------------|----------|
| FP32 | 1x (baseline) | 0% | High accuracy required |
| FP16 | ~2x | <1% | Good speed/accuracy balance |
| INT8 | ~4x | 1-3% | Latency-critical apps |

#### Hardware Requirements
- **Minimum**: CUDA-capable NVIDIA GPU
- **Recommended**: Tensor Core GPUs (V100, T4, A100, RTX series)
- **CUDA**: Version compatibility with TensorRT version
- **Memory**: Sufficient for model + optimization process

---

### 3. TensorFlow SavedModel

#### Overview
TensorFlow's native format for production deployment, containing complete computational graph and trained parameters.

#### Structure
```
saved_model/
├── saved_model.pb          # Model graph
├── variables/
│   ├── variables.data-*    # Model weights
│   └── variables.index     # Weight index
└── assets/                 # Additional files
    └── vocab.txt           # Example asset
```

#### Key Components
- **MetaGraphDef**: Complete graph definition
- **SignatureDef**: Input/output specifications
- **Variables**: Trained weights
- **Assets**: Vocabulary files, configs

#### Serving Signatures
```python
# Multiple signatures in one model
signatures = {
    'predict': prediction_signature,
    'classify': classification_signature,
    'regress': regression_signature
}
```

---

### 4. TorchScript

#### Overview
PyTorch's format for deploying models outside Python, enabling C++ inference and mobile deployment.

#### Compilation Methods

**Tracing**
```python
import torch

# Trace with example input
example = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example)
traced_model.save('model_traced.pt')
```

**Scripting**
```python
# Script for dynamic control flow
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')
```

#### Comparison
| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Tracing | Simple, fast | No control flow | Feed-forward networks |
| Scripting | Full Python support | More complex | Models with if/loops |

---

### 5. Core ML

#### Overview
Apple's framework for integrating ML models into iOS, macOS, watchOS, and tvOS applications.

#### Features
- **Neural Engine**: Apple Silicon optimization
- **On-Device**: Privacy-preserving inference
- **Model Encryption**: IP protection
- **Update Workflow**: Over-the-air model updates

#### Model Types
- **Neural Networks**: CNNs, RNNs, Transformers
- **Tree Ensembles**: Random forests, boosted trees
- **Pipelines**: Multi-model workflows
- **Updatable Models**: On-device learning

---

### 6. TensorFlow Lite

#### Overview
Lightweight TensorFlow for mobile and embedded devices with focus on efficiency.

#### Optimization Features
```
Original Model
    ↓
Quantization (INT8)
    ↓
Operator Fusion
    ↓
Pruning (optional)
    ↓
TFLite Model (.tflite)
```

#### Quantization Types
- **Post-training quantization**: No retraining needed
- **Quantization-aware training**: Better accuracy
- **Dynamic range quantization**: Weights only
- **Integer quantization**: Full INT8

---

### 7. OpenVINO IR

#### Overview
Intel's intermediate representation optimized for Intel hardware (CPU, GPU, VPU, FPGA).

#### Optimization Pipeline
```
Model (TF/PyTorch/ONNX)
        ↓
Model Optimizer
    ├── Graph transformations
    ├── Layer fusion
    └── Precision optimization
        ↓
IR Format (.xml + .bin)
        ↓
Inference Engine
        ↓
Intel Hardware
```

#### Supported Hardware
- Intel CPUs (x86/x64)
- Intel GPUs (Integrated/Discrete)
- Intel Movidius VPUs
- Intel FPGAs (HDDL)

---

## ONNX Deep Dive

### Architecture Details

#### Graph Representation
```
Graph
├── Nodes (Operations)
│   ├── Op Type: "Conv"
│   ├── Inputs: ["input", "conv_weight"]
│   ├── Outputs: ["conv_output"]
│   └── Attributes: {kernel_shape: [3,3], pads: [1,1,1,1]}
├── Initializers (Weights)
│   └── conv_weight: [64, 3, 3, 3] (float32)
├── Inputs
│   └── input: [N, 3, 224, 224]
└── Outputs
    └── output: [N, 1000]
```

### Operator Sets (Opsets)

ONNX uses versioned operator sets for backward compatibility:

| Opset | Release | Key Changes |
|-------|---------|-------------|
| 13 | 2021 | Sequence operations, Einsum |
| 14 | 2021 | Training ops, HardSwish |
| 15 | 2022 | Optional inputs, Bernoulli |
| 16 | 2022 | Grid sample, RoiAlign improvements |
| 17 | 2023 | LayerNormalization updates |

### Converting to ONNX

#### PyTorch to ONNX
```python
import torch
import torch.onnx

# Load PyTorch model
model = torch.load('model.pth')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

#### TensorFlow to ONNX
```python
import tf2onnx
import tensorflow as tf

# Load TensorFlow model
model = tf.keras.models.load_model('model.h5')

# Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = "model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=output_path
)
```

### ONNX Runtime Optimization

```python
import onnxruntime as ort
import numpy as np

# Create inference session with optimizations
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

# Execution providers (prioritized)
providers = [
    'CUDAExecutionProvider',    # GPU
    'CPUExecutionProvider'       # CPU fallback
]

session = ort.InferenceSession(
    "model.onnx",
    sess_options=sess_options,
    providers=providers
)

# Run inference
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
result = session.run([output_name], {input_name: input_data})
```

### ONNX Simplification

```python
import onnx
from onnxsim import simplify

# Load model
model = onnx.load("model.onnx")

# Simplify
model_simplified, check = simplify(model)

if check:
    onnx.save(model_simplified, "model_simplified.onnx")
    print("Model simplified successfully")
```

---

## TensorRT Deep Dive

### Building TensorRT Engines

#### Method 1: From ONNX
```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine_from_onnx(onnx_file, engine_file, precision='fp16'):
    """Build TensorRT engine from ONNX model"""
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = 4 * (1 << 30)  # 4GB
    
    # Set precision
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        # Calibrator needed for INT8
        config.int8_calibrator = MyCalibrator()
    
    # Build engine
    print(f"Building TensorRT engine with {precision} precision...")
    engine = builder.build_engine(network, config)
    
    # Serialize and save
    with open(engine_file, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Engine saved to {engine_file}")
    return engine
```

#### Method 2: From PyTorch (torch2trt)
```python
from torch2trt import torch2trt
import torch

# Load PyTorch model
model = torch.load('model.pth').cuda().eval()

# Create example input
x = torch.ones((1, 3, 224, 224)).cuda()

# Convert to TensorRT
model_trt = torch2trt(
    model,
    [x],
    fp16_mode=True,
    max_workspace_size=1<<30  # 1GB
)

# Save
torch.save(model_trt.state_dict(), 'model_trt.pth')
```

### TensorRT Inference

```python
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np

class TRTInference:
    def __init__(self, engine_path):
        # Load engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data):
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Copy output from device
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        
        return self.outputs[0]['host']

# Usage
engine = TRTInference('model.trt')
result = engine.infer(input_data)
```

### INT8 Calibration

```python
import tensorrt as trt

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_dataset, cache_file='calibration.cache'):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.dataset = calibration_dataset
        self.current_index = 0
        
        # Allocate device memory for batch
        self.device_input = cuda.mem_alloc(self.dataset[0].nbytes)
    
    def get_batch_size(self):
        return 1
    
    def get_batch(self, names):
        if self.current_index < len(self.dataset):
            batch = self.dataset[self.current_index]
            cuda.memcpy_htod(self.device_input, batch)
            self.current_index += 1
            return [int(self.device_input)]
        return None
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
```

### Dynamic Shapes

```python
# Create optimization profile for dynamic shapes
profile = builder.create_optimization_profile()

# Define shape ranges: min, optimal, max
profile.set_shape(
    "input",
    min=(1, 3, 224, 224),
    opt=(8, 3, 224, 224),
    max=(32, 3, 224, 224)
)

config.add_optimization_profile(profile)
```

---

## Conversion Pipelines

### Common Conversion Paths

#### 1. PyTorch → ONNX → TensorRT
```
PyTorch Model (.pth/.pt)
        ↓ torch.onnx.export()
    ONNX (.onnx)
        ↓ trtexec / Python API
    TensorRT Engine (.trt)
        ↓
High-Performance GPU Inference
```

**Complete Pipeline:**
```python
import torch
import torch.onnx
import tensorrt as trt

# Step 1: PyTorch to ONNX
model = torch.load('pytorch_model.pth')
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

# Step 2: ONNX to TensorRT
# Use build_engine_from_onnx() function from earlier
engine = build_engine_from_onnx("model.onnx", "model.trt", precision='fp16')
```

#### 2. TensorFlow → ONNX → TensorRT
```
TensorFlow SavedModel / Keras (.h5)
        ↓ tf2onnx
    ONNX (.onnx)
        ↓ TensorRT
    TensorRT Engine (.trt)
```

**Pipeline:**
```bash
# Step 1: TensorFlow to ONNX
python -m tf2onnx.convert \
    --saved-model tensorflow_model/ \
    --output model.onnx \
    --opset 13

# Step 2: ONNX to TensorRT
trtexec \
    --onnx=model.onnx \
    --saveEngine=model.trt \
    --fp16
```

#### 3. PyTorch → TorchScript → Mobile
```
PyTorch Model
    ↓ torch.jit.trace/script
TorchScript (.pt)
    ↓ optimize_for_mobile
Mobile-Optimized TorchScript
    ↓
iOS/Android Deployment
```

#### 4. TensorFlow → TFLite → Mobile
```
TensorFlow Model
    ↓ TFLiteConverter
TensorFlow Lite (.tflite)
    ↓ Quantization
Optimized TFLite
    ↓
Android/iOS/IoT
```

#### 5. Any Framework → Core ML
```
PyTorch/TensorFlow
        ↓
    ONNX (.onnx)
        ↓ onnx-coreml
    Core ML (.mlmodel)
        ↓
Apple Device Deployment
```

### Multi-Target Conversion Strategy

```
                Training Framework
                (PyTorch/TensorFlow)
                        ↓
                    ONNX (Hub)
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
    TensorRT        OpenVINO        Core ML
    (NVIDIA GPU)    (Intel CPU)     (Apple)
        ↓               ↓               ↓
    Cloud Inference  Edge Devices   Mobile Apps
```

---

## Toolchain Comparison

### 1. ONNX Runtime

**Overview**: Cross-platform inference engine supporting ONNX models.

**Key Features:**
- CPU and GPU support
- Quantization and graph optimization
- Multiple execution providers
- Cross-platform (Windows, Linux, macOS, mobile)

**Execution Providers:**
```python
providers = [
    'TensorrtExecutionProvider',   # NVIDIA GPU
    'CUDAExecutionProvider',        # NVIDIA GPU (CUDA)
    'DnnlExecutionProvider',        # Intel CPU (oneDNN)
    'OpenVINOExecutionProvider',    # Intel hardware
    'CoreMLExecutionProvider',      # Apple devices
    'CPUExecutionProvider'          # Fallback
]
```

**Performance:**
- Good baseline performance
- Hardware-specific optimizations via providers
- Lower latency than native frameworks

**Use Cases:**
- Cross-platform deployment
- Multi-hardware support
- Quick model serving
- Framework-agnostic inference

---

### 2. TensorRT

**Overview**: NVIDIA's high-performance deep learning inference SDK.

**Key Features:**
- Extreme GPU optimization
- Low latency (microseconds)
- Mixed precision (FP16, INT8)
- Dynamic shapes support

**Performance Optimization:**
```
Speedup vs Native:
- FP32: 2-3x
- FP16: 3-5x
- INT8: 5-10x

Memory Reduction:
- FP16: ~50%
- INT8: ~75%
```

**Hardware Requirements:**
- NVIDIA GPU (Kepler or newer)
- CUDA Toolkit
- cuDNN library

**Limitations:**
- NVIDIA-only
- Complex API
- Build time overhead
- Version compatibility

**Use Cases:**
- Real-time inference
- High-throughput serving
- Video processing
- Autonomous systems

---

### 3. OpenVINO

**Overview**: Intel's toolkit for optimizing models for Intel hardware.

**Key Features:**
- Intel CPU optimization (AVX, AVX512)
- Intel GPU support (integrated/discrete)
- VPU support (Movidius)
- Model optimizer for graph transformation

**Supported Hardware:**
- Intel CPUs (Core, Xeon)
- Intel integrated GPUs
- Intel Movidius VPUs
- Intel Gaussian & Neural Accelerators

**Performance:**
- 10-20x speedup on Intel CPUs vs unoptimized
- 5-10x on integrated GPUs
- 20-40x on VPUs (specific workloads)

**Workflow:**
```bash
# Convert model to IR
python mo.py \
    --input_model model.onnx \
    --output_dir openvino_model/

# Benchmark
python benchmark_app.py \
    -m openvino_model/model.xml \
    -d CPU
```

**Use Cases:**
- Edge AI on Intel devices
- Desktop applications
- Industrial IoT
- Retail analytics

---

### 4. TVM (Apache)

**Overview**: End-to-end compiler stack for deploying ML models on various hardware.

**Key Features:**
- Hardware-agnostic compiler
- Auto-tuning for optimization
- Multiple backend support
- Graph-level and operator-level optimization

**Supported Backends:**
- CPUs (x86, ARM)
- GPUs (CUDA, OpenCL, Vulkan)
- FPGAs
- Custom accelerators

**Workflow:**
```python
import tvm
from tvm import relay

# Load ONNX model
onnx_model = onnx.load('model.onnx')
mod, params = relay.frontend.from_onnx(onnx_model)

# Auto-tune for target hardware
target = "cuda"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# Run inference
module = tvm.contrib.graph_executor.GraphModule(lib['default'](tvm.cuda()))
```

**Use Cases:**
- Custom hardware deployment
- Research and experimentation
- Performance optimization
- Multi-backend support

---

## Use Case Scenarios

### Scenario 1: High-Performance Cloud Inference (NVIDIA GPUs)

**Requirements:**
- Low latency (<10ms)
- High throughput (>1000 req/sec)
- GPU infrastructure

**Recommended Stack:**
```
PyTorch Model
    ↓
ONNX (validation)
    ↓
TensorRT (FP16/INT8)
    ↓
Triton Inference Server
```

**Rationale:**
- TensorRT provides best GPU performance
- Triton handles model serving
- ONNX as intermediate validation step

---

### Scenario 2: Cross-Platform ML Application

**Requirements:**
- Deploy on Cloud (AWS, GCP, Azure)
- Support CPU and GPU inference
- Framework flexibility

**Recommended Stack:**
```
Training Framework (Any)
    ↓
ONNX (universal format)
    ↓
ONNX Runtime (multi-provider)
```

**Rationale:**
- ONNX Runtime supports multiple hardware
- Single model format for all platforms
- Good performance without vendor lock-in

---

### Scenario 3: Mobile App Deployment

**Requirements:**
- On-device inference
- Small model size (<50MB)
- Low power consumption

**iOS:**
```
PyTorch/TensorFlow
    ↓
ONNX
    ↓
Core ML
    ↓
iOS App
```

**Android:**
```
TensorFlow/PyTorch
    ↓
TensorFlow Lite (quantized)
    ↓
Android App
```

---

### Scenario 4: Edge Device (IoT/Raspberry Pi)

**Requirements:**
- Limited compute (ARM CPU)
- Low memory (<1GB)
- Real-time constraints

**Recommended Stack:**
```
Training Framework
    ↓
TensorFlow Lite (INT8)
or
ONNX → OpenVINO (Intel devices)
    ↓
Edge Device
```

**Optimizations:**
- INT8 quantization (4x smaller)
- Pruning (remove redundant weights)
- Knowledge distillation (smaller model)

---

## Best Practices

### 1. Model Format Selection

#### Decision Tree
```
Need GPU performance?
    Yes → NVIDIA hardware?
        Yes → TensorRT
        No → ONNX Runtime (CUDA provider)
    No → Intel hardware?
        Yes → OpenVINO
        No → Mobile device?
            Yes → iOS? → Core ML
                 Android? → TFLite
            No → ONNX Runtime (CPU)
```

### 2. Conversion Best Practices

#### Pre-Conversion Checklist
✅ Verify model works in source framework  
✅ Document input/output shapes and dtypes  
✅ Test with various batch sizes  
✅ Check for custom operators  
✅ Verify dynamic shape support if needed

#### Post-Conversion Validation
```python
import numpy as np

# Compare outputs
def validate_conversion(original_model, converted_model, test_input):
    """Validate converted model against original"""
    
    # Original prediction
    orig_output = original_model(test_input)
    
    # Converted prediction
    conv_output = converted_model(test_input)
    
    # Compare
    diff = np.abs(orig_output - conv_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    # Tolerance check
    assert max_diff < 1e-3, "Conversion validation failed"
    
    return True
```

### 3. Optimization Strategy

#### Progressive Optimization
```
1. Baseline (FP32)
   ↓ (Validate accuracy)
2. FP16 (Mixed Precision)
   ↓ (Check accuracy loss <1%)
3. INT8 (Quantization)
   ↓ (Check accuracy loss <3%)
4. Pruning/Distillation
   ↓ (If needed for size/speed)
5. Deploy Best Version
```

### 4. Performance Benchmarking

```python
import time
import numpy as np

def benchmark_model(model, input_shape, num_iterations=100):
    """Benchmark inference performance"""
    
    # Warmup
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(10):
        _ = model(dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start = time.time()
        _ = model(dummy_input)
        times.append(time.time() - start)
    
    # Statistics
    times = np.array(times) * 1000  # Convert to ms
    print(f"Mean latency: {np.mean(times):.2f} ms")
    print(f"Std latency: {np.std(times):.2f} ms")
    print(f"P50 latency: {np.percentile(times, 50):.2f} ms")
    print(f"P95 latency: {np.percentile(times, 95):.2f} ms")
    print(f"P99 latency: {np.percentile(times, 99):.2f} ms")
    print(f"Throughput: {1000/np.mean(times):.2f} infer/sec")
```

### 5. Version Management

```
model_registry/
├── resnet50/
│   ├── v1.0/
│   │   ├── pytorch/
│   │   │   └── model.pth
│   │   ├── onnx/
│   │   │   └── model.onnx
│   │   ├── tensorrt/
│   │   │   ├── model_fp32.trt
│   │   │   ├── model_fp16.trt
│   │   │   └── model_int8.trt
│   │   └── metadata.json
│   └── v2.0/
│       └── ...
└── README.md
```

---

## Code Examples

### Complete Conversion Pipeline

```python
"""
Complete pipeline: PyTorch → ONNX → TensorRT
with validation and benchmarking
"""

import torch
import torch.onnx
import onnx
import onnxruntime as ort
import tensorrt as trt
import numpy as np
import time

class ModelConverter:
    def __init__(self, pytorch_model, input_shape, model_name="model"):
        self.model = pytorch_model
        self.input_shape = input_shape
        self.model_name = model_name
        
    def pytorch_to_onnx(self, opset_version=13):
        """Convert PyTorch to ONNX"""
        print("Converting PyTorch to ONNX...")
        
        self.model.eval()
        dummy_input = torch.randn(*self.input_shape)
        
        onnx_path = f"{self.model_name}.onnx"
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Validate ONNX
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model saved to {onnx_path}")
        
        return onnx_path
    
    def onnx_to_tensorrt(self, onnx_path, precision='fp16'):
        """Convert ONNX to TensorRT"""
        print(f"Converting ONNX to TensorRT ({precision})...")
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 4 * (1 << 30)  # 4GB
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Save engine
        trt_path = f"{self.model_name}_{precision}.trt"
        with open(trt_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"✓ TensorRT engine saved to {trt_path}")
        return trt_path
    
    def validate_conversion(self, onnx_path, test_inputs=10):
        """Validate ONNX conversion"""
        print("Validating conversion...")
        
        # PyTorch inference
        self.model.eval()
        test_data = [torch.randn(*self.input_shape) for _ in range(test_inputs)]
        pytorch_outputs = []
        
        with torch.no_grad():
            for data in test_data:
                pytorch_outputs.append(self.model(data).numpy())
        
        # ONNX Runtime inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_outputs = []
        
        for data in test_data:
            onnx_out = session.run(None, {input_name: data.numpy()})
            onnx_outputs.append(onnx_out[0])
        
        # Compare
        for i, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
            diff = np.abs(pt_out - onnx_out)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            if max_diff > 1e-3:
                print(f"⚠ Test {i}: Large difference detected!")
                print(f"  Max diff: {max_diff}")
                print(f"  Mean diff: {mean_diff}")
            
        print("✓ Validation complete")
    
    def benchmark(self, model_path, model_type='onnx', iterations=100):
        """Benchmark model performance"""
        print(f"\nBenchmarking {model_type.upper()} model...")
        
        if model_type == 'onnx':
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            test_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: test_input})
            
            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.time()
                session.run(None, {input_name: test_input})
                times.append(time.time() - start)
        
        # Print stats
        times = np.array(times) * 1000
        print(f"Mean: {np.mean(times):.2f} ms")
        print(f"Std: {np.std(times):.2f} ms")
        print(f"P95: {np.percentile(times, 95):.2f} ms")
        print(f"Throughput: {1000/np.mean(times):.2f} infer/sec")
    
    def convert_all(self):
        """Run complete conversion pipeline"""
        print(f"\n{'='*60}")
        print(f"Starting conversion pipeline for {self.model_name}")
        print(f"{'='*60}\n")
        
        # Step 1: PyTorch to ONNX
        onnx_path = self.pytorch_to_onnx()
        
        # Step 2: Validate
        self.validate_conversion(onnx_path)
        
        # Step 3: ONNX to TensorRT
        trt_fp16 = self.onnx_to_tensorrt(onnx_path, precision='fp16')
        
        # Step 4: Benchmark
        print("\n" + "="*60)
        print("Performance Benchmarks")
        print("="*60)
        self.benchmark(onnx_path, model_type='onnx')
        
        print(f"\n{'='*60}")
        print("Conversion pipeline complete!")
        print(f"{'='*60}\n")
        
        return {
            'onnx': onnx_path,
            'tensorrt_fp16': trt_fp16
        }

# Usage example
if __name__ == "__main__":
    import torchvision.models as models
    
    # Load model
    model = models.resnet50(pretrained=True)
    
    # Convert
    converter = ModelConverter(
        pytorch_model=model,
        input_shape=(1, 3, 224, 224),
        model_name="resnet50"
    )
    
    paths = converter.convert_all()
    print(f"\nGenerated files:")
    for format_type, path in paths.items():
        print(f"  {format_type}: {path}")
```

---

## Related Documentation

- **[Feature Engineering Guide](./feature-engineering-guide.md)** - Prepare data for models
- **[W&B Guide](./wandb-guide.md)** - Experiment tracking and model versioning
- **[AI Lifecycle](../ai-lifecycle/README.md)** - Complete ML project workflow
- **[Kubernetes Guide](../../infrastructure-devops/kubernetes/kubernetes-guide.md)** - Deploy models at scale
- **[Monitoring Guide](../../monitoring-observability/prometheus-grafana/prometheus-grafana-guide.md)** - Monitor model performance

---

## Additional Resources

### Official Documentation
- [ONNX Documentation](https://onnx.ai/onnx/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [TensorFlow Model Optimization](https://www.tensorflow.org/model_optimization)

### Tools & Utilities
- [Netron](https://netron.app/) - Model visualization
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier) - Graph optimization
- [Polygraphy](https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy) - TensorRT debugging

### Community
- [ONNX GitHub](https://github.com/onnx/onnx)
- [TensorRT Forum](https://forums.developer.nvidia.com/c/accelerated-computing/deep-learning/tensorrt/)
- [ONNX Runtime GitHub](https://github.com/microsoft/onnxruntime)

---

**Last Updated**: December 2025  
**Maintainers**: ML Operations Team  
**Version**: 1.0
