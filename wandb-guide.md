# Weights & Biases (W&B) Guide

## What is Weights & Biases?

Weights & Biases (W&B) is a machine learning platform for experiment tracking, model management, dataset versioning, and collaboration. It helps ML practitioners track experiments, visualize results, and collaborate with teams.

## Key Features

- **Experiment Tracking** - Log metrics, hyperparameters, and outputs
- **Visualizations** - Interactive charts and dashboards
- **Model Registry** - Version and manage models
- **Dataset Versioning** - Track dataset changes
- **Sweeps** - Hyperparameter optimization
- **Reports** - Share findings with stakeholders
- **Collaboration** - Team workspaces and sharing

## Installation

```bash
# Install W&B
pip install wandb

# Verify installation
wandb --version
```

## Setup and Authentication

### Login

```bash
# Interactive login
wandb login

# Login with API key
wandb login YOUR_API_KEY

# Set API key via environment variable
export WANDB_API_KEY=YOUR_API_KEY
```

Get your API key from: https://wandb.ai/authorize

### Configuration

```python
import wandb

# Initialize W&B
wandb.init(
    project="my-project",
    entity="my-team",  # Optional: team name
    name="experiment-1",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }
)
```

## Basic Usage

### Simple Example

```python
import wandb
import random

# Initialize run
wandb.init(project="my-first-project")

# Log metrics
for epoch in range(10):
    loss = random.random()
    accuracy = random.random()
    
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })

# Finish run
wandb.finish()
```

### PyTorch Example

```python
import torch
import torch.nn as nn
import wandb

# Initialize W&B
wandb.init(
    project="pytorch-demo",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32,
        "architecture": "CNN"
    }
)

# Access config
config = wandb.config

# Define model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(config.epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics
        if batch_idx % 100 == 0:
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "loss": loss.item()
            })
    
    # Validation
    val_loss, val_acc = validate(model, val_loader)
    wandb.log({
        "epoch": epoch,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })

# Save model
torch.save(model.state_dict(), "model.pth")
wandb.save("model.pth")

wandb.finish()
```

### TensorFlow/Keras Example

```python
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

# Initialize W&B
wandb.init(
    project="keras-demo",
    config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 32
    }
)

config = wandb.config

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with W&B callback
model.fit(
    x_train, y_train,
    epochs=config.epochs,
    batch_size=config.batch_size,
    validation_data=(x_val, y_val),
    callbacks=[WandbCallback()]
)

wandb.finish()
```

## Advanced Features

### 1. Logging Different Data Types

#### Metrics

```python
# Simple metrics
wandb.log({"loss": 0.5, "accuracy": 0.95})

# With custom step
wandb.log({"loss": 0.5}, step=100)

# Multiple metrics
wandb.log({
    "train/loss": 0.5,
    "train/accuracy": 0.95,
    "val/loss": 0.6,
    "val/accuracy": 0.93
})
```

#### Images

```python
import numpy as np
from PIL import Image

# Log image from array
image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
wandb.log({"example": wandb.Image(image, caption="Random Image")})

# Log PIL image
pil_image = Image.open("sample.png")
wandb.log({"sample": wandb.Image(pil_image)})

# Log multiple images
wandb.log({
    "examples": [
        wandb.Image(img1, caption="Image 1"),
        wandb.Image(img2, caption="Image 2")
    ]
})

# Log image with masks (segmentation)
wandb.log({
    "predictions": wandb.Image(
        image,
        masks={
            "predictions": {"mask_data": pred_mask},
            "ground_truth": {"mask_data": gt_mask}
        }
    )
})
```

#### Tables

```python
# Create table
table = wandb.Table(columns=["id", "prediction", "truth", "correct"])

for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
    table.add_data(i, pred, truth, pred == truth)

wandb.log({"predictions_table": table})

# Table with images
table = wandb.Table(columns=["image", "prediction", "confidence"])
for img, pred, conf in zip(images, predictions, confidences):
    table.add_data(wandb.Image(img), pred, conf)

wandb.log({"results": table})
```

#### Histograms

```python
# Log histogram
wandb.log({"gradients": wandb.Histogram(gradient_values)})

# Log model weights
for name, param in model.named_parameters():
    wandb.log({f"weights/{name}": wandb.Histogram(param.data.cpu())})
```

#### Audio

```python
# Log audio file
wandb.log({"audio": wandb.Audio("sample.wav", caption="Sample Audio")})

# Log from array
import numpy as np
audio_array = np.random.randn(44100)  # 1 second at 44.1kHz
wandb.log({"generated": wandb.Audio(audio_array, sample_rate=44100)})
```

#### Video

```python
# Log video file
wandb.log({"video": wandb.Video("sample.mp4", fps=30)})

# Log from numpy array
video_array = np.random.randint(0, 255, (100, 3, 64, 64), dtype=np.uint8)
wandb.log({"animation": wandb.Video(video_array, fps=10)})
```

#### 3D Objects

```python
# Log 3D point cloud
wandb.log({
    "point_cloud": wandb.Object3D(
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    )
})

# Log 3D mesh
wandb.log({"mesh": wandb.Object3D.from_file("model.obj")})
```

### 2. Hyperparameter Sweeps

#### Define Sweep Configuration

```python
sweep_config = {
    'method': 'bayes',  # grid, random, bayes
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [16, 32, 64, 128]
        },
        'epochs': {
            'value': 10
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        }
    }
}
```

#### Create and Run Sweep

```python
import wandb

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="my-project")

# Define training function
def train():
    # Initialize run
    wandb.init()
    
    # Get hyperparameters
    config = wandb.config
    
    # Build model with config
    model = build_model(
        learning_rate=config.learning_rate,
        dropout=config.dropout
    )
    
    # Train and log
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, config.batch_size)
        val_acc = validate(model)
        
        wandb.log({
            "train_loss": train_loss,
            "val_accuracy": val_acc
        })

# Run sweep
wandb.agent(sweep_id, function=train, count=20)
```

#### Sweep via YAML

Create `sweep.yaml`:

```yaml
program: train.py
method: bayes
metric:
  name: val_accuracy
  goal: maximize
parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.00001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  epochs:
    value: 10
```

Run sweep:
```bash
wandb sweep sweep.yaml
wandb agent YOUR_SWEEP_ID
```

### 3. Artifacts (Model & Dataset Versioning)

#### Log Artifacts

```python
import wandb

run = wandb.init(project="my-project")

# Create artifact
artifact = wandb.Artifact(
    name="my-dataset",
    type="dataset",
    description="Training dataset v1.0"
)

# Add files
artifact.add_file("train.csv")
artifact.add_dir("data/images/")

# Add reference (doesn't upload)
artifact.add_reference("s3://my-bucket/data/")

# Log artifact
run.log_artifact(artifact)

wandb.finish()
```

#### Use Artifacts

```python
import wandb

run = wandb.init(project="my-project")

# Download and use artifact
artifact = run.use_artifact('my-dataset:latest', type='dataset')
artifact_dir = artifact.download()

# Load data
import pandas as pd
data = pd.read_csv(f"{artifact_dir}/train.csv")

wandb.finish()
```

#### Model Versioning

```python
import wandb

run = wandb.init(project="my-project")

# Train model
model = train_model()

# Save model
model.save("model.h5")

# Create model artifact
model_artifact = wandb.Artifact(
    name="my-model",
    type="model",
    description="CNN classifier",
    metadata={
        "architecture": "CNN",
        "dataset": "CIFAR-10",
        "accuracy": 0.95
    }
)

model_artifact.add_file("model.h5")
run.log_artifact(model_artifact)

wandb.finish()
```

#### Link Artifacts to Model Registry

```python
# Link to registry
run.link_artifact(
    artifact=model_artifact,
    target_path="my-team/model-registry/my-model"
)

# Set alias
artifact.aliases = ["production", "v1.0"]
```

### 4. Custom Charts

```python
# Line plot
wandb.log({"custom_plot": wandb.plot.line(
    table=wandb.Table(data=data, columns=["x", "y"]),
    x="x",
    y="y",
    title="Custom Line Plot"
)})

# Scatter plot
wandb.log({"scatter": wandb.plot.scatter(
    table=wandb.Table(data=data, columns=["x", "y", "label"]),
    x="x",
    y="y",
    title="Scatter Plot"
)})

# Bar chart
wandb.log({"bar": wandb.plot.bar(
    table=wandb.Table(data=data, columns=["category", "value"]),
    label="category",
    value="value",
    title="Bar Chart"
)})

# Confusion matrix
wandb.log({"conf_mat": wandb.plot.confusion_matrix(
    probs=None,
    y_true=ground_truth,
    preds=predictions,
    class_names=["cat", "dog", "bird"]
)})

# ROC curve
wandb.log({"roc": wandb.plot.roc_curve(
    y_true=ground_truth,
    y_probas=probabilities,
    labels=class_names
)})

# PR curve
wandb.log({"pr": wandb.plot.pr_curve(
    y_true=ground_truth,
    y_probas=probabilities,
    labels=class_names
)})
```

### 5. System Metrics

```python
# Log GPU/CPU metrics automatically
wandb.init(
    project="my-project",
    monitor_gym=True  # Log gym environment metrics
)

# Manual system logging
import psutil
import GPUtil

wandb.log({
    "cpu_percent": psutil.cpu_percent(),
    "memory_percent": psutil.virtual_memory().percent,
    "gpu_utilization": GPUtil.getGPUs()[0].load * 100
})
```

## Integration Examples

### Hugging Face Transformers

```python
from transformers import Trainer, TrainingArguments
import wandb

wandb.init(project="transformers-demo")

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",
    run_name="bert-finetuning",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
wandb.finish()
```

### LightGBM

```python
import lightgbm as lgb
import wandb
from wandb.lightgbm import wandb_callback, log_summary

wandb.init(project="lightgbm-demo")

# Train with callback
gbm = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    callbacks=[wandb_callback()]
)

# Log feature importance
log_summary(gbm, save_model_checkpoint=True)

wandb.finish()
```

### XGBoost

```python
import xgboost as xgb
import wandb
from wandb.xgboost import wandb_callback

wandb.init(project="xgboost-demo")

# Train with callback
model = xgb.train(
    params,
    dtrain,
    evals=[(dval, "validation")],
    callbacks=[wandb_callback()]
)

wandb.finish()
```

### Scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import wandb

wandb.init(project="sklearn-demo")

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Log metrics
wandb.log({
    "accuracy": accuracy,
    "feature_importance": wandb.plot.bar(
        wandb.Table(
            data=[[name, imp] for name, imp in zip(feature_names, clf.feature_importances_)],
            columns=["feature", "importance"]
        ),
        "feature", "importance"
    )
})

# Log model
import joblib
joblib.dump(clf, "model.pkl")
wandb.save("model.pkl")

wandb.finish()
```

### Ray Tune

```python
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback

def train_function(config):
    wandb.init(
        project="ray-tune",
        config=config,
        reinit=True
    )
    
    # Your training code
    for epoch in range(10):
        loss = train_epoch(config)
        wandb.log({"loss": loss})
        tune.report(loss=loss)

analysis = tune.run(
    train_function,
    config={
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32, 64])
    },
    callbacks=[WandbLoggerCallback(project="ray-tune")]
)
```

## Command Line Interface

```bash
# Initialize project
wandb init

# Login
wandb login

# Pull artifacts
wandb artifact get project/artifact:version

# Run script with W&B
wandb run python train.py

# Sync offline runs
wandb sync

# Restore files from run
wandb restore model.h5 --run wandb/project/run_id

# Pull code from run
wandb pull wandb/project/run_id

# List projects
wandb projects

# Verify setup
wandb verify
```

## Offline Mode

```python
import os

# Disable W&B (for testing)
os.environ["WANDB_MODE"] = "disabled"

# Offline mode (sync later)
os.environ["WANDB_MODE"] = "offline"

# Or in init
wandb.init(mode="offline")

# Sync later
# wandb sync ./wandb/offline-run-xxx
```

## Best Practices

### 1. Organization

```python
# Use consistent naming
wandb.init(
    project="image-classification",
    entity="ml-team",
    name=f"experiment-{model_name}-{timestamp}",
    tags=["baseline", "resnet50", "production"],
    notes="Testing with augmentation"
)

# Group related runs
wandb.init(
    project="my-project",
    group="experiment-1",
    job_type="train"
)
```

### 2. Configuration Management

```python
# Save all hyperparameters
config = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model": "resnet50",
    "dataset": "CIFAR-10",
    "augmentation": True,
    "optimizer": "adam"
}

wandb.init(project="my-project", config=config)

# Update config
wandb.config.update({"new_param": "value"})
```

### 3. Error Handling

```python
try:
    wandb.init(project="my-project")
    
    # Your training code
    train()
    
except Exception as e:
    # Log error
    wandb.log({"error": str(e)})
    raise
finally:
    # Always finish
    wandb.finish()
```

### 4. Memory Management

```python
# Log less frequently
log_interval = 100
for step in range(total_steps):
    if step % log_interval == 0:
        wandb.log({"loss": loss})

# Use define_metric for custom x-axis
wandb.define_metric("epoch")
wandb.define_metric("train/*", step_metric="epoch")
wandb.define_metric("val/*", step_metric="epoch")
```

### 5. Reproducibility

```python
import random
import numpy as np
import torch

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Log seed
wandb.init(
    project="my-project",
    config={"seed": 42}
)

set_seed(wandb.config.seed)

# Log code
wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
```

## Reports

Create reports in W&B UI to:
- Document experiments
- Share findings with team
- Create presentations
- Track progress over time

### Embed Visualizations

```python
# Get run URL
print(f"View run at: {wandb.run.get_url()}")

# Get specific chart URL
# Use in reports or documentation
```

## API Usage

```python
import wandb

api = wandb.Api()

# Get runs from project
runs = api.runs("entity/project")

# Filter runs
runs = api.runs(
    "entity/project",
    filters={"config.learning_rate": 0.001}
)

# Get specific run
run = api.run("entity/project/run_id")

# Access data
print(run.summary)  # Final metrics
print(run.config)   # Hyperparameters
print(run.history())  # All logged data

# Download files
for file in run.files():
    file.download()

# Export data
import pandas as pd
history_df = run.history()
history_df.to_csv("run_history.csv")
```

## Pricing

- **Free Tier** - 100GB storage, unlimited runs
- **Academic** - Free unlimited for students/researchers
- **Teams** - Starting at $50/month
- **Enterprise** - Custom pricing

## Troubleshooting

### Common Issues

```python
# Run not showing up
# Make sure wandb.finish() is called

# Slow logging
# Reduce logging frequency
wandb.log(metrics, commit=False)  # Buffer logs
wandb.log({}, commit=True)  # Flush buffer

# API rate limits
# Add delays between API calls
import time
time.sleep(1)

# Authentication errors
# Re-login
wandb.login(relogin=True)
```

## Resources

- **Website:** https://wandb.ai
- **Documentation:** https://docs.wandb.ai
- **Examples:** https://github.com/wandb/examples
- **Community:** https://wandb.ai/community
- **YouTube:** W&B tutorials and webinars
- **Discord:** Community support

## Quick Reference

| Feature | Code |
|---------|------|
| Initialize | `wandb.init(project="name")` |
| Log metrics | `wandb.log({"metric": value})` |
| Log image | `wandb.log({"img": wandb.Image(img)})` |
| Log table | `wandb.log({"table": wandb.Table(...)})` |
| Save file | `wandb.save("file.txt")` |
| Log artifact | `run.log_artifact(artifact)` |
| Use artifact | `run.use_artifact("name:version")` |
| Create sweep | `wandb.sweep(config)` |
| Run agent | `wandb.agent(sweep_id, function)` |
| Finish run | `wandb.finish()` |

---

*Weights & Biases is a powerful tool for ML experiment tracking and collaboration. Start with simple logging and gradually adopt advanced features as needed.*
