# API Specification

## Purpose

Define the interface for model serving, specifying inputs, outputs, and behavior for integration with downstream systems.

## 1. API Design

### RESTful API Endpoints

#### Prediction Endpoint

```
POST /api/v1/predict
Content-Type: application/json

Request Body:
{
  "features": {
    "age": 35,
    "income": 75000,
    "credit_score": 750,
    "tenure": 5
  }
}

Response (200 OK):
{
  "prediction": "approved",
  "probability": 0.87,
  "model_version": "v1.0.0",
  "inference_time_ms": 45,
  "timestamp": "2024-01-01T12:00:00Z"
}

Error Response (400 Bad Request):
{
  "error": "Missing required field: age",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Batch Prediction Endpoint

```
POST /api/v1/predict-batch
Content-Type: application/json

Request Body:
{
  "predictions": [
    {
      "id": "user_1",
      "features": {"age": 35, "income": 75000}
    },
    {
      "id": "user_2",
      "features": {"age": 42, "income": 95000}
    }
  ]
}

Response (200 OK):
{
  "results": [
    {
      "id": "user_1",
      "prediction": "approved",
      "probability": 0.87
    },
    {
      "id": "user_2",
      "prediction": "approved",
      "probability": 0.92
    }
  ],
  "batch_size": 2,
  "total_time_ms": 120
}
```

#### Model Info Endpoint

```
GET /api/v1/model/info

Response (200 OK):
{
  "name": "credit_approval_model",
  "version": "1.0.0",
  "algorithm": "GradientBoostingClassifier",
  "features": ["age", "income", "credit_score", "tenure"],
  "classes": ["approved", "denied"],
  "performance": {
    "accuracy": 0.92,
    "f1_score": 0.87
  },
  "last_updated": "2024-01-01T10:00:00Z"
}
```

#### Health Check Endpoint

```
GET /health

Response (200 OK):
{
  "status": "healthy",
  "model_loaded": true,
  "database_connected": true,
  "uptime_seconds": 3600
}

Response (503 Service Unavailable):
{
  "status": "unhealthy",
  "reason": "Model loading failed"
}
```

## 2. Request/Response Schema

### Input Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["features"],
  "properties": {
    "features": {
      "type": "object",
      "required": ["age", "income", "credit_score"],
      "properties": {
        "age": {
          "type": "integer",
          "minimum": 18,
          "maximum": 120
        },
        "income": {
          "type": "number",
          "minimum": 0
        },
        "credit_score": {
          "type": "integer",
          "minimum": 300,
          "maximum": 850
        }
      }
    }
  }
}
```

### Output Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "prediction": {
      "type": "string",
      "enum": ["approved", "denied"]
    },
    "probability": {
      "type": "number",
      "minimum": 0,
      "maximum": 1
    },
    "model_version": {
      "type": "string"
    }
  }
}
```

## 3. Error Handling

### Error Response Format

```python
class APIError:
    def __init__(self, error_code, message, status_code, details=None):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.details = details
        self.timestamp = datetime.now().isoformat()

# Example errors
VALIDATION_ERROR = APIError(
    'VALIDATION_ERROR',
    'Input validation failed',
    400,
    {'field': 'age', 'reason': 'must be >= 18'}
)

MODEL_ERROR = APIError(
    'MODEL_ERROR',
    'Model inference failed',
    500,
    {'reason': 'Out of memory'}
)

NOT_FOUND_ERROR = APIError(
    'NOT_FOUND',
    'Model version not found',
    404,
    {'version': 'v2.0.0'}
)
```

## 4. Authentication & Security

### API Key Authentication

```python
@app.middleware
def validate_api_key(request):
    api_key = request.headers.get('X-API-Key')
    if not api_key or not is_valid_key(api_key):
        return {
            'error': 'Unauthorized',
            'message': 'Invalid or missing API key'
        }, 401
```

### Rate Limiting

```python
# Limit requests per minute
RATE_LIMIT = 1000  # requests per minute per API key

def check_rate_limit(api_key):
    current_count = redis_client.incr(f'api_key:{api_key}')
    if current_count > RATE_LIMIT:
        return False
    redis_client.expire(f'api_key:{api_key}', 60)
    return True
```

### Input Validation

```python
def validate_request(request_data):
    """Validate incoming request"""
    errors = []
    
    if 'features' not in request_data:
        errors.append("Missing 'features' field")
    
    features = request_data.get('features', {})
    
    # Validate each feature
    if 'age' not in features or not (18 <= features['age'] <= 120):
        errors.append("age must be between 18 and 120")
    
    if 'income' in features and features['income'] < 0:
        errors.append("income cannot be negative")
    
    if errors:
        raise ValidationError('\n'.join(errors))
```

## 5. Performance Requirements

### SLA Definition

```
Endpoint: /api/v1/predict
P50 Latency: < 50ms
P99 Latency: < 200ms
Availability: 99.9%
Max Throughput: 1000 req/sec
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def predict_with_cache(features_hash):
    # Cache predictions for identical inputs
    # Reduces latency for repeated requests
    pass
```

## 6. Implementation Examples

### Flask

```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data.get('features')
        
        # Validate
        if not features:
            return {'error': 'Missing features'}, 400
        
        # Predict
        prediction = model.predict([features])
        probability = model.predict_proba([features])
        
        return {
            'prediction': prediction[0],
            'probability': float(probability[0][1]),
            'model_version': 'v1.0.0'
        }, 200
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class Features(BaseModel):
    age: int
    income: float
    credit_score: int

class Prediction(BaseModel):
    prediction: str
    probability: float
    model_version: str

@app.post('/api/v1/predict', response_model=Prediction)
def predict(features: Features):
    prediction = model.predict([[features.age, features.income, features.credit_score]])
    probability = model.predict_proba([[features.age, features.income, features.credit_score]])
    
    return Prediction(
        prediction=prediction[0],
        probability=float(probability[0][1]),
        model_version='v1.0.0'
    )

@app.get('/health')
def health():
    return {'status': 'healthy'}
```

## 7. API Documentation

### OpenAPI/Swagger Spec

```yaml
openapi: 3.0.0
info:
  title: Credit Approval Model API
  version: 1.0.0
servers:
  - url: https://api.example.com

paths:
  /api/v1/predict:
    post:
      summary: Get prediction
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                features:
                  type: object
      responses:
        '200':
          description: Successful prediction
          content:
            application/json:
              schema:
                type: object
                properties:
                  prediction:
                    type: string
                  probability:
                    type: number
```

## 8. API Versioning

### Version Management

```
Current: /api/v1/ (Stable)
Beta: /api/v2-beta/ (Testing)
Deprecated: /api/v0/ (Sunset in 3 months)
```

### Backward Compatibility

- Always support previous major version
- Announce deprecation 3 months ahead
- Provide migration guide
- Maintain changelog

## Best Practices

1. ✅ Validate all inputs
2. ✅ Clear error messages
3. ✅ Version your API
4. ✅ Document thoroughly
5. ✅ Monitor performance
6. ✅ Rate limit
7. ✅ Log all requests
8. ✅ Test extensively

---

## Related Documents

- [Infrastructure](./infrastructure.md) - Deployment infrastructure
- [Monitoring Plan](./monitoring-plan.md) - API monitoring

---

*Well-designed APIs enable reliable, scalable model serving*
