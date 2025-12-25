# ML Testing & Validation Guide

## Introduction

Testing machine learning systems requires testing both traditional software components and ML-specific elements like data quality, model performance, and behavioral correctness.

**What You'll Learn:**
- Testing strategies for ML systems
- Data validation frameworks
- Model testing patterns
- Integration testing approaches
- CI/CD for ML

---

## Table of Contents

1. [Testing Pyramid for ML](#testing-pyramid-for-ml)
2. [Data Validation](#data-validation)
3. [Model Testing](#model-testing)
4. [Integration Testing](#integration-testing)
5. [CI/CD Pipelines](#cicd-pipelines)
6. [Production Testing](#production-testing)

---

## Testing Pyramid for ML

```python
# ML Testing Pyramid (Bottom to Top)

┌─────────────────────────────────┐
│   Production Monitoring Tests   │  ← Monitor in production
├─────────────────────────────────┤
│   Shadow/Canary Deployment      │  ← Test with real traffic
├─────────────────────────────────┤
│     Integration Tests           │  ← Test end-to-end pipelines
├─────────────────────────────────┤
│     Model Validation Tests      │  ← Test model behavior
├─────────────────────────────────┤
│     Data Validation Tests       │  ← Test data quality
├─────────────────────────────────┤
│     Unit Tests (Code)           │  ← Test functions/classes
└─────────────────────────────────┘

Focus: Broad base of fast, reliable tests
```

---

## Unit Testing

### Testing ML Code with pytest

```python
# tests/test_preprocessing.py

import pytest
import numpy as np
import pandas as pd
from src.preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test data preprocessing pipeline"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        return DataPreprocessor(
            numeric_features=['feature1', 'feature2'],
            categorical_features=[]
        )
    
    def test_fit_transform(self, preprocessor, sample_data):
        """Test fit_transform produces correct output shape"""
        X = sample_data[['feature1', 'feature2']]
        y = sample_data['target']
        
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)
        
        assert X_transformed.shape == X.shape
        assert y_transformed.shape == y.shape
        assert not np.isnan(X_transformed).any()
    
    def test_scaling(self, preprocessor, sample_data):
        """Test that features are properly scaled"""
        X = sample_data[['feature1', 'feature2']]
        
        X_scaled = preprocessor.fit_transform(X)[0]
        
        # Check mean ≈ 0, std ≈ 1
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)
    
    def test_handle_missing_values(self, preprocessor):
        """Test missing value handling"""
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50]
        })
        
        X_transformed = preprocessor.fit_transform(data)[0]
        
        # No NaN in output
        assert not np.isnan(X_transformed).any()
    
    def test_transform_unseen_data(self, preprocessor, sample_data):
        """Test transform on new data"""
        X_train = sample_data[['feature1', 'feature2']]
        preprocessor.fit(X_train)
        
        # New data
        X_new = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [60, 70]
        })
        
        X_transformed = preprocessor.transform(X_new)
        
        assert X_transformed.shape[0] == 2
        assert not np.isnan(X_transformed).any()
    
    @pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent"])
    def test_imputation_strategies(self, strategy, sample_data):
        """Test different imputation strategies"""
        preprocessor = DataPreprocessor(
            numeric_features=['feature1', 'feature2'],
            imputation_strategy=strategy
        )
        
        data = sample_data.copy()
        data.loc[0, 'feature1'] = np.nan
        
        X_transformed = preprocessor.fit_transform(data[['feature1', 'feature2']])[0]
        assert not np.isnan(X_transformed).any()


# tests/test_model.py

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.model import ModelTrainer

class TestModelTrainer:
    """Test model training logic"""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification data"""
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        return X, y
    
    @pytest.fixture
    def trainer(self):
        """Create trainer instance"""
        return ModelTrainer(
            model_type='random_forest',
            random_state=42
        )
    
    def test_train_returns_model(self, trainer, binary_classification_data):
        """Test that training returns a fitted model"""
        X, y = binary_classification_data
        
        model = trainer.train(X, y)
        
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_model_predictions_valid_shape(self, trainer, binary_classification_data):
        """Test prediction output shape"""
        X, y = binary_classification_data
        X_train, X_test = X[:800], X[800:]
        y_train = y[:800]
        
        model = trainer.train(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape[0] == X_test.shape[0]
        assert set(predictions).issubset({0, 1})
    
    def test_model_probability_predictions(self, trainer, binary_classification_data):
        """Test probability predictions"""
        X, y = binary_classification_data
        X_train, X_test = X[:800], X[800:]
        y_train = y[:800]
        
        model = trainer.train(X_train, y_train)
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (X_test.shape[0], 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
        assert ((probas >= 0) & (probas <= 1)).all()
    
    def test_training_improves_over_baseline(self, trainer, binary_classification_data):
        """Test model beats random baseline"""
        X, y = binary_classification_data
        X_train, X_test = X[:800], X[800:]
        y_train, y_test = y[:800], y[800:]
        
        model = trainer.train(X_train, y_train)
        accuracy = (model.predict(X_test) == y_test).mean()
        
        # Should beat random (0.5) significantly
        assert accuracy > 0.7
    
    def test_deterministic_results(self, binary_classification_data):
        """Test that results are reproducible"""
        X, y = binary_classification_data
        
        trainer1 = ModelTrainer(model_type='random_forest', random_state=42)
        trainer2 = ModelTrainer(model_type='random_forest', random_state=42)
        
        model1 = trainer1.train(X, y)
        model2 = trainer2.train(X, y)
        
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)
        
        assert np.array_equal(pred1, pred2)
```

---

## Data Validation

### Great Expectations

```python
# data_validation/expectations.py

import great_expectations as gx
from great_expectations.core import ExpectationSuite
import pandas as pd
from typing import Dict

class DataValidator:
    """Validate data using Great Expectations"""
    
    def __init__(self, context_root_dir: str = "./gx"):
        self.context = gx.get_context(context_root_dir=context_root_dir)
    
    def create_expectation_suite(
        self,
        suite_name: str,
        data: pd.DataFrame
    ) -> ExpectationSuite:
        """Create expectation suite for dataset"""
        
        # Create or get suite
        suite = self.context.add_or_update_expectation_suite(suite_name)
        
        # Create validator
        validator = self.context.sources.pandas_default.read_dataframe(data)
        validator.expectation_suite_name = suite_name
        
        return validator
    
    def define_feature_expectations(self, validator, feature_config: Dict):
        """Define expectations for features"""
        
        for feature, config in feature_config.items():
            # Column exists
            validator.expect_column_to_exist(feature)
            
            # Data type
            if 'dtype' in config:
                validator.expect_column_values_to_be_of_type(
                    feature,
                    type_=config['dtype']
                )
            
            # Not null (if required)
            if config.get('required', False):
                validator.expect_column_values_to_not_be_null(feature)
            
            # Value range
            if 'min_value' in config:
                validator.expect_column_values_to_be_between(
                    feature,
                    min_value=config['min_value'],
                    max_value=config.get('max_value')
                )
            
            # Allowed values (categorical)
            if 'allowed_values' in config:
                validator.expect_column_values_to_be_in_set(
                    feature,
                    value_set=config['allowed_values']
                )
            
            # Uniqueness
            if config.get('unique', False):
                validator.expect_column_values_to_be_unique(feature)
        
        # Save suite
        validator.save_expectation_suite(discard_failed_expectations=False)
        
        return validator
    
    def validate_data(
        self,
        data: pd.DataFrame,
        suite_name: str
    ) -> Dict:
        """Validate data against expectation suite"""
        
        # Create checkpoint
        checkpoint = self.context.add_or_update_checkpoint(
            name=f"{suite_name}_checkpoint",
            validations=[
                {
                    "batch_request": {
                        "datasource_name": "pandas_default",
                        "data_asset_name": suite_name,
                    },
                    "expectation_suite_name": suite_name,
                }
            ],
        )
        
        # Run validation
        validator = self.context.sources.pandas_default.read_dataframe(data)
        validator.expectation_suite_name = suite_name
        
        results = validator.validate()
        
        return {
            'success': results.success,
            'statistics': results.statistics,
            'failed_expectations': [
                {
                    'expectation_type': exp.expectation_config.expectation_type,
                    'column': exp.expectation_config.kwargs.get('column'),
                    'details': exp.result
                }
                for exp in results.results
                if not exp.success
            ]
        }

# Usage
validator = DataValidator()

# Define expectations
feature_config = {
    'age': {
        'dtype': 'int',
        'required': True,
        'min_value': 0,
        'max_value': 120
    },
    'income': {
        'dtype': 'float',
        'required': True,
        'min_value': 0
    },
    'category': {
        'dtype': 'str',
        'required': True,
        'allowed_values': ['A', 'B', 'C']
    },
    'user_id': {
        'dtype': 'str',
        'required': True,
        'unique': True
    }
}

# Create and save expectations
val = validator.create_expectation_suite("training_data", training_df)
validator.define_feature_expectations(val, feature_config)

# Validate new data
results = validator.validate_data(new_df, "training_data")
if not results['success']:
    print("Validation failed:")
    for failure in results['failed_expectations']:
        print(f"  - {failure['expectation_type']} on {failure['column']}")
```

### Pandera Schema Validation

```python
# data_validation/schemas.py

import pandera as pa
from pandera import Column, Check, DataFrameSchema
import pandas as pd
from typing import Optional

class TrainingDataSchema:
    """Schema for training data validation"""
    
    @staticmethod
    def get_schema() -> DataFrameSchema:
        """Define schema with checks"""
        
        return DataFrameSchema(
            columns={
                # Numeric features
                "age": Column(
                    int,
                    checks=[
                        Check.greater_than_or_equal_to(0),
                        Check.less_than_or_equal_to(120),
                    ],
                    nullable=False,
                    description="Customer age"
                ),
                "income": Column(
                    float,
                    checks=[
                        Check.greater_than(0),
                        Check.less_than(1e7),
                    ],
                    nullable=False
                ),
                "credit_score": Column(
                    int,
                    checks=[
                        Check.in_range(300, 850)
                    ],
                    nullable=True
                ),
                
                # Categorical features
                "category": Column(
                    str,
                    checks=[
                        Check.isin(['A', 'B', 'C', 'D'])
                    ],
                    nullable=False
                ),
                "region": Column(
                    str,
                    checks=[
                        Check.str_matches(r'^[A-Z]{2}$')  # Two-letter code
                    ]
                ),
                
                # Target
                "target": Column(
                    int,
                    checks=[
                        Check.isin([0, 1])
                    ],
                    nullable=False
                ),
            },
            
            # DataFrame-level checks
            checks=[
                # No duplicate rows
                Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found"),
                
                # Reasonable data size
                Check(lambda df: len(df) >= 1000, error="Dataset too small"),
                
                # Class balance check (for target)
                Check(
                    lambda df: (df['target'].value_counts(normalize=True).min() > 0.1),
                    error="Severe class imbalance detected"
                ),
            ],
            
            strict=True,  # No extra columns allowed
            coerce=True   # Try to coerce types
        )
    
    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Validate dataframe against schema"""
        schema = TrainingDataSchema.get_schema()
        return schema.validate(df, lazy=True)

class PredictionDataSchema:
    """Schema for prediction/inference data"""
    
    @staticmethod
    def get_schema() -> DataFrameSchema:
        """Define schema for inference data (no target)"""
        
        return DataFrameSchema(
            columns={
                "age": Column(int, Check.in_range(0, 120)),
                "income": Column(float, Check.greater_than(0)),
                "credit_score": Column(int, Check.in_range(300, 850), nullable=True),
                "category": Column(str, Check.isin(['A', 'B', 'C', 'D'])),
                "region": Column(str, Check.str_matches(r'^[A-Z]{2}$')),
            },
            strict=True,
            coerce=True
        )
    
    @staticmethod
    def validate(df: pd.DataFrame) -> pd.DataFrame:
        """Validate prediction data"""
        schema = PredictionDataSchema.get_schema()
        return schema.validate(df)

# Usage in tests
def test_training_data_schema():
    """Test that training data matches schema"""
    df = pd.read_csv('data/training.csv')
    
    # This will raise if validation fails
    validated_df = TrainingDataSchema.validate(df)
    
    assert len(validated_df) > 0

# Usage in pipeline
try:
    validated_data = TrainingDataSchema.validate(raw_data)
    print("✓ Data validation passed")
except pa.errors.SchemaErrors as err:
    print("✗ Data validation failed:")
    print(err.failure_cases)
    raise
```

---

## Model Testing

### Behavioral Testing

```python
# tests/test_model_behavior.py

import pytest
import numpy as np
import pandas as pd
from src.model import load_model

class TestModelBehavior:
    """Test model behavioral properties"""
    
    @pytest.fixture
    def model(self):
        """Load trained model"""
        return load_model('models/production_model.pkl')
    
    @pytest.fixture
    def baseline_input(self):
        """Baseline input for perturbation tests"""
        return pd.DataFrame({
            'age': [35],
            'income': [50000],
            'credit_score': [700],
            'category': ['B'],
            'region': ['CA']
        })
    
    def test_invariance_to_case(self, model, baseline_input):
        """Test invariance to case changes in categorical features"""
        input1 = baseline_input.copy()
        input2 = baseline_input.copy()
        input2['category'] = input2['category'].str.lower()
        
        pred1 = model.predict_proba(input1)[0]
        pred2 = model.predict_proba(input2)[0]
        
        assert np.allclose(pred1, pred2, atol=0.01)
    
    def test_monotonicity_income(self, model, baseline_input):
        """Test that higher income -> higher approval probability"""
        probas = []
        
        for income in [30000, 50000, 70000, 100000]:
            input_data = baseline_input.copy()
            input_data['income'] = income
            proba = model.predict_proba(input_data)[0, 1]  # Probability of class 1
            probas.append(proba)
        
        # Check monotonic increase
        assert all(probas[i] <= probas[i+1] for i in range(len(probas)-1))
    
    def test_directional_expectation_age(self, model, baseline_input):
        """Test reasonable age effect"""
        young_input = baseline_input.copy()
        young_input['age'] = 25
        
        old_input = baseline_input.copy()
        old_input['age'] = 65
        
        young_proba = model.predict_proba(young_input)[0, 1]
        old_proba = model.predict_proba(old_input)[0, 1]
        
        # Assuming older applicants are more likely approved
        assert old_proba > young_proba
    
    def test_prediction_confidence(self, model, baseline_input):
        """Test that predictions are confident enough"""
        proba = model.predict_proba(baseline_input)[0]
        max_proba = proba.max()
        
        # Model should be reasonably confident
        assert max_proba > 0.6, f"Low confidence: {max_proba}"
    
    def test_small_perturbation_stability(self, model, baseline_input):
        """Test that small changes don't drastically change predictions"""
        original_proba = model.predict_proba(baseline_input)[0, 1]
        
        # Small perturbation (+1% income)
        perturbed = baseline_input.copy()
        perturbed['income'] *= 1.01
        perturbed_proba = model.predict_proba(perturbed)[0, 1]
        
        # Change should be small
        assert abs(original_proba - perturbed_proba) < 0.05
    
    @pytest.mark.parametrize("feature,value", [
        ("age", -5),           # Invalid age
        ("income", -1000),     # Negative income
        ("credit_score", 200), # Invalid credit score
    ])
    def test_handles_invalid_inputs(self, model, baseline_input, feature, value):
        """Test model handles invalid inputs gracefully"""
        invalid_input = baseline_input.copy()
        invalid_input[feature] = value
        
        # Should either raise ValueError or return prediction
        try:
            pred = model.predict(invalid_input)
            assert pred is not None
        except ValueError:
            pass  # Expected behavior

### Performance Testing

```python
# tests/test_model_performance.py

import pytest
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from src.model import load_model
from src.data import load_test_data

class TestModelPerformance:
    """Test model performance on holdout set"""
    
    @pytest.fixture(scope='class')
    def model(self):
        """Load production model"""
        return load_model('models/production_model.pkl')
    
    @pytest.fixture(scope='class')
    def test_data(self):
        """Load test data"""
        return load_test_data('data/test.csv')
    
    @pytest.fixture(scope='class')
    def predictions(self, model, test_data):
        """Generate predictions"""
        X_test, y_test = test_data
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        return y_test, y_pred, y_proba
    
    def test_accuracy_threshold(self, predictions):
        """Test minimum accuracy requirement"""
        y_test, y_pred, _ = predictions
        accuracy = accuracy_score(y_test, y_pred)
        
        MIN_ACCURACY = 0.85
        assert accuracy >= MIN_ACCURACY, f"Accuracy {accuracy:.3f} < {MIN_ACCURACY}"
    
    def test_precision_threshold(self, predictions):
        """Test minimum precision (reduce false positives)"""
        y_test, y_pred, _ = predictions
        precision = precision_score(y_test, y_pred)
        
        MIN_PRECISION = 0.80
        assert precision >= MIN_PRECISION, f"Precision {precision:.3f} < {MIN_PRECISION}"
    
    def test_recall_threshold(self, predictions):
        """Test minimum recall (reduce false negatives)"""
        y_test, y_pred, _ = predictions
        recall = recall_score(y_test, y_pred)
        
        MIN_RECALL = 0.75
        assert recall >= MIN_RECALL, f"Recall {recall:.3f} < {MIN_RECALL}"
    
    def test_auc_threshold(self, predictions):
        """Test minimum AUC-ROC"""
        y_test, _, y_proba = predictions
        auc = roc_auc_score(y_test, y_proba)
        
        MIN_AUC = 0.90
        assert auc >= MIN_AUC, f"AUC {auc:.3f} < {MIN_AUC}"
    
    def test_performance_by_segment(self, model, test_data):
        """Test performance across different segments"""
        X_test, y_test = test_data
        
        # Segment by age
        age_segments = [
            (X_test['age'] < 30, "Young"),
            ((X_test['age'] >= 30) & (X_test['age'] < 50), "Middle"),
            (X_test['age'] >= 50, "Senior")
        ]
        
        for mask, segment_name in age_segments:
            X_segment = X_test[mask]
            y_segment = y_test[mask]
            
            if len(y_segment) > 0:
                y_pred = model.predict(X_segment)
                accuracy = accuracy_score(y_segment, y_pred)
                
                # Each segment should have reasonable performance
                assert accuracy > 0.70, f"{segment_name} segment accuracy {accuracy:.3f} too low"
    
    def test_no_performance_degradation(self, predictions):
        """Test that performance hasn't degraded from baseline"""
        y_test, y_pred, y_proba = predictions
        
        current_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_proba)
        }
        
        # Load baseline metrics
        baseline_metrics = {
            'accuracy': 0.87,
            'f1': 0.84,
            'auc': 0.92
        }
        
        # Allow small degradation (2%)
        TOLERANCE = 0.02
        
        for metric, current_value in current_metrics.items():
            baseline_value = baseline_metrics[metric]
            degradation = baseline_value - current_value
            
            assert degradation <= TOLERANCE, \
                f"{metric} degraded: {baseline_value:.3f} -> {current_value:.3f}"
```

---

## Integration Testing

```python
# tests/test_pipeline_integration.py

import pytest
import pandas as pd
from src.pipeline import MLPipeline

class TestMLPipelineIntegration:
    """Test end-to-end ML pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLPipeline(config_path='config/pipeline.yaml')
    
    @pytest.fixture
    def raw_data(self):
        """Load raw data"""
        return pd.read_csv('data/raw/train.csv')
    
    def test_end_to_end_training(self, pipeline, raw_data):
        """Test complete training pipeline"""
        # Should not raise
        result = pipeline.train(raw_data)
        
        assert 'model' in result
        assert 'metrics' in result
        assert 'preprocessor' in result
        
        # Check metrics
        assert result['metrics']['accuracy'] > 0.80
    
    def test_end_to_end_prediction(self, pipeline):
        """Test complete prediction pipeline"""
        # Load trained pipeline
        pipeline.load('models/production')
        
        # New data
        test_data = pd.read_csv('data/raw/inference.csv')
        
        # Predict
        predictions = pipeline.predict(test_data)
        
        assert len(predictions) == len(test_data)
        assert set(predictions).issubset({0, 1})
    
    def test_pipeline_reproducibility(self, raw_data):
        """Test that pipeline produces consistent results"""
        pipeline1 = MLPipeline(config_path='config/pipeline.yaml', random_state=42)
        pipeline2 = MLPipeline(config_path='config/pipeline.yaml', random_state=42)
        
        result1 = pipeline1.train(raw_data)
        result2 = pipeline2.train(raw_data)
        
        # Predictions should be identical
        test_data = raw_data.sample(100, random_state=42)
        pred1 = result1['model'].predict(test_data.drop('target', axis=1))
        pred2 = result2['model'].predict(test_data.drop('target', axis=1))
        
        assert np.array_equal(pred1, pred2)
    
    def test_pipeline_handles_missing_features(self, pipeline):
        """Test pipeline handles incomplete data"""
        pipeline.load('models/production')
        
        # Data with missing column
        incomplete_data = pd.DataFrame({
            'age': [30],
            'income': [50000]
            # Missing other features
        })
        
        # Should either impute or raise clear error
        try:
            predictions = pipeline.predict(incomplete_data)
            assert predictions is not None
        except ValueError as e:
            assert "missing" in str(e).lower()
```

---

## CI/CD Pipelines

### GitHub Actions ML Pipeline

```yaml
# .github/workflows/ml-pipeline.yml

name: ML Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: "3.10"
  MODEL_REGISTRY: "s3://my-models"

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements-dev.txt
      
      - name: Lint with flake8
        run: |
          flake8 src/ tests/ --max-line-length=100
      
      - name: Type check with mypy
        run: |
          mypy src/
      
      - name: Format check with black
        run: |
          black --check src/ tests/

  data-validation:
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Download data
        run: |
          aws s3 cp s3://my-data/train.csv data/train.csv
      
      - name: Validate data schema
        run: |
          pytest tests/test_data_validation.py -v
      
      - name: Run Great Expectations
        run: |
          great_expectations checkpoint run training_data_checkpoint

  unit-tests:
    runs-on: ubuntu-latest
    needs: code-quality
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run unit tests
        run: |
          pytest tests/test_*.py \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            -v
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  model-training:
    runs-on: ubuntu-latest
    needs: [data-validation, unit-tests]
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Train model
        run: |
          python scripts/train.py \
            --data data/train.csv \
            --output models/candidate_model.pkl \
            --config config/model_config.yaml
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: candidate-model
          path: models/candidate_model.pkl

  model-validation:
    runs-on: ubuntu-latest
    needs: model-training
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: candidate-model
          path: models/
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run model performance tests
        run: |
          pytest tests/test_model_performance.py -v
      
      - name: Run behavioral tests
        run: |
          pytest tests/test_model_behavior.py -v
      
      - name: Compare with baseline
        run: |
          python scripts/compare_models.py \
            --candidate models/candidate_model.pkl \
            --baseline ${{ env.MODEL_REGISTRY }}/production_model.pkl \
            --test-data data/test.csv

  deploy-staging:
    runs-on: ubuntu-latest
    needs: model-validation
    if: github.ref == 'refs/heads/develop'
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: candidate-model
          path: models/
      
      - name: Deploy to staging
        run: |
          aws s3 cp models/candidate_model.pkl \
            s3://my-models/staging/model.pkl
          
          # Update staging endpoint
          python scripts/deploy.py \
            --environment staging \
            --model-path s3://my-models/staging/model.pkl

  deploy-production:
    runs-on: ubuntu-latest
    needs: model-validation
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model
        uses: actions/download-artifact@v3
        with:
          name: candidate-model
          path: models/
      
      - name: Deploy to production (canary)
        run: |
          # Deploy with 10% traffic
          python scripts/deploy.py \
            --environment production \
            --model-path models/candidate_model.pkl \
            --traffic-split 10
      
      - name: Monitor canary
        run: |
          sleep 300  # Wait 5 minutes
          python scripts/check_metrics.py \
            --environment production \
            --canary
      
      - name: Full rollout
        run: |
          python scripts/deploy.py \
            --environment production \
            --traffic-split 100
```

---

## Production Testing

### Shadow Mode Testing

```python
# production_testing/shadow_mode.py

import asyncio
from typing import Dict, Any
import logging
from datetime import datetime

class ShadowModeComparison:
    """Compare new model against production in shadow mode"""
    
    def __init__(
        self,
        production_model,
        shadow_model,
        metrics_client
    ):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.metrics = metrics_client
        self.logger = logging.getLogger(__name__)
    
    async def predict(self, input_data: Dict) -> Dict[str, Any]:
        """
        Run both models:
        - Return production result to user
        - Log shadow model result for comparison
        """
        
        # Run both models in parallel
        production_task = asyncio.create_task(
            self._predict_with_timing(self.production_model, input_data, "production")
        )
        shadow_task = asyncio.create_task(
            self._predict_with_timing(self.shadow_model, input_data, "shadow")
        )
        
        # Wait for both
        production_result, shadow_result = await asyncio.gather(
            production_task,
            shadow_task,
            return_exceptions=True
        )
        
        # Log comparison
        self._compare_results(production_result, shadow_result, input_data)
        
        # Return production result only
        return production_result['prediction']
    
    async def _predict_with_timing(
        self,
        model,
        input_data: Dict,
        model_name: str
    ) -> Dict:
        """Run prediction and measure timing"""
        
        start_time = datetime.now()
        
        try:
            prediction = await model.predict(input_data)
            latency = (datetime.now() - start_time).total_seconds()
            
            return {
                'prediction': prediction,
                'latency': latency,
                'error': None,
                'model': model_name
            }
        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"{model_name} prediction failed: {str(e)}")
            
            return {
                'prediction': None,
                'latency': latency,
                'error': str(e),
                'model': model_name
            }
    
    def _compare_results(
        self,
        production: Dict,
        shadow: Dict,
        input_data: Dict
    ):
        """Log comparison metrics"""
        
        # Latency comparison
        if production['error'] is None and shadow['error'] is None:
            latency_diff = shadow['latency'] - production['latency']
            self.metrics.gauge(
                'shadow_mode.latency_diff',
                latency_diff,
                tags=['model:shadow']
            )
            
            # Prediction agreement
            prod_pred = production['prediction']
            shadow_pred = shadow['prediction']
            
            agreement = (prod_pred == shadow_pred)
            self.metrics.increment(
                'shadow_mode.agreement',
                tags=[f'match:{agreement}']
            )
            
            if not agreement:
                self.logger.warning(
                    f"Prediction mismatch: "
                    f"production={prod_pred}, shadow={shadow_pred}, "
                    f"input={input_data}"
                )
        
        # Error rate comparison
        self.metrics.increment(
            f'shadow_mode.{production["model"]}.requests',
            tags=[f'error:{production["error"] is not None}']
        )
        self.metrics.increment(
            f'shadow_mode.{shadow["model"]}.requests',
            tags=[f'error:{shadow["error"] is not None}']
        )
```

---

## Production Checklist

```markdown
✅ Unit Tests
  □ All functions tested
  □ Edge cases covered
  □ >80% code coverage
  □ Tests are fast (<1s each)

✅ Data Validation
  □ Schema validation implemented
  □ Data quality checks automated
  □ Drift detection configured
  □ Invalid data handling tested

✅ Model Tests
  □ Performance thresholds defined
  □ Behavioral tests written
  □ Fairness metrics tracked
  □ Explainability verified

✅ Integration Tests
  □ End-to-end pipeline tested
  □ Reproducibility verified
  □ Error handling validated
  □ Rollback tested

✅ CI/CD
  □ Automated testing on PR
  □ Model validation automated
  □ Deployment pipeline ready
  □ Rollback procedure defined

✅ Production Testing
  □ Shadow mode implemented
  □ Canary deployment configured
  □ A/B testing ready
  □ Monitoring/alerting set up
```

---

*This guide covers ML testing. For deployment and monitoring, see [Model Serving Guide](model-serving-guide.md) and [ML Observability Guide](ml-observability-guide.md).*
