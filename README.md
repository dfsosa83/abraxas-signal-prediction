# Abraxas - Forex Trading Signal Prediction System

## Project Overview

This project implements a sophisticated machine learning pipeline for predicting buy and sell signals in the forex market. The system utilizes multiple models and techniques to generate accurate predictions on an hourly basis.

## Key Features

- Hourly prediction generation
- Dual model approach with meta-learning
- Separate models for buy and sell signals
- Comprehensive feature engineering
- Conformal prediction intervals for confidence estimation

## Model Architecture

### Primary Models
1. **LightGBM Classifier**: Predicts buy/sell signals (Class 1)
2. **XGBoost Classifier**: Predicts buy/sell signals (Class 1)

### Meta-Model
- **Super Learner (LightGBM Classifier)**: Combines predictions from primary models for final signal generation

### Signal-Specific Models
- Dedicated models for buy signals
- Dedicated models for sell signals

## Feature Engineering

The project includes a comprehensive feature engineering function that generates a wide range of technical indicators to enhance model performance.

## Prediction Confidence

Conformal prediction intervals are implemented to provide confidence thresholds for each class probability, enhancing the reliability of predictions.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/abraxas-signal-prediction.git

# Navigate to the project directory
cd forex-signal-prediction

# Create and activate the conda environment
conda env create -f environment.yml
conda activate forex-signal-env



## 1. Modularity

Modularity refers to designing your pipeline as a collection of independent, interchangeable components. Each component should have a specific function and be able to work independently of others.

**Key points:**
- Create separate modules or functions for distinct tasks (e.g., data preprocessing, feature engineering, model training).
- Use clear interfaces between modules to ensure they can be easily connected or replaced.
- Avoid hard-coding parameters; instead, use configuration files or function arguments.



## 2. Scalability

Scalability ensures that your pipeline can handle larger datasets and more complex models without significant changes to the core structure.

**Key points:**
- Use efficient data processing techniques (e.g., chunking for large datasets).
- Implement parallel processing where possible.
- Design with future growth in mind (e.g., ability to add new features or models easily).


## 3. Automation

Automation minimizes manual intervention, making the pipeline more efficient and less prone to human error.

**Key points:**
- Implement automated data ingestion and preprocessing.
- Create automated model training and evaluation scripts.
- Set up automated deployment of new models.
- Use workflow management tools (e.g., Apache Airflow) for complex pipelines.

## 4. Error Handling

Robust error handling and logging ensure that your pipeline can gracefully handle unexpected situations and provide useful information for debugging.

**Key points:**
- Implement try-except blocks for error-prone operations.
- Use logging to record important information, warnings, and errors.
- Create meaningful error messages that help in troubleshooting.
- Implement fallback mechanisms for critical components.

**Example:**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data):
    try:
        model = fit_model(data)
        return model
    except ValueError as e:
        logger.error(f"Error in model training: {str(e)}")
        return None
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        raise

def run_pipeline():
    try:
        data = load_data()
        model = train_model(data)
        if model is not None:
            deploy_model(model)
        else:
            logger.warning("Model training failed, using fallback model")
            use_fallback_model()
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        send_alert_to_team(str(e))
```