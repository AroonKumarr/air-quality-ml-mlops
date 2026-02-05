# Pearls AQI Predictor

Predict the Air Quality Index (AQI) in your city for the next 3 days using a 100% serverless stack.

## Project Structure

```
air-quality-ml/
│
├── .devcontainer/              # VSCode dev container setup
├── .github/                   # CI/CD workflows
├── .env                       # Environment variables (API keys etc.)
├── .gitignore
├── README.md
├── requirements.txt
├── AQI_predict.pdf            # Project documentation/report
│
├── app_core/                  # Core ML system (heart of project)
│   ├── feature_engineering/   # Feature computation pipeline
│   │   ├── data_loader.py
│   │   ├── feature_computation.py
│   │   ├── feature_pipeline.py
│   │   └── feature_registry.py
│   │
│   ├── model_definitions/     # Model training definitions
│   │   └── train_models.py
│   │
│   ├── prediction_engine/     # Inference + explainability
│   │   ├── prediction_service.py
│   │   └── model_explainability.py
│   │
│   ├── training_engine/       # Training orchestration logic
│   └── utils/                 # Config & logging
│       ├── config.py
│       └── logger.py
│
├── data_pipeline/             # Orchestration layer (MLOps pipelines)
│   ├── model_training_runner.py
│   ├── training_orchestration.py
│   ├── inference_orchestration.py
│   └── feature_orchestration.py
│
├── automation_scripts/        # Data ingestion + validation + Hopsworks
│   ├── historical_data_loader.py
│   ├── live_aqi_ingestion.py
│   ├── data_quality_service.py
│   ├── model_benchmarking.py
│   └── hopsworks_*.py
│
├── datasets/                  # Data lake
│   ├── source_data/
│   ├── historical_data/
│   ├── curated_data/          # Final features (parquet/csv)
│   └── data_checkpoints/
│
├── model_artifacts/           # Saved trained models (created after training)
│   ├── lightgbm/
│   ├── xgboost/
│   ├── random_forest/
│   ├── ridge/
│   └── neural_network/
│
├── settings/                  # YAML configs
│   ├── config.yaml
│   └── model_config.yaml
│
├── web_interface/             # Streamlit dashboard + API
│   ├── app.py
│   ├── islamabad_predictor.py
│   └── api/
│
├── jupyter/                   # Experiment notebooks
│   ├── etl_pipeline_experiment.ipynb
│   └── hopsworks_integration_experiment.ipynb
│
├── unit_tests/                # Test cases
└── venv/                      # Local virtual environment

```

## Features

- **Feature Pipeline**: Fetches AQI data, computes time-based features (hour, day, month), AQI change rate
- **Training Pipeline**: Supports Scikit-learn (Random Forest, Ridge Regression) and Deep Learning models
- **Automated Pipelines**: GitHub Actions for hourly feature updates and daily model retraining
- **Web Dashboard**: Interactive UI showing predictions and forecasts
- **Explainability**: SHAP/LIME for feature importance
- **Alerts**: Notifications for hazardous AQI levels

## Guidelines

1. Perform EDA to identify trends
2. Use variety of forecasting models (statistical to deep learning)
3. Use SHAP or LIME for feature importance explanations
4. Add alerts for hazardous AQI levels

## Final Deliverables

1. End-to-end AQI prediction system
2. A scalable, automated pipeline
3. An interactive dashboard showcasing real-time and forecasted AQI data

## Setup

1. Clone this repository
2. Create virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and configure
6. Run the app: `streamlit run webapp/app.py`

## Author
Aroon Kumar

## Project Ownership
This project is maintained and extended by Aroon Kumar.