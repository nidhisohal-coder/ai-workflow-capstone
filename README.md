# AI Enterprise Workflow Capstone

## Overview

This repository contains a complete end-to-end AI enterprise workflow implementation including:

- Time-series feature engineering
- Model training and evaluation
- REST API deployment using Flask
- Docker containerization
- Unit testing
- Production simulation
- Post-production monitoring

This project demonstrates how to take a predictive model from experimentation to production-ready deployment.

---

## Business Objective

Forecast next 30-day revenue using historical transactional data to support business planning and forecasting decisions.

---

## Project Architecture

Data → Feature Engineering → Model Training → API → Docker → Monitoring


### Key Components

- `src/` – Data ingestion and feature engineering
- `app/` – Flask API and model service
- `scripts/` – Post-production analysis
- `models/` – Serialized model artifact
- `reports/` – Monitoring outputs (MAE, RMSE, plots)
- `Dockerfile` – Containerization setup

---

## Modeling Approach

Tested models:

- Baseline (Mean / Last Value)
- RandomForestRegressor
- Ridge Regression
- HistGradientBoostingRegressor

Selected deployment model was serialized using `joblib`.

---

## API Endpoints

### Train Model
POST /train

### Predict Revenue
POST /predict
{
"country": "all",
"date": "YYYY-MM-DD"
}


### View Logs
GET /logs

## Docker Usage
Build image:
docker build -t capstone-api .


Run container:
docker run -p 8080:8080 capstone-api


---

## Post-Production Monitoring

Run monitoring script:
python scripts/post_production_analysis.py


Outputs:

- MAE and RMSE
- Monitoring plot
- Evaluation CSV

---

## Results

Post-production monitoring detected model drift, demonstrating:

- Logging integrity
- Gold standard comparison
- Performance degradation detection
- Production readiness

---

## Author

Nidhi Sohal  
AI Enterprise Workflow Capstone

