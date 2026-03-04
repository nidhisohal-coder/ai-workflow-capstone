AI ENTERPRISE WORKFLOW CAPSTONE

OVERVIEW

This repository contains a complete end-to-end AI enterprise workflow
implementation demonstrating how to move from data exploration to
production-ready deployment.

The project includes:

-   Time-series feature engineering
-   Model training and validation
-   REST API deployment using Flask
-   Docker containerization
-   Unit testing
-   Production simulation
-   Post-production monitoring and drift detection

This implementation reflects a real-world MLOps workflow rather than a
notebook-only prototype.

------------------------------------------------------------------------

BUSINESS OBJECTIVE

Forecast next 30-day revenue using historical transactional data to
support business planning and forward-looking decision-making.

The target variable represents the sum of revenue over the next 30 days,
making this a supervised time-series forecasting problem.

------------------------------------------------------------------------

END-TO-END ARCHITECTURE

Raw Data --> Feature Engineering (Time-Series Lags & Rolling Features) --> Model Training & Validation
--> Serialized Model Artifact --> Flask API (Train / Predict / Logs) --> Docker Container
--> Production Simulation & Monitoring

------------------------------------------------------------------------

PROJECT STRUCTURE

-   src/ – Data ingestion and reusable feature engineering modules
-   app/ – Flask API and model service logic
-   scripts/ – Post-production monitoring & evaluation
-   models/ – Serialized model artifact (joblib)
-   reports/ – Monitoring outputs (MAE, RMSE, evaluation plot)
-   Dockerfile – Container configuration
-   requirements.txt – Python dependencies

------------------------------------------------------------------------

MODELING APPROACH

The following models were evaluated using time-aware validation:

-   Baseline (Mean / Last Value)
-   RandomForestRegressor
-   Ridge Regression
-   HistGradientBoostingRegressor

TimeSeriesSplit was used to preserve chronological integrity.

The selected production model was serialized using joblib and deployed
behind a REST API.

------------------------------------------------------------------------

API ENDPOINTS

Train Model: POST /train

Predict Revenue: POST /predict { “country”: “all”, “date”: “YYYY-MM-DD”
}

Returns: - Predicted 30-day revenue - Date used for feature row - Input
echo

View Logs: GET /logs

Logs capture: - Timestamp - Country - Requested date - Used row date -
Prediction value

------------------------------------------------------------------------

DOCKER USAGE

Build image: docker build -t capstone-api .

Run container: docker run -p 8080:8080 capstone-api

Run tests inside container: docker run –rm capstone-api python -m pytest
-q

------------------------------------------------------------------------

POST-PRODUCTION MONITORING

Run monitoring script: python scripts/post_production_analysis.py

Outputs: - MAE (Mean Absolute Error) - RMSE (Root Mean Squared Error) -
Time-series monitoring plot - Evaluation CSV

The monitoring pipeline compares API predictions to a reconstructed gold
standard from production data.

------------------------------------------------------------------------

KEY OUTCOMES

-   Modular architecture separating ingestion, modeling, API, and
    monitoring
-   Dockerized, portable deployment
-   Structured logging for production simulation
-   Automated post-production performance evaluation
-   Demonstrated detection of model drift

This repository reflects a production-oriented AI workflow aligned with
enterprise deployment standards.

------------------------------------------------------------------------

AUTHOR

Nidhi Sohal AI Enterprise Workflow Capstone
