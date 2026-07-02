# Urban Mobility Forecasting

Portfolio-oriented data engineering and machine learning project for analyzing urban taxi trips and predicting trip outcomes from large-scale mobility data.

## Project Idea

This project uses NYC Taxi trip records to build an end-to-end analytics and prediction workflow:

- distributed data processing with PySpark DataFrames and Spark SQL;
- feature engineering for temporal, distance, payment, and route-related attributes;
- supervised machine learning with Spark MLlib;
- hyperparameter tuning and model evaluation;
- a TensorFlow baseline for deep learning comparison;
- a simulated Spark Streaming pipeline for online inference.

## Planned Dataset

Primary source: NYC Taxi & Limousine Commission Trip Record Data  
Dataset page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The project will start with a limited time window, such as one or two months of Yellow Taxi trip records, to keep local development practical while preserving the Big Data workflow.

## Main Objectives

1. Clean and transform trip records using Spark.
2. Explore traffic and fare patterns across time and location.
3. Predict trip fare or duration using Spark MLlib regression models.
4. Classify trip behavior, such as payment type or short/medium/long trip category.
5. Compare Spark MLlib models with a TensorFlow neural network.
6. Simulate real-time trip ingestion and run streaming inference.

## Repository Structure

```text
urban-mobility-forecasting/
  data/
    raw/          # original downloaded files, not committed
    processed/    # cleaned or sampled intermediate files, not committed
  models/         # trained model artifacts, not committed
  notebooks/      # main analysis and project notebook
  reports/        # exported PDF/report files
  src/            # reusable Python helpers
  streaming/      # sample streaming input and streaming utilities
```

## Local Setup

The project is intended to run with Python 3.x, PySpark, pandas, numpy, matplotlib/seaborn, and TensorFlow.

```bash
pip install -r requirements.txt
```

## Status

Initial project scaffolding. The implementation notebook and data pipeline will be added next.
