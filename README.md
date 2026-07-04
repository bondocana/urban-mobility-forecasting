# Urban Mobility Forecasting

Portfolio-oriented data engineering and machine learning project for analyzing urban taxi trips and predicting trip outcomes from large-scale mobility data.

## 1. Introduction

### Dataset Overview

This project is based on the **NYC Taxi & Limousine Commission Trip Record Data**, a public dataset that contains detailed records of taxi trips in New York City.

Dataset source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The dataset includes operational and financial information about taxi rides, such as pickup and drop-off timestamps, trip distance, passenger count, fare amount, payment type, taxes, tips, tolls, and total trip cost. These fields make the dataset suitable for large-scale mobility analysis, distributed data processing, and predictive modeling.

For local development, the project will use a limited time window from the Yellow Taxi trip records, such as one or two months of data. This keeps the project practical to run locally while still preserving the characteristics of a Big Data workflow: large files, structured records, distributed processing, aggregations, machine learning, and streaming-style inference.

### Project Context

Urban taxi trips are influenced by time, location, distance, traffic patterns, payment behavior, and passenger demand. By analyzing taxi trip records, we can identify mobility patterns and build predictive models that estimate trip outcomes before or during a ride.

The project follows an end-to-end data workflow:

- loading raw taxi trip records;
- cleaning invalid or incomplete trips;
- transforming temporal, distance, and payment-related attributes;
- analyzing trip patterns with Spark DataFrames and Spark SQL;
- training machine learning models with Spark MLlib;
- comparing classical machine learning results with a TensorFlow neural network;
- simulating real-time inference using Spark Streaming.

### Objectives

The main objectives of the project are:

1. Analyze NYC taxi trip records using distributed processing techniques.
2. Clean and transform raw trip data into a reliable analytical dataset.
3. Explore fare, distance, duration, and payment patterns across different time intervals.
4. Build and evaluate Spark MLlib models for regression and classification tasks.
5. Use a Spark ML Pipeline to make the feature engineering and modeling workflow reproducible.
6. Apply hyperparameter tuning to improve model performance.
7. Train a TensorFlow model as a deep learning baseline.
8. Simulate a streaming workflow for online inference on newly arriving taxi trip records.

### Target Problems

The project will focus on two predictive tasks:

- **Regression:** predict the trip fare or trip duration based on distance, pickup time, passenger count, payment-related fields, and engineered temporal features.
- **Classification:** classify trips into categories, such as short/medium/long trips or payment behavior classes.

These tasks are appropriate for the dataset because taxi trip records contain both numerical variables and categorical attributes that can be used to estimate ride outcomes and describe mobility behavior.

## Methodology Roadmap

This project uses NYC Taxi trip records to build an end-to-end analytics and prediction workflow:

- distributed data processing with PySpark DataFrames and Spark SQL;
- feature engineering for temporal, distance, payment, and route-related attributes;
- supervised machine learning with Spark MLlib;
- hyperparameter tuning and model evaluation;
- a TensorFlow baseline for deep learning comparison;
- a simulated Spark Streaming pipeline for online inference.

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
