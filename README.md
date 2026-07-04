# Urban Mobility Forecasting

Portfolio-oriented data engineering and machine learning project for analyzing urban taxi trips and predicting trip outcomes from large-scale mobility data.

## 1. Introduction

### Dataset Overview

This project is based on the **NYC Taxi & Limousine Commission Trip Record Data**, a public dataset that contains detailed records of taxi trips in New York City.

Dataset source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

The dataset includes operational and financial information about taxi rides, such as pickup and drop-off timestamps, trip distance, passenger count, fare amount, payment type, taxes, tips, tolls, and total trip cost. These fields make the dataset suitable for large-scale mobility analysis, distributed data processing, and predictive modeling.

The project uses a large-scale taxi trip dataset as the main input, built from multiple monthly Yellow Taxi trip record files, ideally covering a full year of trips. The main Spark processing, SQL analysis, aggregations, and machine learning workflow are intended to run on this large dataset, not only on a small sample. Smaller samples may be used only for quick local prototyping or debugging, while the final analysis should demonstrate processing over the larger multi-file dataset.

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

## 2. Spark Processing

The second stage of the project focuses on preparing the raw taxi trip records for analysis and modeling using Apache Spark. Spark is used because the dataset is expected to contain many millions of rows across multiple monthly Parquet files, which makes distributed processing more appropriate than loading everything directly into pandas.

### Data Loading

The raw Yellow Taxi trip records will be stored in `data/raw/` as monthly Parquet files. Spark can read all files from this folder as one distributed DataFrame:

```python
trips_df = spark.read.parquet("data/raw/yellow_tripdata_*.parquet")
```

A Spark DataFrame behaves similarly to a table: it has columns, rows, types, filters, joins, and aggregations. The important difference is that Spark does not need to load all data into local memory at once. Instead, it builds an execution plan and processes the data in partitions.

### Initial Inspection

Before cleaning, the project will inspect the dataset structure and size:

- print the schema;
- count the number of rows;
- check missing values in important columns;
- inspect numerical ranges for distance, fare, passenger count, and timestamps;
- identify invalid or unrealistic records.

This step is important because public mobility datasets often contain noisy records, such as zero-distance trips, negative fares, missing timestamps, or extremely long durations.

### Cleaning Rules

The cleaned dataset will keep only records that are plausible for modeling and analysis. Typical cleaning rules include:

- keep trips with valid pickup and drop-off timestamps;
- remove trips with zero or negative duration;
- remove trips with negative fare amounts;
- remove trips with negative or zero trip distance;
- remove rows with missing values in key modeling columns;
- keep passenger counts within a realistic interval;
- remove extreme outliers when they would distort model training.

These rules are applied with the Spark DataFrame API using operations such as `filter`, `withColumn`, `dropna`, and type conversions.

### Feature Engineering

After cleaning, new columns will be derived from the original data:

- `trip_duration_minutes`: difference between drop-off and pickup time;
- `pickup_hour`: hour of day when the trip started;
- `pickup_day_of_week`: day of week;
- `pickup_month`: trip month;
- `is_weekend`: whether the trip happened during the weekend;
- `average_speed_mph`: distance divided by duration;
- `fare_per_mile`: fare amount divided by trip distance;
- `trip_distance_bucket`: short, medium, or long distance category.

These features help transform raw trip records into variables that are more useful for analysis and machine learning.

### DataFrame Aggregations

The project will include Spark DataFrame aggregations such as:

- number of trips per hour;
- average fare per hour;
- average trip duration by day of week;
- average distance by payment type;
- total revenue by month;
- distribution of short, medium, and long trips.

These aggregations demonstrate distributed group-by operations over a large dataset.

### Spark SQL Analysis

In addition to the DataFrame API, the project will also use Spark SQL. The cleaned DataFrame will be registered as a temporary SQL view:

```python
clean_trips_df.createOrReplaceTempView("taxi_trips")
```

After that, SQL queries can be executed directly:

```python
spark.sql("""
    SELECT
        pickup_hour,
        COUNT(*) AS trip_count,
        AVG(fare_amount) AS avg_fare,
        AVG(trip_duration_minutes) AS avg_duration
    FROM taxi_trips
    GROUP BY pickup_hour
    ORDER BY pickup_hour
""")
```

Spark SQL is useful because it allows the same dataset to be analyzed using familiar relational operations, while Spark still executes the work in a distributed way.

### Output of This Stage

The result of this stage will be a cleaned and enriched Spark DataFrame that can be used for:

- exploratory analysis and visualizations;
- Spark MLlib regression and classification models;
- the Spark ML Pipeline;
- TensorFlow training after sampling or converting selected features;
- simulated streaming inference.

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
