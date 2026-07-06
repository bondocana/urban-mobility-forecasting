# Urban Mobility Forecasting

Big Data project using NYC Yellow Taxi trip records from 2025.

The goal of this project is to build a complete local data workflow: starting from raw monthly taxi trip files, cleaning and analyzing them with Spark, training machine learning models, comparing results, and simulating a small streaming use case.

## Dataset

I used the public NYC TLC Yellow Taxi Trip Records for 2025:

`https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page`

The raw data is stored locally as monthly Parquet files in `data/raw/`. The full raw and processed datasets are not committed to GitHub because the files are large.

For the final run, the processed dataset contains `34,921,920` cleaned taxi trips.

## Project Structure

- `notebooks/urban_mobility_forecasting.ipynb` - main notebook with the full workflow and final results
- `src/materialize_processed_dataset.py` - creates the cleaned Parquet dataset
- `src/run_spark_processing.py` - Spark DataFrame and Spark SQL processing
- `src/run_mllib_models.py` - Spark MLlib regression and classification models
- `src/run_pipeline_model.py` - Spark ML Pipeline for fare prediction
- `src/run_udf_tuning.py` - UDF analysis and hyperparameter tuning
- `src/run_tensorflow_model.py` - TensorFlow model trained on the processed dataset
- `src/run_streaming_simulation.py` - Spark micro-batch streaming simulation

## Workflow

1. Load the 2025 Yellow Taxi Parquet files with Spark.
2. Select useful columns and check data quality.
3. Clean invalid trips and create extra features such as duration, pickup hour, weekend flag, and distance bucket.
4. Run Spark DataFrame aggregations and Spark SQL queries.
5. Train Spark MLlib models for fare prediction and trip category classification.
6. Build a reusable Spark ML Pipeline.
7. Run hyperparameter tuning with `TrainValidationSplit`.
8. Train a TensorFlow neural network on the processed dataset.
9. Simulate streaming analytics with Spark micro-batches.

## Main Results

| Step | Main metric |
|---|---:|
| Spark MLlib Linear Regression | RMSE `5.5359`, R2 `0.9064` |
| Spark MLlib Logistic Regression | Accuracy `0.8078`, F1 `0.7978` |
| Spark ML Pipeline | RMSE `5.5427`, R2 `0.9061` |
| Tuned Spark model | RMSE `5.5420`, R2 `0.9061` |
| TensorFlow model | RMSE `5.3210`, R2 `0.9135` |
| Streaming simulation | `5,000` events in `5` micro-batches |

The TensorFlow model had the best fare prediction result, while the Spark models were useful as scalable baselines and for showing the distributed processing workflow.
