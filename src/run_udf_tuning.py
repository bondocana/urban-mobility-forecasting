import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.storagelevel import StorageLevel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET = PROJECT_ROOT / "data" / "processed" / "clean_taxi_trips.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "tuning_results"
METRICS_FILE = RESULTS_DIR / "udf_tuning_metrics.json"

WINDOWS_JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
if os.name == "nt" and Path(WINDOWS_JAVA_HOME).exists():
    os.environ.setdefault("JAVA_HOME", WINDOWS_JAVA_HOME)
    os.environ["PATH"] = f"{WINDOWS_JAVA_HOME}\\bin;{os.environ['PATH']}"

RANDOM_SEED = 42
DEFAULT_SAMPLE_FRACTION = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UDF analysis and Spark ML hyperparameter tuning.")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of processed rows used for tuning and evaluation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full processed dataset instead of a sample.",
    )
    return parser.parse_args()


def pickup_period(hour):
    if hour is None:
        return "unknown"
    if 5 <= hour <= 10:
        return "morning"
    if 11 <= hour <= 15:
        return "midday"
    if 16 <= hour <= 20:
        return "evening_peak"
    return "night"


pickup_period_udf = F.udf(pickup_period, StringType())


def show_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()
    sample_fraction = 1.0 if args.full else args.sample_fraction

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("UrbanMobilityForecastingUDFTuning")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        show_title("Loading processed dataset")
        clean_df = spark.read.parquet(str(PROCESSED_DATASET))
        input_rows = clean_df.count()
        print(f"Processed rows: {input_rows:,}")

        show_title("Applying pickup period UDF")
        with_period_df = clean_df.withColumn(
            "pickup_period",
            pickup_period_udf(F.col("pickup_hour")),
        )

        period_summary_rows = (
            with_period_df
            .groupBy("pickup_period")
            .agg(
                F.count("*").alias("trip_count"),
                F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
                F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
                F.round(F.avg("trip_duration_minutes"), 2).alias("avg_duration_minutes"),
            )
            .orderBy(F.desc("trip_count"))
            .collect()
        )

        period_summary = [row.asDict() for row in period_summary_rows]
        for row in period_summary:
            print(row)

        numeric_features = [
            "passenger_count",
            "trip_distance",
            "trip_duration_minutes",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_month",
            "is_weekend",
            "PULocationID",
            "DOLocationID",
        ]
        feature_columns = numeric_features

        selected_df = (
            clean_df
            .select(*feature_columns, F.col("fare_amount").alias("label"))
            .dropna()
        )

        if sample_fraction < 1.0:
            model_df = selected_df.sample(
                withReplacement=False,
                fraction=sample_fraction,
                seed=RANDOM_SEED,
            )
        else:
            model_df = selected_df

        model_df = model_df.persist(StorageLevel.DISK_ONLY)
        model_rows = model_df.count()
        if sample_fraction < 1.0:
            print(f"Sample rows used for tuning: {model_rows:,}")
        else:
            print(f"Full rows used for tuning: {model_rows:,}")

        train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

        show_title("Running TrainValidationSplit hyperparameter tuning")
        assembler = VectorAssembler(
            inputCols=feature_columns,
            outputCol="raw_features",
        )
        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=False,
            withStd=True,
        )
        linear_regression = LinearRegression(
            featuresCol="features",
            labelCol="label",
            predictionCol="prediction",
            maxIter=20,
        )

        tuning_pipeline = Pipeline(
            stages=[
                assembler,
                scaler,
                linear_regression,
            ]
        )

        param_grid = (
            ParamGridBuilder()
            .addGrid(linear_regression.regParam, [0.01, 0.1])
            .addGrid(linear_regression.elasticNetParam, [0.0, 0.5])
            .build()
        )

        evaluator = RegressionEvaluator(
            labelCol="label",
            predictionCol="prediction",
            metricName="rmse",
        )

        train_validation_split = TrainValidationSplit(
            estimator=tuning_pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            trainRatio=0.8,
            seed=RANDOM_SEED,
            parallelism=2,
        )

        tuning_model = train_validation_split.fit(train_df)
        predictions_df = tuning_model.transform(test_df)

        rmse = RegressionEvaluator(metricName="rmse").evaluate(predictions_df)
        mae = RegressionEvaluator(metricName="mae").evaluate(predictions_df)
        r2 = RegressionEvaluator(metricName="r2").evaluate(predictions_df)

        best_lr_model = tuning_model.bestModel.stages[-1]
        best_params = {
            "regParam": best_lr_model.getRegParam(),
            "elasticNetParam": best_lr_model.getElasticNetParam(),
        }

        grid_results = []
        for params, validation_rmse in zip(param_grid, tuning_model.validationMetrics):
            grid_results.append(
                {
                    "regParam": params[linear_regression.regParam],
                    "elasticNetParam": params[linear_regression.elasticNetParam],
                    "validation_rmse": validation_rmse,
                }
            )

        print(f"Best parameters: {best_params}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test R2: {r2:.4f}")
        predictions_df.select("label", "prediction").show(10, truncate=False)

        metrics = {
            "input_rows": input_rows,
            "sample_fraction": sample_fraction,
            "rows_used": model_rows,
            "udf": {
                "name": "pickup_period_udf",
                "input_column": "pickup_hour",
                "output_column": "pickup_period",
                "summary": period_summary,
            },
            "tuning": {
                "method": "TrainValidationSplit",
                "train_ratio": 0.8,
                "param_grid": grid_results,
                "best_params": best_params,
                "test_metrics": {
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                },
                "features": feature_columns,
            },
        }

        METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"UDF and tuning metrics saved to: {METRICS_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
