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
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET = PROJECT_ROOT / "data" / "processed" / "clean_taxi_trips.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "pipeline_results"
METRICS_FILE = RESULTS_DIR / "pipeline_metrics.json"

WINDOWS_JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
if os.name == "nt" and Path(WINDOWS_JAVA_HOME).exists():
    os.environ.setdefault("JAVA_HOME", WINDOWS_JAVA_HOME)
    os.environ["PATH"] = f"{WINDOWS_JAVA_HOME}\\bin;{os.environ['PATH']}"

RANDOM_SEED = 42
DEFAULT_SAMPLE_FRACTION = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Spark ML Pipeline for fare prediction.")
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=DEFAULT_SAMPLE_FRACTION,
        help="Fraction of processed rows used for training and evaluation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full processed dataset instead of a sample.",
    )
    return parser.parse_args()


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
        .appName("UrbanMobilityForecastingPipeline")
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

        pipeline_features = [
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

        selected_df = (
            clean_df
            .select(*pipeline_features, F.col("fare_amount").alias("label"))
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
            print(f"Sample rows used for pipeline: {model_rows:,}")
        else:
            print(f"Full rows used for pipeline: {model_rows:,}")

        train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

        show_title("Training Spark ML Pipeline")
        assembler = VectorAssembler(
            inputCols=pipeline_features,
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
            maxIter=25,
            regParam=0.1,
            elasticNetParam=0.0,
        )

        fare_prediction_pipeline = Pipeline(
            stages=[
                assembler,
                scaler,
                linear_regression,
            ]
        )

        pipeline_model = fare_prediction_pipeline.fit(train_df)
        predictions_df = pipeline_model.transform(test_df)

        rmse = RegressionEvaluator(metricName="rmse").evaluate(predictions_df)
        mae = RegressionEvaluator(metricName="mae").evaluate(predictions_df)
        r2 = RegressionEvaluator(metricName="r2").evaluate(predictions_df)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        predictions_df.select("label", "prediction").show(10, truncate=False)

        metrics = {
            "input_rows": input_rows,
            "sample_fraction": sample_fraction,
            "rows_used": model_rows,
            "target": "fare_amount",
            "features": pipeline_features,
            "pipeline_stages": [
                "VectorAssembler",
                "StandardScaler",
                "LinearRegression",
            ],
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Pipeline metrics saved to: {METRICS_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
