import argparse
import json
import os
from pathlib import Path

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET = PROJECT_ROOT / "data" / "processed" / "clean_taxi_trips.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "ml_results"
METRICS_FILE = RESULTS_DIR / "mllib_metrics.json"

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ.setdefault("JAVA_HOME", JAVA_HOME)
os.environ["PATH"] = f"{JAVA_HOME}\\bin;{os.environ['PATH']}"

RANDOM_SEED = 42
DEFAULT_SAMPLE_FRACTION = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spark MLlib models for the taxi project.")
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
        .appName("UrbanMobilityForecastingMLlib")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        show_title("Loading processed dataset")
        clean_df = spark.read.parquet(str(PROCESSED_DATASET))
        total_rows = clean_df.count()
        print(f"Processed rows: {total_rows:,}")

        selected_df = (
            clean_df
            .select(
                "passenger_count",
                "trip_distance",
                "fare_amount",
                "total_amount",
                "trip_duration_minutes",
                "pickup_hour",
                "pickup_day_of_week",
                "pickup_month",
                "is_weekend",
                "PULocationID",
                "DOLocationID",
                "trip_distance_bucket",
            )
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
        sample_count = model_df.count()
        if sample_fraction < 1.0:
            print(f"Sample rows used for MLlib: {sample_count:,}")
        else:
            print(f"Full rows used for MLlib: {sample_count:,}")

        show_title("Method 1 - Linear Regression for fare prediction")
        regression_features = [
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
        regression_assembler = VectorAssembler(
            inputCols=regression_features,
            outputCol="features",
        )
        regression_ready_df = (
            regression_assembler
            .transform(model_df)
            .select(F.col("fare_amount").alias("label"), "features")
        )
        regression_train_df, regression_test_df = regression_ready_df.randomSplit(
            [0.8, 0.2],
            seed=RANDOM_SEED,
        )

        linear_regression = LinearRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=25,
            regParam=0.1,
            elasticNetParam=0.0,
        )
        regression_model = linear_regression.fit(regression_train_df)
        regression_predictions_df = regression_model.transform(regression_test_df)

        rmse = RegressionEvaluator(metricName="rmse").evaluate(regression_predictions_df)
        mae = RegressionEvaluator(metricName="mae").evaluate(regression_predictions_df)
        r2 = RegressionEvaluator(metricName="r2").evaluate(regression_predictions_df)

        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        regression_predictions_df.select("label", "prediction").show(10, truncate=False)

        show_title("Method 2 - Logistic Regression for trip category classification")
        classification_features = [
            "passenger_count",
            "fare_amount",
            "total_amount",
            "trip_duration_minutes",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_month",
            "is_weekend",
            "PULocationID",
            "DOLocationID",
        ]

        label_indexer = StringIndexer(
            inputCol="trip_distance_bucket",
            outputCol="label",
            handleInvalid="skip",
        )
        indexed_df = label_indexer.fit(model_df).transform(model_df)

        classification_assembler = VectorAssembler(
            inputCols=classification_features,
            outputCol="features",
        )
        classification_ready_df = (
            classification_assembler
            .transform(indexed_df)
            .select("label", "features", "trip_distance_bucket")
        )

        class_distribution_df = (
            indexed_df
            .groupBy("trip_distance_bucket")
            .count()
            .orderBy(F.desc("count"))
        )
        print("Class distribution in sampled dataset:")
        class_distribution_df.show(truncate=False)

        classification_train_df, classification_test_df = classification_ready_df.randomSplit(
            [0.8, 0.2],
            seed=RANDOM_SEED,
        )

        logistic_regression = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            regParam=0.05,
            elasticNetParam=0.0,
            family="multinomial",
        )
        classification_model = logistic_regression.fit(classification_train_df)
        classification_predictions_df = classification_model.transform(classification_test_df)

        accuracy = MulticlassClassificationEvaluator(metricName="accuracy").evaluate(
            classification_predictions_df
        )
        f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(
            classification_predictions_df
        )
        weighted_precision = MulticlassClassificationEvaluator(metricName="weightedPrecision").evaluate(
            classification_predictions_df
        )
        weighted_recall = MulticlassClassificationEvaluator(metricName="weightedRecall").evaluate(
            classification_predictions_df
        )

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(f"Weighted precision: {weighted_precision:.4f}")
        print(f"Weighted recall: {weighted_recall:.4f}")
        classification_predictions_df.select("label", "prediction", "trip_distance_bucket").show(
            10,
            truncate=False,
        )

        confusion_matrix_df = (
            classification_predictions_df
            .groupBy("label", "prediction")
            .count()
            .orderBy("label", "prediction")
        )
        print("Confusion matrix counts:")
        confusion_matrix_df.show(50, truncate=False)

        metrics = {
            "input_rows": total_rows,
            "sample_fraction": sample_fraction,
            "sample_rows": sample_count,
            "linear_regression": {
                "target": "fare_amount",
                "features": regression_features,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            },
            "logistic_regression": {
                "target": "trip_distance_bucket",
                "features": classification_features,
                "accuracy": accuracy,
                "f1": f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
            },
        }

        METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics saved to: {METRICS_FILE}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
