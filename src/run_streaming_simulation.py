import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "streaming_results"
RESULTS_FILE = RESULTS_DIR / "streaming_summary.json"

WINDOWS_JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
if os.name == "nt" and Path(WINDOWS_JAVA_HOME).exists():
    os.environ.setdefault("JAVA_HOME", WINDOWS_JAVA_HOME)
    os.environ["PATH"] = f"{WINDOWS_JAVA_HOME}\\bin;{os.environ['PATH']}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Spark micro-batch streaming simulation.")
    parser.add_argument(
        "--rows-per-second",
        type=int,
        default=1000,
        help="Number of synthetic taxi events generated per second.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=5,
        help="How long the streaming query should run.",
    )
    return parser.parse_args()


def show_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("UrbanMobilityForecastingStreamingSimulation")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        show_title("Starting Spark micro-batch streaming simulation")
        total_rows = args.rows_per_second * args.duration_seconds
        simulated_stream_df = spark.range(0, total_rows).withColumnRenamed("id", "value")

        taxi_stream_df = simulated_stream_df.select(
            (F.to_timestamp(F.lit("2025-01-01 00:00:00")) + (F.col("value") * F.expr("INTERVAL 1 SECOND"))).alias(
                "event_timestamp"
            ),
            (F.col("value") / args.rows_per_second).cast("int").alias("micro_batch_id"),
            ((F.col("value") % 6) + 1).cast("double").alias("passenger_count"),
            F.round(((F.col("value") % 300) / 20.0) + 0.5, 2).alias("trip_distance"),
            (((F.col("value") % 24))).cast("int").alias("pickup_hour"),
            (((F.col("value") % 7) + 1)).cast("int").alias("pickup_day_of_week"),
            (((F.col("value") % 12) + 1)).cast("int").alias("pickup_month"),
            ((F.col("value") % 7).isin([0, 6])).cast("int").alias("is_weekend"),
            ((F.col("value") % 260) + 1).cast("int").alias("PULocationID"),
            (((F.col("value") + 37) % 260) + 1).cast("int").alias("DOLocationID"),
        )

        taxi_stream_df = (
            taxi_stream_df
            .withColumn(
                "trip_duration_minutes",
                F.round(F.col("trip_distance") * 4 + (F.col("pickup_hour") % 5) + 3, 2),
            )
            .withColumn(
                "estimated_fare",
                F.round(
                    F.lit(3.0)
                    + F.col("trip_distance") * 2.65
                    + F.col("trip_duration_minutes") * 0.35
                    + F.col("is_weekend") * 1.25,
                    2,
                ),
            )
        )

        enriched_stream_df = taxi_stream_df.withColumn(
            "pickup_period",
            F.when(F.col("pickup_hour").between(5, 10), "morning")
            .when(F.col("pickup_hour").between(11, 15), "midday")
            .when(F.col("pickup_hour").between(16, 20), "evening_peak")
            .otherwise("night"),
        )

        streaming_summary_df = (
            enriched_stream_df
            .groupBy("pickup_period")
            .agg(
                F.count("*").alias("trip_count"),
                F.round(F.avg("estimated_fare"), 2).alias("avg_estimated_fare"),
                F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
                F.round(F.avg("trip_duration_minutes"), 2).alias("avg_duration_minutes"),
            )
        )

        micro_batch_summary_df = (
            enriched_stream_df
            .groupBy("micro_batch_id")
            .agg(
                F.count("*").alias("trip_count"),
                F.round(F.avg("estimated_fare"), 2).alias("avg_estimated_fare"),
                F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
            )
            .orderBy("micro_batch_id")
        )

        result_df = streaming_summary_df.orderBy(F.desc("trip_count"))
        batch_result_df = micro_batch_summary_df.orderBy("micro_batch_id")

        show_title("Per-micro-batch summary")
        batch_result_df.show(truncate=False)

        show_title("Final streaming summary")
        result_df.show(truncate=False)

        summary_rows = [row.asDict() for row in result_df.collect()]
        micro_batch_rows = [row.asDict() for row in batch_result_df.collect()]
        metrics = {
            "streaming_engine": "Spark micro-batch simulation",
            "source_format": "synthetic taxi events generated with Spark range",
            "rows_per_second": args.rows_per_second,
            "duration_seconds": args.duration_seconds,
            "total_events": total_rows,
            "micro_batch_count": args.duration_seconds,
            "aggregation": "trip counts, average estimated fare, average distance, and average duration by pickup period",
            "micro_batches": micro_batch_rows,
            "summary": summary_rows,
            "local_note": (
                "This local Windows run simulates Spark streaming micro-batches with Spark DataFrames because "
                "native Structured Streaming writeStream requires Hadoop winutils.exe for checkpoint metadata."
            ),
        }

        RESULTS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Streaming summary saved to: {RESULTS_FILE}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
