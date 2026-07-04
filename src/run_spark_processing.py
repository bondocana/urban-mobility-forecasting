import os
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
SUMMARY_DIR = PROCESSED_DATA_DIR / "spark_processing_summary"
CLEAN_OUTPUT_DIR = PROCESSED_DATA_DIR / "clean_taxi_trips.parquet"

JAVA_HOME = r"C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot"
os.environ.setdefault("JAVA_HOME", JAVA_HOME)
os.environ["PATH"] = f"{JAVA_HOME}\\bin;{os.environ['PATH']}"


def show_title(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_DATA_DIR.glob("yellow_tripdata_2025-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No 2025 Yellow Taxi Parquet files found in {RAW_DATA_DIR}")

    show_title("Input files")
    for path in parquet_files:
        print(f"{path.name} - {path.stat().st_size / (1024 * 1024):.2f} MB")

    spark = (
        SparkSession.builder
        .appName("UrbanMobilityForecastingProcessing")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        show_title("Loading raw dataset")
        trips_df = spark.read.parquet(*[str(path) for path in parquet_files])
        raw_count = trips_df.count()
        print(f"Raw rows: {raw_count:,}")
        trips_df.printSchema()

        selected_columns = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "PULocationID",
            "DOLocationID",
            "payment_type",
            "fare_amount",
            "tip_amount",
            "tolls_amount",
            "total_amount",
        ]
        missing_columns = [column for column in selected_columns if column not in trips_df.columns]
        if missing_columns:
            raise ValueError(f"Missing expected columns: {missing_columns}")

        trips_selected_df = trips_df.select(*selected_columns)

        show_title("Missing values")
        important_columns = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "payment_type",
            "fare_amount",
            "total_amount",
        ]
        missing_summary_df = trips_selected_df.select([
            F.count(F.when(F.col(column).isNull(), column)).alias(column)
            for column in important_columns
        ])
        missing_summary_df.show(truncate=False)

        show_title("Numeric ranges before cleaning")
        numeric_summary_df = trips_selected_df.select(
            F.min("passenger_count").alias("min_passenger_count"),
            F.max("passenger_count").alias("max_passenger_count"),
            F.min("trip_distance").alias("min_trip_distance"),
            F.max("trip_distance").alias("max_trip_distance"),
            F.min("fare_amount").alias("min_fare_amount"),
            F.max("fare_amount").alias("max_fare_amount"),
            F.min("total_amount").alias("min_total_amount"),
            F.max("total_amount").alias("max_total_amount"),
        )
        numeric_summary_df.show(truncate=False)

        show_title("Cleaning and feature engineering")
        trips_features_df = (
            trips_selected_df
            .withColumn(
                "trip_duration_minutes",
                (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 60.0,
            )
            .withColumn("pickup_hour", F.hour("tpep_pickup_datetime"))
            .withColumn("pickup_day_of_week", F.dayofweek("tpep_pickup_datetime"))
            .withColumn("pickup_month", F.month("tpep_pickup_datetime"))
            .withColumn("is_weekend", F.col("pickup_day_of_week").isin([1, 7]).cast("int"))
            .withColumn("fare_per_mile", F.col("fare_amount") / F.col("trip_distance"))
            .withColumn("average_speed_mph", F.col("trip_distance") / (F.col("trip_duration_minutes") / 60.0))
        )

        clean_trips_df = (
            trips_features_df
            .dropna(subset=[
                "tpep_pickup_datetime",
                "tpep_dropoff_datetime",
                "passenger_count",
                "trip_distance",
                "payment_type",
                "fare_amount",
                "total_amount",
                "trip_duration_minutes",
            ])
            .filter(F.col("trip_duration_minutes") > 0)
            .filter(F.col("trip_duration_minutes") <= 180)
            .filter(F.col("trip_distance") > 0)
            .filter(F.col("trip_distance") <= 100)
            .filter(F.col("passenger_count").between(1, 6))
            .filter(F.col("fare_amount") > 0)
            .filter(F.col("fare_amount") <= 500)
            .filter(F.col("total_amount") > 0)
            .filter(F.col("total_amount") <= 1000)
            .filter(F.col("average_speed_mph").between(1, 100))
            .filter(F.col("fare_per_mile").between(1, 100))
            .withColumn(
                "trip_distance_bucket",
                F.when(F.col("trip_distance") < 2, "short")
                .when(F.col("trip_distance") < 8, "medium")
                .otherwise("long"),
            )
        )

        clean_trips_df = clean_trips_df.cache()
        clean_count = clean_trips_df.count()
        print(f"Clean rows: {clean_count:,}")
        print(f"Rows removed: {raw_count - clean_count:,}")
        print(f"Retention rate: {clean_count / raw_count:.2%}")
        clean_trips_df.select(
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "trip_duration_minutes",
            "pickup_hour",
            "trip_distance_bucket",
        ).show(10, truncate=False)

        show_title("DataFrame aggregations")
        trips_by_hour_df = (
            clean_trips_df
            .groupBy("pickup_hour")
            .agg(
                F.count("*").alias("trip_count"),
                F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
                F.round(F.avg("trip_duration_minutes"), 2).alias("avg_duration_minutes"),
                F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
            )
            .orderBy("pickup_hour")
        )
        trips_by_hour_df.show(24, truncate=False)

        payment_type_summary_df = (
            clean_trips_df
            .groupBy("payment_type")
            .agg(
                F.count("*").alias("trip_count"),
                F.round(F.avg("total_amount"), 2).alias("avg_total_amount"),
                F.round(F.avg("tip_amount"), 2).alias("avg_tip_amount"),
                F.round(F.avg("trip_distance"), 2).alias("avg_distance"),
            )
            .orderBy(F.desc("trip_count"))
        )
        payment_type_summary_df.show(truncate=False)

        distance_bucket_summary_df = (
            clean_trips_df
            .groupBy("trip_distance_bucket")
            .agg(
                F.count("*").alias("trip_count"),
                F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
                F.round(F.avg("trip_duration_minutes"), 2).alias("avg_duration_minutes"),
            )
            .orderBy("trip_distance_bucket")
        )
        distance_bucket_summary_df.show(truncate=False)

        show_title("Spark SQL analysis")
        clean_trips_df.createOrReplaceTempView("taxi_trips")

        monthly_revenue_df = spark.sql("""
            SELECT
                pickup_month,
                COUNT(*) AS trip_count,
                ROUND(SUM(total_amount), 2) AS total_revenue,
                ROUND(AVG(total_amount), 2) AS avg_total_amount,
                ROUND(AVG(trip_distance), 2) AS avg_distance
            FROM taxi_trips
            GROUP BY pickup_month
            ORDER BY pickup_month
        """)
        monthly_revenue_df.show(12, truncate=False)

        weekend_bucket_df = spark.sql("""
            SELECT
                is_weekend,
                trip_distance_bucket,
                COUNT(*) AS trip_count,
                ROUND(AVG(fare_amount), 2) AS avg_fare,
                ROUND(AVG(trip_duration_minutes), 2) AS avg_duration_minutes
            FROM taxi_trips
            GROUP BY is_weekend, trip_distance_bucket
            ORDER BY is_weekend, trip_distance_bucket
        """)
        weekend_bucket_df.show(truncate=False)

        show_title("Saving processed outputs")
        (
            clean_trips_df
            .write
            .mode("overwrite")
            .partitionBy("pickup_month")
            .parquet(str(CLEAN_OUTPUT_DIR))
        )
        print(f"Clean dataset saved to: {CLEAN_OUTPUT_DIR}")

        trips_by_hour_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(SUMMARY_DIR / "trips_by_hour"))
        payment_type_summary_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(SUMMARY_DIR / "payment_type_summary"))
        distance_bucket_summary_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(SUMMARY_DIR / "distance_bucket_summary"))
        monthly_revenue_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(SUMMARY_DIR / "monthly_revenue"))
        weekend_bucket_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(SUMMARY_DIR / "weekend_bucket_summary"))
        print(f"Summary CSV folders saved to: {SUMMARY_DIR}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
