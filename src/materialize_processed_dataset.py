from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = PROCESSED_DATA_DIR / "clean_taxi_trips.parquet"
BATCH_SIZE = 250_000

SELECTED_COLUMNS = [
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


def and_mask(left: pa.Array, right: pa.Array) -> pa.Array:
    return pc.and_(pc.fill_null(left, False), pc.fill_null(right, False))


def add_features(table: pa.Table) -> pa.Table:
    duration_us = pc.cast(
        pc.subtract(table["tpep_dropoff_datetime"], table["tpep_pickup_datetime"]),
        pa.int64(),
    )
    duration_minutes = pc.divide(pc.cast(duration_us, pa.float64()), pa.scalar(60_000_000.0))
    pickup_day_of_week = pc.day_of_week(table["tpep_pickup_datetime"])
    trip_distance = table["trip_distance"]

    table = table.append_column("trip_duration_minutes", duration_minutes)
    table = table.append_column("pickup_hour", pc.hour(table["tpep_pickup_datetime"]))
    table = table.append_column("pickup_day_of_week", pickup_day_of_week)
    table = table.append_column("pickup_month", pc.month(table["tpep_pickup_datetime"]))
    table = table.append_column(
        "is_weekend",
        pc.cast(
            pc.or_(pc.equal(pickup_day_of_week, 5), pc.equal(pickup_day_of_week, 6)),
            pa.int8(),
        ),
    )
    table = table.append_column("fare_per_mile", pc.divide(table["fare_amount"], trip_distance))
    table = table.append_column(
        "average_speed_mph",
        pc.divide(trip_distance, pc.divide(duration_minutes, pa.scalar(60.0))),
    )
    table = table.append_column(
        "trip_distance_bucket",
        pc.if_else(
            pc.less(trip_distance, 2),
            pa.scalar("short"),
            pc.if_else(pc.less(trip_distance, 8), pa.scalar("medium"), pa.scalar("long")),
        ),
    )

    return table


def clean_batch(batch: pa.RecordBatch) -> pa.Table:
    table = pa.Table.from_batches([batch]).select(SELECTED_COLUMNS)
    table = add_features(table)

    mask = pc.is_valid(table["tpep_pickup_datetime"])
    for column in [
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "payment_type",
        "fare_amount",
        "total_amount",
        "trip_duration_minutes",
        "average_speed_mph",
    ]:
        mask = and_mask(mask, pc.is_valid(table[column]))

    mask = and_mask(mask, pc.greater(table["trip_duration_minutes"], 0))
    mask = and_mask(mask, pc.less_equal(table["trip_duration_minutes"], 180))
    mask = and_mask(mask, pc.greater(table["trip_distance"], 0))
    mask = and_mask(mask, pc.less_equal(table["trip_distance"], 100))
    mask = and_mask(mask, pc.greater_equal(table["passenger_count"], 1))
    mask = and_mask(mask, pc.less_equal(table["passenger_count"], 6))
    mask = and_mask(mask, pc.greater(table["fare_amount"], 0))
    mask = and_mask(mask, pc.greater(table["total_amount"], 0))
    mask = and_mask(mask, pc.greater_equal(table["average_speed_mph"], 1))
    mask = and_mask(mask, pc.less_equal(table["average_speed_mph"], 100))

    return table.filter(mask)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    parquet_files = sorted(RAW_DATA_DIR.glob("yellow_tripdata_2025-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No 2025 Yellow Taxi Parquet files found in {RAW_DATA_DIR}")

    writer = None
    raw_rows = 0
    clean_rows = 0

    try:
        for parquet_path in parquet_files:
            parquet_file = pq.ParquetFile(parquet_path)
            raw_rows += parquet_file.metadata.num_rows
            file_clean_rows = 0

            print(f"Processing {parquet_path.name} ({parquet_file.metadata.num_rows:,} rows)")
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=SELECTED_COLUMNS):
                clean_table = clean_batch(batch)
                file_clean_rows += clean_table.num_rows

                if clean_table.num_rows == 0:
                    continue

                if writer is None:
                    writer = pq.ParquetWriter(OUTPUT_FILE, clean_table.schema, compression="snappy")
                writer.write_table(clean_table)

            clean_rows += file_clean_rows
            print(f"  Clean rows: {file_clean_rows:,}")
    finally:
        if writer is not None:
            writer.close()

    print("Done")
    print(f"Raw rows: {raw_rows:,}")
    print(f"Clean rows: {clean_rows:,}")
    print(f"Rows removed: {raw_rows - clean_rows:,}")
    print(f"Retention rate: {clean_rows / raw_rows:.2%}")
    print(f"Processed dataset: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
