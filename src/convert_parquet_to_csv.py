from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CSV_OUTPUT_DIR = PROJECT_ROOT / "data" / "csv_visualization"
BATCH_SIZE = 100_000


def convert_file(parquet_path: Path) -> None:
    csv_path = CSV_OUTPUT_DIR / f"{parquet_path.stem}.csv"
    parquet_file = pq.ParquetFile(parquet_path)

    print(f"Converting {parquet_path.name}")
    print(f"  Rows: {parquet_file.metadata.num_rows:,}")
    print(f"  Output: {csv_path}")

    if csv_path.exists():
        csv_path.unlink()

    writer = None
    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pacsv.CSVWriter(csv_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {size_mb:.2f} MB")


def main() -> None:
    CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(RAW_DATA_DIR.glob("yellow_tripdata_2025-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No 2025 Yellow Taxi Parquet files found in {RAW_DATA_DIR}")

    for parquet_path in parquet_files:
        convert_file(parquet_path)

    print(f"Done. CSV files are in: {CSV_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
