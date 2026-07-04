from pathlib import Path

import csv

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2025-01.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PREVIEW_FILE = OUTPUT_DIR / "yellow_tripdata_2025_01_preview_1000_rows.csv"
COLUMNS_FILE = OUTPUT_DIR / "yellow_tripdata_2025_01_columns.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    parquet_file = pq.ParquetFile(RAW_FILE)
    schema = parquet_file.schema_arrow
    preview_table = parquet_file.read_row_group(0).slice(0, 1000)

    with PREVIEW_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(preview_table.column_names)
        rows = zip(*[preview_table[column].to_pylist() for column in preview_table.column_names])
        writer.writerows(rows)

    with COLUMNS_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["column", "dtype"])
        for field in schema:
            writer.writerow([field.name, str(field.type)])

    print(f"Rows in January file: {parquet_file.metadata.num_rows:,}")
    print(f"Columns: {len(schema)}")
    print(f"Saved preview: {PREVIEW_FILE}")
    print(f"Saved columns: {COLUMNS_FILE}")
    print("First preview columns:")
    print(", ".join(preview_table.column_names))


if __name__ == "__main__":
    main()
