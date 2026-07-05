import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATASET = PROJECT_ROOT / "data" / "processed" / "clean_taxi_trips.parquet"
RESULTS_DIR = PROJECT_ROOT / "data" / "processed" / "tensorflow_results"
METRICS_FILE = RESULTS_DIR / "tensorflow_metrics.json"

RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 65536

FEATURE_COLUMNS = [
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
TARGET_COLUMN = "fare_amount"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TensorFlow model on processed taxi data.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows per TensorFlow batch.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for fast local test runs. Omit to use the full processed dataset.",
    )
    return parser.parse_args()


def batch_to_numpy(batch, columns: list[str]) -> np.ndarray:
    arrays = [
        batch.column(column).to_numpy(zero_copy_only=False).astype(np.float32)
        for column in columns
    ]
    return np.column_stack(arrays)


def compute_feature_stats(batch_size: int, max_rows: int | None) -> dict:
    parquet_file = pq.ParquetFile(PROCESSED_DATASET)
    total_count = 0
    feature_sum = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    feature_squares = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    target_sum = 0.0
    target_squares = 0.0

    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=FEATURE_COLUMNS + [TARGET_COLUMN],
    ):
        if max_rows is not None and total_count >= max_rows:
            break

        features = batch_to_numpy(batch, FEATURE_COLUMNS)
        target = batch.column(TARGET_COLUMN).to_numpy(zero_copy_only=False).astype(np.float32)

        if max_rows is not None:
            remaining_rows = max_rows - total_count
            features = features[:remaining_rows]
            target = target[:remaining_rows]

        row_count = len(target)
        if row_count == 0:
            break

        total_count += row_count
        feature_sum += features.sum(axis=0, dtype=np.float64)
        feature_squares += np.square(features, dtype=np.float64).sum(axis=0)
        target_sum += float(target.sum(dtype=np.float64))
        target_squares += float(np.square(target, dtype=np.float64).sum())

    feature_mean = feature_sum / total_count
    feature_variance = feature_squares / total_count - np.square(feature_mean)
    feature_std = np.sqrt(np.maximum(feature_variance, 1e-12))

    target_mean = target_sum / total_count
    target_variance = target_squares / total_count - target_mean**2
    target_std = float(np.sqrt(max(target_variance, 1e-12)))

    return {
        "row_count": total_count,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


def make_dataset(
    split: str,
    batch_size: int,
    max_rows: int | None,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    target_mean: float,
    target_std: float,
) -> tf.data.Dataset:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    def generator():
        parquet_file = pq.ParquetFile(PROCESSED_DATASET)
        processed_rows = 0

        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=FEATURE_COLUMNS + [TARGET_COLUMN],
        ):
            if max_rows is not None and processed_rows >= max_rows:
                break

            features = batch_to_numpy(batch, FEATURE_COLUMNS)
            target = batch.column(TARGET_COLUMN).to_numpy(zero_copy_only=False).astype(np.float32)

            if max_rows is not None:
                remaining_rows = max_rows - processed_rows
                features = features[:remaining_rows]
                target = target[:remaining_rows]

            row_count = len(target)
            if row_count == 0:
                break

            row_ids = np.arange(processed_rows, processed_rows + row_count)
            processed_rows += row_count

            if split == "test":
                mask = row_ids % 5 == 0
            else:
                mask = row_ids % 5 != 0

            if not np.any(mask):
                continue

            normalized_features = (features[mask] - feature_mean) / feature_std
            normalized_target = (target[mask] - target_mean) / target_std
            yield normalized_features.astype(np.float32), normalized_target.astype(np.float32)

    output_signature = (
        tf.TensorSpec(shape=(None, len(FEATURE_COLUMNS)), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature,
    ).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(len(FEATURE_COLUMNS),)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name="scaled_rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="scaled_mae"),
        ],
    )
    return model


def evaluate_model(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    target_mean: float,
    target_std: float,
) -> dict:
    row_count = 0
    sum_absolute_error = 0.0
    sum_squared_error = 0.0
    sum_target = 0.0
    sum_target_squared = 0.0

    for features, scaled_target in dataset:
        scaled_prediction = model.predict_on_batch(features).reshape(-1)
        target = scaled_target.numpy() * target_std + target_mean
        prediction = scaled_prediction * target_std + target_mean
        error = prediction - target

        row_count += len(target)
        sum_absolute_error += float(np.abs(error).sum(dtype=np.float64))
        sum_squared_error += float(np.square(error, dtype=np.float64).sum())
        sum_target += float(target.sum(dtype=np.float64))
        sum_target_squared += float(np.square(target, dtype=np.float64).sum())

    rmse = math.sqrt(sum_squared_error / row_count)
    mae = sum_absolute_error / row_count
    total_sum_squares = sum_target_squared - (sum_target**2 / row_count)
    r2 = 1.0 - (sum_squared_error / total_sum_squares)

    return {
        "row_count": row_count,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    stats = compute_feature_stats(args.batch_size, args.max_rows)
    row_count = stats["row_count"]
    train_rows = row_count - ((row_count + 4) // 5)
    test_rows = (row_count + 4) // 5

    print(f"Rows available for TensorFlow: {row_count:,}")
    print(f"Training rows: {train_rows:,}")
    print(f"Test rows: {test_rows:,}")

    train_ds = make_dataset(
        split="train",
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
    )
    test_ds = make_dataset(
        split="test",
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        feature_mean=stats["feature_mean"],
        feature_std=stats["feature_std"],
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
    )

    model = build_model()
    steps_per_epoch = math.ceil(row_count / args.batch_size)
    history = model.fit(
        train_ds.repeat(),
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        shuffle=False,
        verbose=2,
    )
    evaluation = evaluate_model(
        model=model,
        dataset=test_ds,
        target_mean=stats["target_mean"],
        target_std=stats["target_std"],
    )
    print(f"Test RMSE: {evaluation['rmse']:.4f}")
    print(f"Test MAE: {evaluation['mae']:.4f}")
    print(f"Test R2: {evaluation['r2']:.4f}")

    metrics = {
        "rows_used": row_count,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "feature_mean": stats["feature_mean"].tolist(),
        "feature_std": stats["feature_std"].tolist(),
        "target_mean": stats["target_mean"],
        "target_std": stats["target_std"],
        "architecture": [
            "Dense(64, relu)",
            "Dense(32, relu)",
            "Dense(1)",
        ],
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "test_metrics": evaluation,
    }

    METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"TensorFlow metrics saved to: {METRICS_FILE}")


if __name__ == "__main__":
    main()
