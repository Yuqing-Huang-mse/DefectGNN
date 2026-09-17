"""Predict labeled graph data with a trained DefectGNN model."""

import argparse
import os
import pickle
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="DefectGNN prediction")
    parser.add_argument("--config", default="train.yaml", type=Path)
    parser.add_argument("--graph-data-file", required=True, type=Path)
    parser.add_argument("--targets-data-file", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("predictions"), type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--gpu-id", default="0")
    return parser.parse_args()


ARGS = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if ARGS.device == "cpu" else ARGS.gpu_id

import numpy as np
import pandas as pd
import tensorflow as tf

from defectgnn import registers
from defectgnn.utils.common_util import CommonArgs, setup_imports


def load_graph_paths(csv_path):
    frame = pd.read_csv(csv_path)
    column = "graph_file" if "graph_file" in frame else frame.columns[-1]
    paths = []
    for value in frame[column]:
        path = Path(value)
        candidates = [path, csv_path.parent / path, csv_path.parent / "graphs" / path.name]
        candidates.extend(parent / path for parent in csv_path.parents)
        resolved = next((item.resolve() for item in candidates if item.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"Graph file not found: {value}")
        paths.append([str(resolved)])
    return frame, np.asarray(paths, dtype=object)


def load_model_files(model_dir):
    model_dir = model_dir.resolve()
    best_dir = model_dir / "best" if (model_dir / "best").is_dir() else model_dir
    full_models = list(best_dir.glob("best-full-model-epoch*"))
    weights = best_dir / "best-model"
    normalizer = best_dir / "normalizer.pkl"
    if not full_models:
        raise FileNotFoundError(f"Full model not found in: {best_dir}")
    if not Path(f"{weights}.index").is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}.index")
    if not normalizer.is_file():
        raise FileNotFoundError(f"Normalizer not found: {normalizer}")
    full_model = max(
        full_models,
        key=lambda path: int(path.name.rsplit("epoch", 1)[-1]),
    )
    with normalizer.open("rb") as handle:
        return full_model, weights, pickle.load(handle)


def main():
    setup_imports()
    config = CommonArgs().get_args(str(ARGS.config.resolve()))
    graph_frame, graph_paths = load_graph_paths(ARGS.graph_data_file.resolve())
    full_model, weights, normalizers = load_model_files(ARGS.model_dir)

    container = registers.data_container.get_class(config.names.data_container_name).from_files(
        graph_data_file_list=graph_paths,
        targets_data_file=str(ARGS.targets_data_file.resolve()),
        target_normalizer=normalizers,
        target_types=config.data.target_types,
        target_col=config.data.target_col,
        normalize_labels=True,
        atom_feature_scheme=config.data.atom_feature_scheme,
    )
    provider = registers.data_provider.get_class(config.names.data_provider_name)(
        container, train="[]", validation="[]", test="[]",
        batch_size=ARGS.batch_size, random_seed=config.data.random_seed,
        shuffle=False,
        logging=type("Logger", (), {"info": staticmethod(lambda *_: None)})(),
    )

    model = tf.keras.models.load_model(str(full_model), compile=False)
    model.load_weights(str(weights))

    rows = []
    all_values = {name: [[], []] for name in config.data.target_types}
    sample_count = len(container.graph_data_list)
    for start in range(0, sample_count, ARGS.batch_size):
        indices = np.arange(start, min(start + ARGS.batch_size, sample_count))
        inputs, targets = provider.idx_to_data(indices)
        predictions = model(inputs, training=False)

        for target_type in config.data.target_types:
            target = normalizers[target_type].inverse_transform(
                targets[target_type].numpy().reshape(-1, 1)).reshape(-1)
            prediction = normalizers[target_type].inverse_transform(
                predictions[target_type].numpy().reshape(-1, 1)).reshape(-1)
            all_values[target_type][0].extend(target)
            all_values[target_type][1].extend(prediction)

            counts = [len(container.target_data_list[i]["targets"][target_type]) for i in indices]
            sample_indices = np.repeat(indices, counts)
            for sample_index, true, pred in zip(sample_indices, target, prediction):
                graph_index = int(container.raw_structure_list[int(sample_index)])
                sample_id = graph_frame.iloc[graph_index].get("sample_id", graph_index)
                rows.append({
                    "sample_id": sample_id,
                    "vacancy_sample_index": int(sample_index),
                    "target_type": target_type,
                    "target": float(true),
                    "prediction": float(pred),
                    "residual": float(true - pred),
                })

    metrics = []
    for target_type, (target, prediction) in all_values.items():
        target, prediction = np.asarray(target), np.asarray(prediction)
        residual = target - prediction
        denominator = np.sum((target - target.mean()) ** 2)
        metrics.append({
            "target_type": target_type,
            "n": len(target),
            "mae": np.mean(np.abs(residual)),
            "rmse": np.sqrt(np.mean(residual**2)),
            "r2": 0.0 if denominator == 0 else 1 - np.sum(residual**2) / denominator,
        })

    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ARGS.output_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(ARGS.output_dir / "metrics.csv", index=False)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Device: {ARGS.device}; output: {ARGS.output_dir.resolve()}")


if __name__ == "__main__":
    main()
