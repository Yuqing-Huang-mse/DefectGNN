# DefectGNN training code

This repository contains the DefectGNN model implementation and the code required
to train the regression models reported in the accompanying paper. Raw and
processed datasets are not included.

## Repository layout

- `defectgnn/models/`: the DefectGNN architecture and message-passing layers.
- `defectgnn/trainers/`: regression and GradNorm training logic.
- `defectgnn/tasks/`: training loop, validation, checkpointing, and prediction export.
- `defectgnn/data/`: data containers and TensorFlow input pipeline.
- `configs/train.yaml`: example configuration for vacancy-aware atom-level training.
- `train.py`: command-line training entry point.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

The code uses TensorFlow and TensorFlow Addons. For reproducible archival use,
record the exact package versions and CUDA/cuDNN versions from the environment
used for the paper before publishing a release.

## Input data

Training uses two inputs configured under `paths` in `configs/train.yaml`:

1. `graph_data_file_df`: a CSV file whose first column is an index and whose
   remaining value column contains one path per row to a pickled graph
   dictionary.
2. `targets_data_file`: a pickle file containing a list of target dictionaries,
   in the same order as the graph files listed by the CSV.

Each graph dictionary contains NumPy-compatible arrays. The core fields used by
the model are:

- `atom_features_list`: node-feature matrix with shape `(n_atoms, n_features)`;
- `id_i_list`, `id_j_list`: directed edge endpoint indices;
- `dist_list`: edge distances;
- `angle_mij_list`: triplet angles;
- `bond_mi_id_for_angle_mij_list`, `bond_ij_id_for_angle_mij_list`: indices
  mapping triplets to directed edges;
- `id_swap`: the reverse-edge index for each directed edge.

Each target dictionary has the following structure. The example configuration
uses the `atom` entries; `path` entries are only required when path-level
prediction is enabled:

```python
{
    "targets": {"atom": atom_level_values, "path": path_level_values},
    "reduce_to_target_indices": {
        "atom": atom_indices,
        "path": directed_edge_indices,
    },
}
```

Arrays may be NumPy arrays or other values accepted by `numpy.asarray`. Pickle
files must only be loaded from trusted sources because Python pickle is not a
safe interchange format for untrusted data.

## Training

Edit the three paths at the top of `configs/train.yaml`, then run from the
repository root:

```bash
python train.py --config configs/train.yaml
```

The task loads graph/target pairs, constructs samples, creates the configured
train/validation/test split, trains the selected DefectGNN model, and writes logs,
split indices, checkpoints, metrics, and predictions below `paths.output_path`.
The example uses the vacancy-aware DefectGNN and atom-level regression. The base
data container converts each pristine graph into vacancy-centred training
samples using the atom indices supplied in the target dictionary.

