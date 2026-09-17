# DefectGNN

Code for training and evaluating the DefectGNN regression model.

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Input data

The training and held-out test graphs are available on
[Figshare](https://doi.org/10.6084/m9.figshare.33869527).

```text
equiatomic_config/
├── train/
│   ├── graph_files.csv
│   ├── targets.pkl
│   ├── graphs/
│   └── split_indices/
└── test/
    ├── graph_files.csv
    ├── targets.pkl
    └── graphs/
```

`graph_files.csv` lists the graph pickle files, and `targets.pkl` stores the
corresponding labels in the same order. Use only trusted pickle files.

## Training

Set `graph_data_file_df`, `targets_data_file`, and `output_path` in `train.yaml`.
For the published 80/10 training/validation split, set `data.train` and
`data.validation` to the supplied `.npy` index files and use the supplied empty
test-index file for `data.test`. Then run:

```bash
python train.py --config train.yaml
```

## Prediction

```bash
python predict.py \
  --config train.yaml \
  --graph-data-file path/to/graph_files.csv \
  --targets-data-file path/to/targets.pkl \
  --model-dir path/to/training_run \
  --output-dir predictions \
  --device gpu
```

Use `--device cpu` for CPU inference. GPU inference uses GPU 0 by default; use
`--gpu-id` to select another device. Results are written to `predictions.csv`
and `metrics.csv`.
