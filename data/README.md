# Data

MedMNIST v2 datasets used in this work. Files are not committed (see `.gitignore`); they are downloaded on first run.

## Datasets

| Dataset | Task | Train / Val / Test | Image size | Notes |
|---|---|---|---|---|
| BreastMNIST | Binary (malignant vs benign) | 546 / 78 / 156 | 28×28 (also 64, 128, 224 in v2) | Class-imbalanced; small |
| PneumoniaMNIST | Binary (pneumonia vs normal) | 4708 / 524 / 624 | 28×28 (also 64, 128, 224) | Larger, more standard |

## Download

```python
from medmnist import BreastMNIST, PneumoniaMNIST
BreastMNIST(split='train', download=True, root='data/raw')
PneumoniaMNIST(split='train', download=True, root='data/raw')
```

## Processed splits

`fl/partition.py` produces deterministic IID and Dirichlet(α) partitions in `data/processed/<dataset>/<partition>/<client_id>.npz`. Seeds in the config control these.

## Citation

Yang, Shi, Wei, Liu, Zhao, Ke, Pfister, Ni. *MedMNIST v2 — A large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific Data 10:41, 2023.
