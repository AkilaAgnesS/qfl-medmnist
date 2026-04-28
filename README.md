# qfl-medmnist

Experimental code for the paper *Sustainable Quantum Federated Learning for
Medical Image Analysis: A Four-Axis Accounting Protocol* (Sundaresan &
Solomon, under review at *Sustainable Computing: Informatics and Systems*,
2026).

This repository contains the training scripts, configuration files, model
definitions, and per-experiment SUSQA reports for all results in the paper.
The reusable SUSQA logger is published as a separate package at
[github.com/AkilaAgnesS/susqa](https://github.com/AkilaAgnesS/susqa).

## What is in here

- `models/` - the three case-study architectures (C1 classical CNN, C2
  compressed MLP, C3 hybrid CNN-VQC)
- `experiments/` - centralised and federated training drivers
- `data/loaders.py` - MedMNIST v2 loaders for BreastMNIST, PneumoniaMNIST,
  and DermaMNIST
- `configs/` - YAML configurations for every (case study, dataset, setting,
  seed) cell reported in the paper
- `notebooks/` - figure and table generation scripts
- `results/` - per-experiment SUSQA reports as JSON, plus the aggregated
  CSVs used to produce the paper tables and figures

## Three case studies

| Case study | Architecture           | Trainable params P |
|------------|------------------------|--------------------|
| C1         | Classical CNN          | 26,434             |
| C2         | Compressed MLP         | 12,714             |
| C3         | Hybrid CNN-VQC         | 3,530              |

The C3 hybrid uses an 8-qubit, 2-layer variational ansatz with `R_Y` and
`R_Z` rotations and a CNOT entangling ring (`G = 56` gates per forward
pass).

## Three datasets

All from the MedMNIST v2 benchmark suite:

- **BreastMNIST** - 2-class breast ultrasound (benign / malignant)
- **PneumoniaMNIST** - 2-class chest X-ray (normal / pneumonia)
- **DermaMNIST** - 7-class dermoscopy (multi-class, heavily imbalanced)

## Four settings

- Centralised training (single-site baseline)
- FedAvg with IID partitioning across 5 clients
- FedAvg with Dirichlet(alpha = 0.5) heterogeneity
- FedAvg with Dirichlet(alpha = 0.1) extreme heterogeneity

## Install

```bash
git clone https://github.com/AkilaAgnesS/qfl-medmnist.git
cd qfl-medmnist
python -m venv venv
venv\Scripts\activate                  # Windows
pip install -r requirements.txt
pip install git+https://github.com/AkilaAgnesS/susqa.git
```

Requires Python 3.10 or newer.

## Reproduce a single experiment

```bash
# Centralised C3 on BreastMNIST, seed 0
python -m experiments.run_centralized --config configs/baseline_C3_hybrid_breast__seed0.yaml

# Federated C3 on PneumoniaMNIST, IID, seed 0
python -m experiments.run_federated --config configs/fl_C3_hybrid_pneumonia_iid__seed0.yaml
```

Each run writes a SUSQA report to `results/<experiment_id>/susqa_report.json`.

## Reproduce all paper results

The overnight runner script chains every configuration. Expect roughly 60
hours on a single workstation (longer for the DermaMNIST cells):

```bash
python scripts/run_all.py
```

## Aggregate results into the paper tables

```bash
python notebooks/06_effect_sizes.py     # Cohen's d, bootstrap CIs, TOST
python notebooks/07_paper_figures.py    # Pareto plot, convergence plot
```

## Citation

```bibtex
@article{sundaresan2026susqa,
  author  = {Sundaresan, Akila Agnes and Solomon, Appadurai Arun},
  title   = {Sustainable Quantum Federated Learning for Medical Image Analysis: A Four-Axis Accounting Protocol},
  journal = {Sustainable Computing: Informatics and Systems},
  year    = {2026},
  note    = {Under review}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Contact

Akila Agnes Sundaresan, Department of Computer Science and Engineering, GMR
Institute of Technology, Rajam, Andhra Pradesh, India.
[`akila.s@gmrit.edu.in`](mailto:akila.s@gmrit.edu.in) ·
ORCID [0000-0002-3117-6290](https://orcid.org/0000-0002-3117-6290)
