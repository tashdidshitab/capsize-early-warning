#!/usr/bin/env bash
# Reproduces every number and figure in the paper from scratch (2 CPU cores, ~8 GB RAM).
# Approximate wall time: simulation 5 min, features 15 min, nonlinear analysis 10 min,
# experiments E1-E4 ~6 h (deep ensembles dominate).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data results figs
export PYTHONPATH="$PWD/src"
python src/nonlin.py                    # Sec. 3: Floquet charts, continuation, safe basins  -> figs 2-5, results/nonlin.json
python src/gen_dataset.py C11 8000      # Sec. 4: Monte Carlo voyages
python src/gen_dataset.py A2 8000
python src/fig_mc.py                    # Fig. 6, results/mc.json
python src/features.py C11              # Sec. 5: decision instants, observation model, physics indicators
python src/features.py A2
python src/run_experiments.py e1 e3 e2 e4   # Sec. 6: all learning experiments
python src/fig_ml.py                    # Figs. 7-9
python src/single_feature.py           # Table S3
python src/run_e5.py                    # Fig. 10, Table S8 (recalibration)
python src/fig_e5.py                    # Fig. 10
python src/fig_framework.py             # Fig. 1
