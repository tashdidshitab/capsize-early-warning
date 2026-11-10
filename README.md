# Real-time capsizing-risk warning for ships in extreme seas

Supplementary material (code, data summaries and results) for the paper

> T.A. Shitab, M.E.H. Emon. *Real-time capsizing-risk warning for ships in extreme seas: a hybrid nonlinear-dynamics and probabilistic machine-learning decision-support framework.* Submitted to *Maritime Transport Research* (2026).

The repository reproduces every number, table and figure in the paper: the nonlinear-dynamics analysis of two benchmark vessels (C11 post-Panamax containership and ITTC A-2 fishing vessel), the Monte Carlo campaigns of 16,000 short-crested-sea voyages, the online physics indicators (Floquet exponent, finite-time Lyapunov exponent, energy-to-separatrix ratio) and the probabilistic early-warning models (PI-TCN-E deep ensemble and five baselines).

## Contents

```
src/                      all code (see "Pipeline" below)
  shiproll.py             1.5-DOF nonlinear roll model, Grim effective wave, JONSWAP short-crested seas,
                          vectorised RK4 integrator
  nonlin.py               Floquet charts, regular-wave response maps, continuation (hysteresis), safe basins
  gen_dataset.py          Monte Carlo voyage campaigns
  features.py             decision instants, sensor/stability-computer error model, physics indicators
  common.py               splits, probabilistic and operational (false alarms/h, detection, lead time) metrics
  models.py               PI-TCN-E and Data-TCN-E deep ensembles (JAX/Optax)
  run_experiments.py      E1 in-distribution, E2 severe-sea extrapolation, E3 cross-vessel, E4 ablation
  run_e5.py               E5 few-voyage recalibration after cross-vessel transfer
  single_feature.py       single-indicator discrimination (Table S3)
  fig_*.py, style.py      figures
data/
  voyage_summary_C11.csv  one row per simulated voyage: Hs, Tp, heading, Froude number, speed,
  voyage_summary_A2.csv   maximum roll, 25-degree exceedance, capsize flag, capsize time
results/                  every number reported in the paper (JSON) and test-set predictions (NPZ)
figs/                     all figures at 300 dpi
docs/                     Supplementary_Material.docx (extended tables, settings, reproducibility map)
run_all.sh                one-command reproduction
```

### Result files

| File | Paper item |
|---|---|
| `results/nonlin.json` | Section 3: Floquet thresholds, hysteresis, integrity factors |
| `results/mc.json` | Section 4: exceedance and capsize statistics |
| `results/e1.json`, `results/e1_preds.npz` | Tables 2–3, Figs. 7–9 (test-set probabilities for every model) |
| `results/e2.json` | Table 4 (severe-sea extrapolation) |
| `results/e3.json`, `results/e3_unc.npz` | Table 5 (zero-shot cross-vessel), Table S7 |
| `results/e5.json` | Fig. 10, Table S8 (few-voyage recalibration) |
| `results/e4.json` | Table 6 (ablation) |
| `results/single_feature.json` | Table S3 |

The full voyage time series (`data/runs_*.npz`, ~370 MB per vessel) and feature files (`data/feat_*.npz`, ~600 MB per vessel) exceed GitHub's file-size limit. They are regenerated deterministically by `gen_dataset.py` and `features.py` (fixed seeds) in about 20 minutes.

## Pipeline

```bash
pip install -r requirements.txt
./run_all.sh
```

Tested with Python 3.13 on a 2-core CPU with 8 GB RAM (no GPU). Approximate run times: nonlinear analysis 10 min, simulation 2 min, features 15 min, learning experiments about 6 h (the deep ensembles dominate). Individual stages can be run separately from the repository root (with `PYTHONPATH=src`) in the order listed in `run_all.sh`.

## Model summary

Equation of motion (per unit roll inertia):

    phi'' + 2*alpha*phi' + gamma*phi'^3 + w0^2 * g(t) * r(phi) = w0^2 * m_d(t)

with a softening righting lever `r(phi) = phi*(1 - |phi/phi_v|^n)`, restoring ratio `g(t) = GM(t)/GM0` driven by the Grim effective wave, and a Froude–Krylov direct moment. Danger threshold: 25 degrees (IMO second-generation intact stability criteria, level 2). Prediction target: exceedance within H = 1, 2 or 4 natural roll periods from a currently safe state.

## Licence and citation

Code and data are released under the MIT licence (see `LICENSE`). Please cite both the paper and this archive (see `CITATION.cff`).

## Contact

Tashdid Ahmed Shitab — tashdid.shitab.ahmed@gmail.com
Department of Naval Architecture and Marine Engineering, Military Institute of Science and Technology (MIST), Dhaka, Bangladesh
