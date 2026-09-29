# OW-CMORL

OW-CMORL (Online Window-Conditioned Multi-Objective Reinforcement Learning) maintains a Pareto solution set while operating conditions change. It encodes a recent window of observed context factors and a short forecast with a temporal Transformer, retrieves context-matched policy snapshots from an expert bank, performs local updates, and merges the resulting solutions into a global Pareto archive. The expert bank supports cosine HNSW retrieval, with exact retrieval as a fallback.

The code covers four dynamic control environments: building, EV charging, cogeneration, and chlor-alkali. Evaluation can use the same 20-regime seed plan across methods and report hypervolume (HV), expected utility (EU), and adaptation metrics.

## Source layout

- `src/dynamic_morl/`: OW-CMORL training, context encoding, expert bank, environment adapters, and evaluation.
- `src/baseline_envs.py`, `src/baseline_eval.py`: shared environment and evaluation adapters for baselines.
- `scripts/run_multienv_dynamic_suite.py`: launch OW-CMORL runs by environment and seed.
- `scripts/reevaluate_dynamic_run_shared_protocol.py`: evaluate a saved run on the shared protocol.
- `scripts/run_shared_protocol_pool.py`: orchestrate the full benchmark, including baselines.
- `scripts/generate_20_regime_report.py`: aggregate completed shared-regime results.
- `CAPQL/`, `PGMORL/`, `Q-Pensieve/`, `MORL-CA/`, and `morl/`: baseline and MORL source code.
- `externals/` and `sustaingym/`: external RL utilities and environment source code.

## Prerequisites

Use Python 3.10 or newer. Install the core Python packages in an environment with a PyTorch build appropriate for your machine:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib torch \
  gym==0.26.2 gymnasium mo-gymnasium 'pymoo<0.6' hnswlib
```

The implementation imports the legacy `pymoo.factory` API, so use a compatible pymoo release. The included baseline and simulator implementations may require additional packages from their upstream projects. If `hnswlib` is unavailable, the expert bank falls back to exact retrieval.

## Train

Preview a single-environment launch without starting training:

```bash
python scripts/run_multienv_dynamic_suite.py \
  --env-key cogen \
  --configs dynamic \
  --seeds 0 \
  --prefix demo \
  --results-root results_shared_protocol \
  --shared-regime-eval-episodes 20 \
  --dry-run
```

Remove `--dry-run` to train. The run above writes to `results_shared_protocol/demo_cogen_dynamic_seed0/`. For a direct OW-CMORL launch with an explicit training budget:

```bash
python -m src.dynamic_morl.run \
  --env-name cogen_dynamic \
  --obj-num 4 \
  --ref-point 0 0 0 0 \
  --auto-ref-point \
  --num-time-steps 16384 \
  --num-init-steps 8192 \
  --num-steps 8 \
  --num-processes 1 \
  --ppo-epoch 2 \
  --num-mini-batch 2 \
  --strict-online-context \
  --shared-regime-eval-episodes 20 \
  --save-dir results/demo_cogen_dynamic_seed0
```

Increase the training budget and use multiple seeds for research comparisons. Run `python -m src.dynamic_morl.run --help` for the full set of context, bank, and training options.

## Evaluate a saved run

Re-evaluate a completed run on the 20-regime shared protocol:

```bash
python scripts/reevaluate_dynamic_run_shared_protocol.py \
  --source-run-dir results_shared_protocol/demo_cogen_dynamic_seed0 \
  --target-run-dir results_shared_protocol/demo_cogen_dynamic_seed0_reeval20 \
  --shared-regime-eval-episodes 20 \
  --shared-regime-seed-offset 0
```

Use a new target directory for each re-evaluation. The target contains `regime_fronts/shared_final.json`, per-regime exports, and a summary under `final/`. To reproduce comparisons, use the same shared plan and seed offset for every method.

## Tests

Run the repository tests from the root directory with:

```bash
python -m pytest -q tests
```

The memory-ablation and HV rescoring tests are skipped when the optional `pymoo` dependency is not installed.

## Publication scope

This repository tracks Python source, this README, and `.gitignore` only. Datasets, checkpoints, logs, figures, tables, reports, and other generated artifacts remain local.
