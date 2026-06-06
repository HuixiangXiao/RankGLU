# Reproducibility

## Main Comparison

The main comparison separates three internally controlled variants:

```text
original_backbone:
  MSE objective, no cross-sectional score normalization, linear decoder

ranking_aware_backbone:
  cross-sectional z-score score normalization, MSE-IC objective, stronger MLP decoder

rankglu:
  ranking-aware protocol plus residual bottleneck GLU prediction head
```

Run:

```bash
python scripts/run_multiseed_protocol.py --sections main --main-seeds 0,1,2,3,4 --universe csi300
```

For CSI800:

```bash
python scripts/run_multiseed_protocol.py --sections main --main-seeds 0,1,2,3,4 --universe csi800
```

## Ablation And Diagnostic Runs

The ablation block starts from a relation-path stress setting and removes
components. This is intentionally separate from the retained RankGLU method.

```bash
python scripts/run_multiseed_protocol.py --sections ablation --ablation-seeds 0,1,2
```

Diagnostic runs are single-seed probes:

```bash
python scripts/run_multiseed_protocol.py --sections diagnostic --diagnostic-seed 0
```

## Outputs

Every protocol run writes:

```text
plan.csv
runs.csv
aggregate.csv
core_ablation_delta.csv
final_results.txt
logs/
env/
```

The `env/` snapshots contain only experiment variables selected by the runner,
not the full host environment.

## Runtime

The full protocol is computationally heavy. On a single high-end GPU, the full
CSI300 protocol can take many hours. Use `--sections main` or fewer seeds for
quick verification.
