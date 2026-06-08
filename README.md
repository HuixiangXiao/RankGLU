# RankGLU

Residual gated score formation for cross-sectional stock prediction.

RankGLU studies a narrow but important bottleneck in stock ranking models:
after temporal and cross-stock encoders form a stock embedding, the prediction
head decides how that embedding becomes a daily cross-sectional ranking score.
The proposed head keeps a direct linear scoring path and adds a bounded
bottleneck GLU interaction path:

```text
score = Linear(LayerNorm(x))
      + gamma * Linear((Wv LayerNorm(x)) * sigmoid(Wg LayerNorm(x)))
```

This repository contains the core PyTorch implementation, the ranking-aware
training protocol, and the multi-seed experiment runner used for the manuscript
"RankGLU: Residual Gated Score Formation for Cross-Sectional Stock Prediction".

## Main Result

The retained method is RankGLU score formation with the ranking-aware training
protocol:

| Dataset | Model | IC | Best IC | ICIR | RankIC | RankICIR |
|---|---:|---:|---:|---:|---:|---:|
| CSI300 | Original | 0.0654 +/- 0.0052 | 0.0700 | 0.4347 | 0.0696 | 0.4501 |
| CSI300 | Rank-aware | 0.0697 +/- 0.0030 | 0.0745 | 0.4732 | 0.0752 | 0.5028 |
| CSI300 | RankGLU | 0.0727 +/- 0.0037 | 0.0768 | 0.4801 | 0.0814 | 0.5314 |
| CSI800 | Original | 0.0507 +/- 0.0027 | 0.0537 | 0.4305 | 0.0601 | 0.4802 |
| CSI800 | Rank-aware | 0.0502 +/- 0.0018 | 0.0533 | 0.4056 | 0.0642 | 0.4966 |
| CSI800 | RankGLU | 0.0506 +/- 0.0032 | 0.0561 | 0.3975 | 0.0628 | 0.4698 |

On CSI300, RankGLU has the strongest five-seed mean IC among the internally
controlled variants. On CSI800, it remains competitive and has the highest
best-seed IC, but the broader universe gives a more conservative mean
comparison.

## Repository Layout

```text
rankglu/                         Core model and training utilities
scripts/train_rankglu.py          Single-run training entrypoint
scripts/run_multiseed_protocol.py Multi-seed protocol for main, ablation, diagnostic runs
scripts/run_all_multiseed.*       Windows and Linux convenience wrappers
configs/                          Example environment settings
docs/data.md                      Expected processed data layout
docs/reproducibility.md           Reproduction notes
```

Processed data, model weights, logs, and result folders are intentionally not
tracked by Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Smoke Test

The smoke test uses synthetic tensors and does not require market data:

```bash
python scripts/smoke_test.py
```

## Data

Place processed pickle files under `data/opensource/`:

```text
data/opensource/csi300_dl_train.pkl
data/opensource/csi300_dl_test.pkl
data/opensource/csi800_dl_train.pkl
data/opensource/csi800_dl_test.pkl
```

See [docs/data.md](docs/data.md) for the expected tensor layout and data notes.

## Run Experiments

Single RankGLU run:

```bash
python scripts/train_rankglu.py
```

Main five-seed comparison:

```bash
python scripts/run_multiseed_protocol.py --sections main --universe csi300
```

Full protocol:

```bash
bash scripts/run_all_multiseed.sh
```

Windows PowerShell:

```powershell
.\scripts\run_all_multiseed.ps1 -Universe csi300 -Gpu 0
```

Outputs are written to `results/<timestamp>/` or
`results/<universe>_multiseed_<timestamp>/`, including `runs.csv`,
`aggregate.csv`, `core_ablation_delta.csv`, logs, and a text summary.

## Configuration

The default single-run entrypoint is the retained RankGLU setting:

```text
CS_NORM=zscore
LOSS_MODE=mse_ic
IC_WEIGHT=0.1
DECODER_TYPE=residual_bottleneck_glu
DECODER_BOTTLENECK=128
S_SCORE_TYPE=dot
S_VALUE_GATE_TYPE=none
```

Relation-path calibration variants are included as diagnostic stress tests, not
as the retained method. This mirrors the manuscript: relation-path changes can
give high single-seed peaks but are less stable under multi-seed evaluation.

## Citation

```bibtex
@article{xiao2026rankglu,
  title={RankGLU: Residual Gated Score Formation for Cross-Sectional Stock Prediction},
  author={Xiao, Huixiang and Xu, Jian and Qu, Feiyu and Xie, Zixuan and Li, Xiangyu},
  year={2026},
  note={Manuscript}
}
```

Related encoder references are listed in [NOTICE.md](NOTICE.md) for
reproducibility context.

## License

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).

This repository is for research reproducibility and is not investment advice.
