# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rankglu import MASTERVariant, ResidualBottleneckGLUDecoder


def main():
    torch.manual_seed(0)
    model = MASTERVariant(
        d_feat=158,
        d_model=64,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.0,
        S_dropout_rate=0.0,
        gate_input_start_index=158,
        gate_input_end_index=221,
        beta=5,
        cs_norm="zscore",
        stronger_head=True,
        decoder_type="residual_bottleneck_glu",
        decoder_bottleneck=32,
    )
    model.eval()
    x = torch.randn(16, 8, 221)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (16,), y.shape
    assert torch.isfinite(y).all()

    head = ResidualBottleneckGLUDecoder(d_model=64, dropout=0.0, bottleneck=32)
    with torch.no_grad():
        score = head(torch.randn(16, 64))
    assert score.shape == (16,), score.shape
    print("smoke_test: ok")


if __name__ == "__main__":
    main()
