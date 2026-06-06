# -*- coding: utf-8 -*-
import os
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rankglu import RankGLUTrainer

raw_universe = os.getenv('UNIVERSE', 'csi300').lower()
universe = {'cs300': 'csi300', 'cs800': 'csi800'}.get(raw_universe, raw_universe)
prefix = os.getenv('PREFIX', 'opensource')
variant_name = os.getenv('VARIANT_NAME', 'rankglu')
cs_norm = os.getenv('CS_NORM', 'zscore')
loss_mode = os.getenv('LOSS_MODE', 'mse_ic')
ic_weight = float(os.getenv('IC_WEIGHT', '0.1'))
stronger_head = os.getenv('STRONGER_HEAD', '1') == '1'
feature_layer_type = os.getenv('FEATURE_LAYER_TYPE', 'linear')
decoder_type = os.getenv('DECODER_TYPE', 'residual_bottleneck_glu')
t_score_type = os.getenv('T_SCORE_TYPE', 'dot')
t_ffn_type = os.getenv('T_FFN_TYPE', 'relu')
s_score_type = os.getenv('S_SCORE_TYPE', 'dot')
s_score_dot_ratio = float(os.getenv('S_SCORE_DOT_RATIO', '0.1'))
s_attn_norm = os.getenv('S_ATTN_NORM', 'softmax')
s_value_gate_type = os.getenv('S_VALUE_GATE_TYPE', 'none')
s_ffn_type = os.getenv('S_FFN_TYPE', 'relu')
s_ffn_res_scale_learnable = os.getenv('S_FFN_RES_SCALE_LEARNABLE', '0') == '1'

train_data_dir = REPO_ROOT / 'data'
train_path = train_data_dir / prefix / f'{universe}_dl_train.pkl'
test_path = train_data_dir / prefix / f'{universe}_dl_test.pkl'
if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        f'Missing processed data. Expected {train_path} and {test_path}. '
        'See docs/data.md for the expected layout.'
    )

with open(train_path, 'rb') as f:
    dl_train = pickle.load(f)
with open(test_path, 'rb') as f:
    dl_test = pickle.load(f)

print('Data Loaded.', flush=True)

d_feat = 158
d_model = int(os.getenv('D_MODEL', '256'))
t_nhead = int(os.getenv('T_NHEAD', '4'))
s_nhead = int(os.getenv('S_NHEAD', '2'))
dropout = float(os.getenv('DROPOUT', '0.5'))
feature_bottleneck = int(os.getenv('FEATURE_BOTTLENECK', '64'))
feature_dropout = float(os.getenv('FEATURE_DROPOUT', '0.0'))
decoder_bottleneck = int(os.getenv('DECODER_BOTTLENECK', '128'))
decoder_glu_scale = float(os.getenv('DECODER_GLU_SCALE', os.getenv('DECODER_GLU_SCAL', '1.0')))
temporal_agg_type = os.getenv('TEMPORAL_AGG_TYPE', 'attention')
temporal_score_type = os.getenv('TEMPORAL_SCORE_TYPE', 'dot')
temporal_gate_ratio = float(os.getenv('TEMPORAL_GATE_RATIO', '0.1'))
temporal_gate_bottleneck_env = os.getenv('TEMPORAL_GATE_BOTTLENECK', '')
temporal_gate_bottleneck = int(temporal_gate_bottleneck_env) if temporal_gate_bottleneck_env else None
temporal_last_blend_ratio = float(os.getenv('TEMPORAL_LAST_BLEND_RATIO', '0.0'))
t_ffn_bottleneck = int(os.getenv('T_FFN_BOTTLENECK', '170'))
s_ffn_bottleneck = int(os.getenv('S_FFN_BOTTLENECK', '170'))
s_value_gate_ratio = float(os.getenv('S_VALUE_GATE_RATIO', '0.0'))
s_attn_res_scale = float(os.getenv('S_ATTN_RES_SCALE', '1.0'))
s_ffn_res_scale = float(os.getenv('S_FFN_RES_SCALE', '1.0'))
gate_input_start_index = 158
gate_input_end_index = 221
beta = 5 if universe == 'csi300' else 2
market_gate_alpha = float(os.getenv('MARKET_GATE_ALPHA', '1.0'))
market_gate_norm = os.getenv('MARKET_GATE_NORM', 'softmax')
n_epoch = int(os.getenv('N_EPOCH', '40'))
lr = float(os.getenv('LR', '1e-5'))
GPU = int(os.getenv('GPU', '0'))
seed = int(os.getenv('SEED', '0'))
model_dir = REPO_ROOT / 'model'
model_dir.mkdir(parents=True, exist_ok=True)

print(
    f'universe={universe} | prefix={prefix} | Variant={variant_name} | cs_norm={cs_norm} | loss_mode={loss_mode} | ic_weight={ic_weight} | stronger_head={stronger_head} | feature_layer_type={feature_layer_type} | feature_bottleneck={feature_bottleneck} | feature_dropout={feature_dropout} | decoder_type={decoder_type} | decoder_bottleneck={decoder_bottleneck} | decoder_glu_scale={decoder_glu_scale} | market_gate_alpha={market_gate_alpha} | market_gate_norm={market_gate_norm} | temporal_agg_type={temporal_agg_type} | temporal_score_type={temporal_score_type} | temporal_gate_ratio={temporal_gate_ratio} | temporal_gate_bottleneck={temporal_gate_bottleneck} | temporal_last_blend_ratio={temporal_last_blend_ratio} | t_score_type={t_score_type} | t_ffn_type={t_ffn_type} | t_ffn_bottleneck={t_ffn_bottleneck} | s_score_type={s_score_type} | s_score_dot_ratio={s_score_dot_ratio} | s_attn_norm={s_attn_norm} | s_value_gate_type={s_value_gate_type} | s_value_gate_ratio={s_value_gate_ratio} | s_attn_res_scale={s_attn_res_scale} | s_ffn_type={s_ffn_type} | s_ffn_bottleneck={s_ffn_bottleneck} | s_ffn_res_scale={s_ffn_res_scale} | s_ffn_res_scale_learnable={s_ffn_res_scale_learnable}',
    flush=True,
)

model = RankGLUTrainer(
    d_feat=d_feat,
    d_model=d_model,
    t_nhead=t_nhead,
    s_nhead=s_nhead,
    T_dropout_rate=dropout,
    S_dropout_rate=dropout,
    beta=beta,
    market_gate_alpha=market_gate_alpha,
    market_gate_norm=market_gate_norm,
    gate_input_end_index=gate_input_end_index,
    gate_input_start_index=gate_input_start_index,
    cs_norm=cs_norm,
    stronger_head=stronger_head,
    feature_layer_type=feature_layer_type,
    feature_bottleneck=feature_bottleneck,
    feature_dropout=feature_dropout,
    decoder_type=decoder_type,
    decoder_bottleneck=decoder_bottleneck,
    decoder_glu_scale=decoder_glu_scale,
    temporal_agg_type=temporal_agg_type,
    temporal_score_type=temporal_score_type,
    temporal_gate_ratio=temporal_gate_ratio,
    temporal_gate_bottleneck=temporal_gate_bottleneck,
    temporal_last_blend_ratio=temporal_last_blend_ratio,
    t_score_type=t_score_type,
    t_ffn_type=t_ffn_type,
    t_ffn_bottleneck=t_ffn_bottleneck,
    s_score_type=s_score_type,
    s_score_dot_ratio=s_score_dot_ratio,
    s_attn_norm=s_attn_norm,
    s_value_gate_type=s_value_gate_type,
    s_value_gate_ratio=s_value_gate_ratio,
    s_attn_res_scale=s_attn_res_scale,
    s_ffn_type=s_ffn_type,
    s_ffn_bottleneck=s_ffn_bottleneck,
    s_ffn_res_scale=s_ffn_res_scale,
    s_ffn_res_scale_learnable=s_ffn_res_scale_learnable,
    loss_mode=loss_mode,
    ic_weight=ic_weight,
    n_epochs=n_epoch,
    lr=lr,
    GPU=GPU,
    seed=seed,
    train_stop_loss_thred=-1,
    save_path=str(model_dir),
    save_prefix=f'{universe}_{prefix}_{variant_name}',
)

train_loader = model._init_data_loader(dl_train, shuffle=True, drop_last=True)
best_test_ic = -1e9
best_epoch = -1
start = time.time()

for epoch in range(n_epoch):
    train_loss = model.train_epoch(train_loader)
    model.fitted = epoch
    _, test_metrics = model.predict(dl_test)
    if test_metrics['IC'] > best_test_ic:
        best_test_ic = test_metrics['IC']
        best_epoch = epoch
    print(
        'Epoch {epoch:02d} | train_loss {train_loss:.6f} | test_ic {test_ic:.4f} | test_icir {test_icir:.4f} | test_ric {test_ric:.4f} | test_ricir {test_ricir:.4f}'.format(
            epoch=epoch,
            train_loss=train_loss,
            test_ic=test_metrics['IC'],
            test_icir=test_metrics['ICIR'],
            test_ric=test_metrics['RIC'],
            test_ricir=test_metrics['RICIR'],
        ),
        flush=True,
    )

print('Best test IC {:.4f} at epoch {}'.format(best_test_ic, best_epoch), flush=True)
print('Total time {:.2f}s'.format(time.time() - start), flush=True)
