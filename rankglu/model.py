import math

import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from .base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, d_model * 0 + max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[: x.shape[1], :]


class ReLUFFN(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class BottleneckGLUFFN(nn.Module):
    def __init__(self, d_model, dropout, bottleneck=None):
        super().__init__()
        hidden = bottleneck if bottleneck is not None else max(d_model // 2, 32)
        if hidden <= 0:
            raise ValueError('ffn bottleneck must be positive.')
        self.value = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.value(x) * torch.sigmoid(self.gate(x))
        hidden = self.dropout(hidden)
        return self.out_dropout(self.out(hidden))


class ResidualBottleneckGLUFFN(nn.Module):
    def __init__(self, d_model, dropout, bottleneck=None):
        super().__init__()
        hidden = bottleneck if bottleneck is not None else max(d_model // 2, 32)
        if hidden <= 0:
            raise ValueError('ffn bottleneck must be positive.')
        self.shortcut = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, d_model)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = self.shortcut(x)
        hidden = self.value(x) * torch.sigmoid(self.gate(x))
        hidden = self.dropout(hidden)
        return shortcut + self.out_dropout(self.out(hidden))


def build_ffn(d_model, dropout, ffn_type='relu', bottleneck=None):
    if ffn_type == 'relu':
        return ReLUFFN(d_model, dropout)
    if ffn_type == 'bottleneck_glu':
        return BottleneckGLUFFN(d_model, dropout, bottleneck=bottleneck)
    if ffn_type == 'residual_bottleneck_glu':
        return ResidualBottleneckGLUFFN(d_model, dropout, bottleneck=bottleneck)
    raise ValueError(f'unsupported ffn_type: {ffn_type}')


class LinearFeatureLayer(nn.Module):
    def __init__(self, d_input, d_model):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)

    def forward(self, x):
        return self.proj(x)


class ResidualBottleneckGLUFeatureLayer(nn.Module):
    def __init__(self, d_input, d_model, bottleneck=None, dropout=0.0):
        super().__init__()
        hidden = bottleneck if bottleneck is not None else max(d_model // 4, 32)
        if hidden <= 0:
            raise ValueError('feature bottleneck must be positive.')
        self.norm = nn.LayerNorm(d_input)
        self.shortcut = nn.Linear(d_input, d_model)
        self.value = nn.Linear(d_input, hidden)
        self.gate = nn.Linear(d_input, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = self.norm(x)
        shortcut = self.shortcut(x)
        hidden = self.value(x) * torch.sigmoid(self.gate(x))
        hidden = self.dropout(hidden)
        return shortcut + self.out(hidden)


def build_feature_layer(d_input, d_model, feature_layer_type='linear', bottleneck=None, dropout=0.0):
    if feature_layer_type == 'linear':
        return LinearFeatureLayer(d_input, d_model)
    if feature_layer_type == 'residual_bottleneck_glu':
        return ResidualBottleneckGLUFeatureLayer(d_input, d_model, bottleneck=bottleneck, dropout=dropout)
    raise ValueError(f'unsupported feature_layer_type: {feature_layer_type}')


def sparsemax(logits, dim=-1):
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = z_sorted.cumsum(dim=dim) - 1.0

    rhos = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.dim()
    view_shape[dim] = -1
    rhos = rhos.view(view_shape)

    support = rhos * z_sorted > z_cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = z_cumsum.gather(dim, support_size.long() - 1) / support_size.to(z.dtype)
    return torch.clamp(z - tau, min=0.0)


def entmax15(logits, dim=-1, n_iter=50):
    z = logits - logits.max(dim=dim, keepdim=True).values
    tau_lo = z.min(dim=dim, keepdim=True).values - 2.0
    tau_hi = z.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        tau_mid = (tau_lo + tau_hi) / 2.0
        probs = torch.clamp((z - tau_mid) / 2.0, min=0.0).pow(2)
        sum_probs = probs.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(sum_probs > 1.0, tau_mid, tau_lo)
        tau_hi = torch.where(sum_probs > 1.0, tau_hi, tau_mid)

    probs = torch.clamp((z - tau_hi) / 2.0, min=0.0).pow(2)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class SAttention(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout,
        score_type='dot',
        score_dot_ratio=0.1,
        attn_norm='softmax',
        ffn_type='relu',
        ffn_bottleneck=None,
        value_gate_type='none',
        value_gate_ratio=1.0,
        attn_res_scale=1.0,
        ffn_res_scale=1.0,
        ffn_res_scale_learnable=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.score_type = score_type
        self.score_dot_ratio = float(score_dot_ratio)
        self.attn_norm = attn_norm
        self.value_gate_type = value_gate_type
        self.value_gate_ratio = float(value_gate_ratio)
        attn_res_scale = float(attn_res_scale)
        self.temperature = math.sqrt(self.d_model / nhead)
        if score_type not in {'dot', 'cosine', 'cosine_dot_hybrid'}:
            raise ValueError(f'unsupported score_type: {score_type}')
        if self.score_dot_ratio < 0.0 or self.score_dot_ratio > 1.0:
            raise ValueError('s score dot ratio must be in [0, 1].')
        if attn_res_scale < 0.0:
            raise ValueError('s attention residual scale must be non-negative.')
        if attn_norm not in {'softmax', 'entmax15', 'sparsemax'}:
            raise ValueError(f'unsupported s attention norm: {attn_norm}')
        self.register_buffer('attn_res_scale', torch.tensor(attn_res_scale))

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        if value_gate_type == 'none':
            self.vgate = None
        elif value_gate_type == 'centered_glu':
            self.vgate = nn.Linear(d_model, d_model)
        else:
            raise ValueError(f'unsupported value_gate_type: {value_gate_type}')

        self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = build_ffn(d_model, dropout, ffn_type=ffn_type, bottleneck=ffn_bottleneck)
        if ffn_res_scale_learnable:
            self.ffn_res_scale = nn.Parameter(torch.tensor(float(ffn_res_scale)))
        else:
            self.register_buffer('ffn_res_scale', torch.tensor(float(ffn_res_scale)))

    def _attention_logits(self, qh, kh):
        dot_logits = torch.matmul(qh, kh.transpose(1, 2)) / self.temperature
        if self.score_type == 'dot':
            return dot_logits
        qh_norm = torch.nn.functional.normalize(qh, p=2, dim=-1, eps=1e-6)
        kh_norm = torch.nn.functional.normalize(kh, p=2, dim=-1, eps=1e-6)
        cosine_logits = self.temperature * torch.matmul(qh_norm, kh_norm.transpose(1, 2))
        if self.score_type == 'cosine':
            return cosine_logits
        return cosine_logits + self.score_dot_ratio * torch.tanh(dot_logits)

    def _attention_norm(self, logits):
        if self.attn_norm == 'softmax':
            return torch.softmax(logits, dim=-1)
        if self.attn_norm == 'entmax15':
            return entmax15(logits, dim=-1)
        return sparsemax(logits, dim=-1)

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)
        if self.vgate is not None:
            centered_gate = 2.0 * torch.sigmoid(self.vgate(x)).transpose(0, 1) - 1.0
            v_gate = 1.0 + self.value_gate_ratio * centered_gate
            v = v * v_gate

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]

            attn = self._attention_norm(self._attention_logits(qh, kh))
            attn = self.attn_dropout[i](attn)
            att_output.append(torch.matmul(attn, vh).transpose(0, 1))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + self.attn_res_scale * att_output
        xt = self.norm2(xt)
        return xt + self.ffn_res_scale * self.ffn(xt)


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, score_type='dot', ffn_type='relu', ffn_bottleneck=None):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.score_type = score_type
        self.temperature = math.sqrt(d_model / nhead)
        if score_type not in {'dot', 'scaled_dot', 'cosine'}:
            raise ValueError(f'unsupported temporal encoder score_type: {score_type}')
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.ModuleList([Dropout(p=dropout) for _ in range(nhead)])
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = build_ffn(d_model, dropout, ffn_type=ffn_type, bottleneck=ffn_bottleneck)

    def _attention_logits(self, qh, kh):
        logits = torch.matmul(qh, kh.transpose(1, 2))
        if self.score_type == 'dot':
            return logits
        if self.score_type == 'scaled_dot':
            return logits / self.temperature
        qh = torch.nn.functional.normalize(qh, p=2, dim=-1, eps=1e-6)
        kh = torch.nn.functional.normalize(kh, p=2, dim=-1, eps=1e-6)
        return self.temperature * torch.matmul(qh, kh.transpose(1, 2))

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim :]
                kh = k[:, :, i * dim :]
                vh = v[:, :, i * dim :]
            else:
                qh = q[:, :, i * dim : (i + 1) * dim]
                kh = k[:, :, i * dim : (i + 1) * dim]
                vh = v[:, :, i * dim : (i + 1) * dim]
            attn = torch.softmax(self._attention_logits(qh, kh), dim=-1)
            attn = self.attn_dropout[i](attn)
            att_output.append(torch.matmul(attn, vh))
        att_output = torch.concat(att_output, dim=-1)

        xt = x + att_output
        xt = self.norm2(xt)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0, residual_alpha=1.0, norm_type='softmax'):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta
        if norm_type not in {'softmax', 'sparsemax', 'entmax15'}:
            raise ValueError(f'unsupported market gate norm_type: {norm_type}')
        self.norm_type = norm_type
        residual_alpha = float(residual_alpha)
        if residual_alpha < 0.0 or residual_alpha > 1.0:
            raise ValueError('market gate alpha must be in [0, 1].')
        self.residual_alpha = residual_alpha

    def _normalize(self, logits):
        if self.norm_type == 'softmax':
            return torch.softmax(logits, dim=-1)
        if self.norm_type == 'sparsemax':
            return sparsemax(logits, dim=-1)
        return entmax15(logits, dim=-1)

    def forward(self, gate_input):
        logits = self.trans(gate_input) / self.t
        raw_gate = self.d_output * self._normalize(logits)
        return 1.0 + self.residual_alpha * (raw_gate - 1.0)


class TemporalAttention(nn.Module):
    def __init__(self, d_model, score_type='dot'):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)
        self.d_model = d_model
        self.score_type = score_type
        self.feature_gate = None
        self.temporal_gate_ratio = 0.0
        self.last_blend_ratio = 0.0
        if score_type not in {'dot', 'cosine'}:
            raise ValueError(f'unsupported temporal score_type: {score_type}')

    def enable_feature_gate(self, d_model, dropout=0.0, gate_ratio=0.1, bottleneck=None):
        hidden = bottleneck if bottleneck is not None else d_model
        if hidden <= 0:
            raise ValueError('temporal gate bottleneck must be positive.')
        gate_ratio = float(gate_ratio)
        if gate_ratio < 0.0 or gate_ratio > 1.0:
            raise ValueError('temporal gate ratio must be in [0, 1].')
        self.feature_gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        nn.init.zeros_(self.feature_gate[-1].weight)
        nn.init.zeros_(self.feature_gate[-1].bias)
        self.temporal_gate_ratio = gate_ratio

    def enable_value_gate(self, d_model, dropout=0.0, gate_ratio=0.1, bottleneck=None):
        self.enable_feature_gate(d_model=d_model, dropout=dropout, gate_ratio=gate_ratio, bottleneck=bottleneck)

    def set_last_blend(self, ratio=0.0):
        ratio = float(ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError('temporal last blend ratio must be in [0, 1].')
        self.last_blend_ratio = ratio

    def _attention_logits(self, h):
        query = h[:, -1, :]
        if self.score_type == 'dot':
            return torch.matmul(h, query.unsqueeze(-1)).squeeze(-1)
        h_norm = torch.nn.functional.normalize(h, p=2, dim=-1, eps=1e-6)
        query_norm = torch.nn.functional.normalize(query, p=2, dim=-1, eps=1e-6)
        return math.sqrt(self.d_model) * torch.matmul(h_norm, query_norm.unsqueeze(-1)).squeeze(-1)

    def forward(self, z):
        h = self.trans(z)
        lam = self._attention_logits(h)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        if self.feature_gate is not None:
            gate = 1.0 + self.temporal_gate_ratio * torch.tanh(self.feature_gate(z))
            z = z * gate
        pooled = torch.matmul(lam, z).squeeze(1)
        if self.last_blend_ratio == 0.0:
            return pooled
        return (1.0 - self.last_blend_ratio) * pooled + self.last_blend_ratio * z[:, -1, :]


class CrossSectionNorm(nn.Module):
    def __init__(self, mode='none', eps=1e-6):
        super().__init__()
        self.mode = mode
        self.eps = eps

    def forward(self, score):
        if self.mode == 'none':
            return score
        centered = score - score.mean(dim=0, keepdim=True)
        if self.mode == 'demean':
            return centered
        if self.mode == 'zscore':
            scale = centered.std(dim=0, keepdim=True, unbiased=False).clamp_min(self.eps)
            return centered / scale
        raise ValueError(f'unsupported cs norm mode: {self.mode}')


class BaselineDecoder(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, x):
        return self.proj(x).squeeze(-1)


class StrongerDecoder(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        return self.head(x).squeeze(-1)


class BottleneckGLUDecoder(nn.Module):
    def __init__(self, d_model, dropout, bottleneck=None, glu_scale=1.0):
        super().__init__()
        hidden = bottleneck if bottleneck is not None else max(d_model // 2, 32)
        if hidden <= 0:
            raise ValueError('bottleneck must be positive.')
        glu_scale = float(glu_scale)
        if glu_scale < 0.0:
            raise ValueError('decoder glu scale must be non-negative.')
        self.norm = nn.LayerNorm(d_model)
        self.value = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        self.register_buffer('glu_scale', torch.tensor(glu_scale))

    def forward(self, x):
        x = self.norm(x)
        hidden = self.value(x) * torch.sigmoid(self.gate(x))
        hidden = self.dropout(hidden)
        return (self.glu_scale * self.out(hidden)).squeeze(-1)


class ResidualBottleneckGLUDecoder(nn.Module):
    def __init__(self, d_model, dropout, bottleneck=None, glu_scale=1.0):
        super().__init__()
        hidden = bottleneck if bottleneck is not None else max(d_model // 2, 32)
        if hidden <= 0:
            raise ValueError('bottleneck must be positive.')
        glu_scale = float(glu_scale)
        if glu_scale < 0.0:
            raise ValueError('decoder glu scale must be non-negative.')
        self.norm = nn.LayerNorm(d_model)
        self.shortcut = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        self.register_buffer('glu_scale', torch.tensor(glu_scale))

    def forward(self, x):
        x = self.norm(x)
        shortcut = self.shortcut(x)
        hidden = self.value(x) * torch.sigmoid(self.gate(x))
        hidden = self.dropout(hidden)
        return (shortcut + self.glu_scale * self.out(hidden)).squeeze(-1)


class ResidualBottleneckSwiGLUDecoder(nn.Module):
    def __init__(self, d_model, dropout, bottleneck=None, glu_scale=1.0):
        super().__init__()
        hidden = bottleneck if bottleneck is not None else max(d_model // 2, 32)
        if hidden <= 0:
            raise ValueError('bottleneck must be positive.')
        glu_scale = float(glu_scale)
        if glu_scale < 0.0:
            raise ValueError('decoder glu scale must be non-negative.')
        self.norm = nn.LayerNorm(d_model)
        self.shortcut = nn.Linear(d_model, 1)
        self.value = nn.Linear(d_model, hidden)
        self.gate = nn.Linear(d_model, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        self.register_buffer('glu_scale', torch.tensor(glu_scale))

    def forward(self, x):
        x = self.norm(x)
        shortcut = self.shortcut(x)
        hidden = self.value(x) * torch.nn.functional.silu(self.gate(x))
        hidden = self.dropout(hidden)
        return (shortcut + self.glu_scale * self.out(hidden)).squeeze(-1)


def build_decoder(d_model, dropout, stronger_head=False, decoder_type=None, decoder_bottleneck=None, decoder_glu_scale=1.0):
    if decoder_type is None:
        decoder_type = 'stronger' if stronger_head else 'baseline'
    if decoder_type == 'baseline':
        return BaselineDecoder(d_model)
    if decoder_type == 'stronger':
        return StrongerDecoder(d_model, dropout)
    if decoder_type == 'bottleneck_glu':
        return BottleneckGLUDecoder(d_model, dropout, bottleneck=decoder_bottleneck, glu_scale=decoder_glu_scale)
    if decoder_type == 'residual_bottleneck_glu':
        return ResidualBottleneckGLUDecoder(d_model, dropout, bottleneck=decoder_bottleneck, glu_scale=decoder_glu_scale)
    if decoder_type == 'residual_bottleneck_swiglu':
        return ResidualBottleneckSwiGLUDecoder(d_model, dropout, bottleneck=decoder_bottleneck, glu_scale=decoder_glu_scale)
    raise ValueError(f'unsupported decoder_type: {decoder_type}')


class RankGLUNetwork(nn.Module):
    def __init__(
        self,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        T_dropout_rate,
        S_dropout_rate,
        gate_input_start_index,
        gate_input_end_index,
        beta,
        market_gate_alpha=1.0,
        market_gate_norm='softmax',
        cs_norm='none',
        stronger_head=False,
        feature_layer_type='linear',
        feature_bottleneck=None,
        feature_dropout=0.0,
        decoder_type=None,
        decoder_bottleneck=None,
        decoder_glu_scale=1.0,
        temporal_agg_type='attention',
        temporal_score_type='dot',
        temporal_gate_ratio=0.1,
        temporal_gate_bottleneck=None,
        temporal_last_blend_ratio=0.0,
        t_score_type='dot',
        t_ffn_type='relu',
        t_ffn_bottleneck=None,
        s_score_type='dot',
        s_score_dot_ratio=0.1,
        s_attn_norm='softmax',
        s_value_gate_type='none',
        s_value_gate_ratio=1.0,
        s_attn_res_scale=1.0,
        s_ffn_type='relu',
        s_ffn_bottleneck=None,
        s_ffn_res_scale=1.0,
        s_ffn_res_scale_learnable=False,
    ):
        super().__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(
            self.d_gate_input,
            d_feat,
            beta=beta,
            residual_alpha=market_gate_alpha,
            norm_type=market_gate_norm,
        )

        self.input_proj = build_feature_layer(
            d_input=d_feat,
            d_model=d_model,
            feature_layer_type=feature_layer_type,
            bottleneck=feature_bottleneck,
            dropout=feature_dropout,
        )
        self.pos_encoder = PositionalEncoding(d_model)
        self.temporal_encoder = TAttention(
            d_model=d_model,
            nhead=t_nhead,
            dropout=T_dropout_rate,
            score_type=t_score_type,
            ffn_type=t_ffn_type,
            ffn_bottleneck=t_ffn_bottleneck,
        )
        self.stock_encoder = SAttention(
            d_model=d_model,
            nhead=s_nhead,
            dropout=S_dropout_rate,
            score_type=s_score_type,
            score_dot_ratio=s_score_dot_ratio,
            attn_norm=s_attn_norm,
            ffn_type=s_ffn_type,
            ffn_bottleneck=s_ffn_bottleneck,
            value_gate_type=s_value_gate_type,
            value_gate_ratio=s_value_gate_ratio,
            attn_res_scale=s_attn_res_scale,
            ffn_res_scale=s_ffn_res_scale,
            ffn_res_scale_learnable=s_ffn_res_scale_learnable,
        )
        self.temporal_pool = TemporalAttention(d_model=d_model, score_type=temporal_score_type)
        self.temporal_pool.set_last_blend(temporal_last_blend_ratio)
        self.decoder = build_decoder(
            d_model=d_model,
            dropout=max(T_dropout_rate, S_dropout_rate),
            stronger_head=stronger_head,
            decoder_type=decoder_type,
            decoder_bottleneck=decoder_bottleneck,
            decoder_glu_scale=decoder_glu_scale,
        )
        if temporal_agg_type == 'attention':
            pass
        elif temporal_agg_type in {'feature_gated', 'value_gated'}:
            self.temporal_pool.enable_value_gate(
                d_model=d_model,
                dropout=max(T_dropout_rate, S_dropout_rate),
                gate_ratio=temporal_gate_ratio,
                bottleneck=temporal_gate_bottleneck,
            )
        else:
            raise ValueError(f'unsupported temporal_agg_type: {temporal_agg_type}')
        self.cs_norm = CrossSectionNorm(mode=cs_norm)

    def forward(self, x):
        src = x[:, :, : self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index : self.gate_input_end_index]
        src = src * self.feature_gate(gate_input).unsqueeze(1)

        hidden = self.input_proj(src)
        hidden = self.pos_encoder(hidden)
        hidden = self.temporal_encoder(hidden)
        hidden = self.stock_encoder(hidden)
        hidden = self.temporal_pool(hidden)
        score = self.decoder(hidden)
        return self.cs_norm(score)


class RankGLUTrainer(SequenceModel):
    def __init__(
        self,
        d_feat,
        d_model,
        t_nhead,
        s_nhead,
        gate_input_start_index,
        gate_input_end_index,
        T_dropout_rate,
        S_dropout_rate,
        beta,
        market_gate_alpha=1.0,
        market_gate_norm='softmax',
        cs_norm='none',
        stronger_head=False,
        feature_layer_type='linear',
        feature_bottleneck=None,
        feature_dropout=0.0,
        decoder_type=None,
        decoder_bottleneck=None,
        decoder_glu_scale=1.0,
        temporal_agg_type='attention',
        temporal_score_type='dot',
        temporal_gate_ratio=0.1,
        temporal_gate_bottleneck=None,
        temporal_last_blend_ratio=0.0,
        t_score_type='dot',
        t_ffn_type='relu',
        t_ffn_bottleneck=None,
        s_score_type='dot',
        s_score_dot_ratio=0.1,
        s_attn_norm='softmax',
        s_value_gate_type='none',
        s_value_gate_ratio=1.0,
        s_attn_res_scale=1.0,
        s_ffn_type='relu',
        s_ffn_bottleneck=None,
        s_ffn_res_scale=1.0,
        s_ffn_res_scale_learnable=False,
        loss_mode='mse',
        ic_weight=0.1,
        **kwargs,
    ):
        self.d_model = d_model
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        self.market_gate_alpha = market_gate_alpha
        self.market_gate_norm = market_gate_norm
        self.cs_norm = cs_norm
        self.stronger_head = stronger_head
        self.feature_layer_type = feature_layer_type
        self.feature_bottleneck = feature_bottleneck
        self.feature_dropout = feature_dropout
        self.decoder_type = decoder_type
        self.decoder_bottleneck = decoder_bottleneck
        self.decoder_glu_scale = decoder_glu_scale
        self.temporal_agg_type = temporal_agg_type
        self.temporal_score_type = temporal_score_type
        self.temporal_gate_ratio = temporal_gate_ratio
        self.temporal_gate_bottleneck = temporal_gate_bottleneck
        self.temporal_last_blend_ratio = temporal_last_blend_ratio
        self.t_score_type = t_score_type
        self.t_ffn_type = t_ffn_type
        self.t_ffn_bottleneck = t_ffn_bottleneck
        self.s_score_type = s_score_type
        self.s_score_dot_ratio = s_score_dot_ratio
        self.s_attn_norm = s_attn_norm
        self.s_value_gate_type = s_value_gate_type
        self.s_value_gate_ratio = s_value_gate_ratio
        self.s_attn_res_scale = s_attn_res_scale
        self.s_ffn_type = s_ffn_type
        self.s_ffn_bottleneck = s_ffn_bottleneck
        self.s_ffn_res_scale = s_ffn_res_scale
        self.s_ffn_res_scale_learnable = s_ffn_res_scale_learnable
        self.loss_mode = loss_mode
        self.ic_weight = ic_weight
        super().__init__(**kwargs)
        self.init_model()

    def init_model(self):
        self.model = RankGLUNetwork(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta,
            market_gate_alpha=self.market_gate_alpha,
            market_gate_norm=self.market_gate_norm,
            cs_norm=self.cs_norm,
            stronger_head=self.stronger_head,
            feature_layer_type=self.feature_layer_type,
            feature_bottleneck=self.feature_bottleneck,
            feature_dropout=self.feature_dropout,
            decoder_type=self.decoder_type,
            decoder_bottleneck=self.decoder_bottleneck,
            decoder_glu_scale=self.decoder_glu_scale,
            temporal_agg_type=self.temporal_agg_type,
            temporal_score_type=self.temporal_score_type,
            temporal_gate_ratio=self.temporal_gate_ratio,
            temporal_gate_bottleneck=self.temporal_gate_bottleneck,
            temporal_last_blend_ratio=self.temporal_last_blend_ratio,
            t_score_type=self.t_score_type,
            t_ffn_type=self.t_ffn_type,
            t_ffn_bottleneck=self.t_ffn_bottleneck,
            s_score_type=self.s_score_type,
            s_score_dot_ratio=self.s_score_dot_ratio,
            s_attn_norm=self.s_attn_norm,
            s_value_gate_type=self.s_value_gate_type,
            s_value_gate_ratio=self.s_value_gate_ratio,
            s_attn_res_scale=self.s_attn_res_scale,
            s_ffn_type=self.s_ffn_type,
            s_ffn_bottleneck=self.s_ffn_bottleneck,
            s_ffn_res_scale=self.s_ffn_res_scale,
            s_ffn_res_scale_learnable=self.s_ffn_res_scale_learnable,
        )
        super().init_model()

    @staticmethod
    def _masked_mse(pred, label):
        mask = ~torch.isnan(label)
        return torch.mean((pred[mask] - label[mask]) ** 2)

    @staticmethod
    def _corr_loss(pred, label, eps=1e-6):
        mask = ~torch.isnan(label)
        pred = pred[mask]
        label = label[mask]
        pred_centered = pred - pred.mean()
        label_centered = label - label.mean()
        denom = torch.sqrt(pred_centered.pow(2).mean().clamp_min(eps)) * torch.sqrt(label_centered.pow(2).mean().clamp_min(eps))
        corr = (pred_centered * label_centered).mean() / denom.clamp_min(eps)
        return 1.0 - corr

    def loss_fn(self, pred, label):
        mse = self._masked_mse(pred, label)
        if self.loss_mode == 'mse':
            return mse
        if self.loss_mode == 'mse_ic':
            return mse + self.ic_weight * self._corr_loss(pred, label)
        raise ValueError(f'unsupported loss mode: {self.loss_mode}')
