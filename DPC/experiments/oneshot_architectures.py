"""One-shot policy architectures for benchmark sweeps."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from neuromancer.modules.activations import activations
from neuromancer.modules.blocks import MLP, MLP_bounds

from DPC.config import TARGET_PROBS


T = 24
FLAT_INPUT = 26
H_OFFSET = 50.0
H_SCALE = 49.0
V_SCALE = 588000.0
P_SCALE = 100.0
NONLIN = activations["gelu"]


def _normalise_flat_inputs(x, d):
    h_norm = (x[:, 0, 0:1] - H_OFFSET) / H_SCALE
    v_norm = x[:, 0, 1:2] / V_SCALE
    d_norm = d[:, :, 0] / P_SCALE
    return torch.cat([h_norm, v_norm, d_norm], dim=-1)


def _normalise_sequence_inputs(x, d):
    batch = x.size(0)
    h_norm = ((x[:, 0, 0:1] - H_OFFSET) / H_SCALE).unsqueeze(1).expand(batch, T, 1)
    v_norm = (x[:, 0, 1:2] / V_SCALE).unsqueeze(1).expand(batch, T, 1)
    d_norm = d[:, :, 0:1] / P_SCALE
    return torch.cat([d_norm, h_norm, v_norm], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = T):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MLPContinuousPolicy(nn.Module):
    def __init__(self, hidden_sizes=(512, 512, 512)):
        super().__init__()
        self.mlp = MLP_bounds(
            insize=FLAT_INPUT,
            outsize=T * 2,
            hsizes=list(hidden_sizes),
            nonlin=NONLIN,
            min=0.0,
            max=1.0,
        )

    def forward(self, x, d):
        return self.mlp(_normalise_flat_inputs(x, d)).reshape(-1, T, 2)


class MLPDiscretePolicy(nn.Module):
    def __init__(self, hidden_sizes=(512, 512, 512)):
        super().__init__()
        self.mlp = MLP(
            insize=FLAT_INPUT,
            outsize=T * 3,
            hsizes=list(hidden_sizes),
            nonlin=NONLIN,
        )
        self._initialize_mode_bias()

    def _initialize_mode_bias(self):
        target = torch.tensor(TARGET_PROBS, dtype=torch.float32)
        logit_bias = torch.log(target)
        final_layer = self.mlp.linear[-1]
        with torch.no_grad():
            bias = final_layer.bias.data.reshape(T, 3)
            bias[:] = logit_bias
            final_layer.bias.data = bias.reshape(-1)

    def forward(self, x, d):
        return self.mlp(_normalise_flat_inputs(x, d)).reshape(-1, T, 3)


class ConvBackbone(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 3, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        in_channels = 3
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=padding))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_size
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_size

    def forward(self, x):
        y = x.transpose(1, 2)
        y = self.net(y)
        return y.transpose(1, 2)


class LSTMBackbone(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_dim = hidden_size * (2 if bidirectional else 1)

    def forward(self, x):
        y, _ = self.lstm(x)
        return y


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        dim_ff: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pos_enc = PositionalEncoding(hidden_size, max_len=T)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_dim = hidden_size

    def forward(self, x):
        y = self.input_proj(x)
        y = self.pos_enc(y)
        return self.transformer(y)


def _build_backbone(
    architecture: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    nhead: int,
    dim_ff: int,
    cnn_kernel_size: int,
):
    if architecture == "cnn":
        return ConvBackbone(hidden_size=hidden_size, num_layers=max(2, num_layers), kernel_size=cnn_kernel_size, dropout=dropout)
    if architecture == "lstm":
        return LSTMBackbone(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=False)
    if architecture == "bilstm":
        return LSTMBackbone(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=True)
    if architecture == "transformer":
        return TransformerBackbone(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            dim_ff=dim_ff,
        )
    raise ValueError(f"Unknown sequence architecture '{architecture}'")


class SequenceContinuousPolicy(nn.Module):
    def __init__(
        self,
        architecture: str,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        dim_ff: int = 128,
        cnn_kernel_size: int = 3,
    ):
        super().__init__()
        self.backbone = _build_backbone(
            architecture=architecture,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            dim_ff=dim_ff,
            cnn_kernel_size=cnn_kernel_size,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, self.backbone.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x, d):
        features = self.backbone(_normalise_sequence_inputs(x, d))
        return self.head(features)


class SequenceDiscretePolicy(nn.Module):
    def __init__(
        self,
        architecture: str,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 4,
        dim_ff: int = 128,
        cnn_kernel_size: int = 3,
    ):
        super().__init__()
        self.backbone = _build_backbone(
            architecture=architecture,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            nhead=nhead,
            dim_ff=dim_ff,
            cnn_kernel_size=cnn_kernel_size,
        )
        self.head = nn.Sequential(
            nn.Linear(self.backbone.out_dim, self.backbone.out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_dim, 3),
        )
        self._initialize_mode_bias()

    def _initialize_mode_bias(self):
        target = torch.tensor(TARGET_PROBS, dtype=torch.float32)
        logit_bias = torch.log(target)
        final_layer = self.head[-1]
        with torch.no_grad():
            final_layer.bias.copy_(logit_bias)

    def forward(self, x, d):
        features = self.backbone(_normalise_sequence_inputs(x, d))
        return self.head(features)


def build_oneshot_architecture(
    architecture: str = "transformer",
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    nhead: int = 4,
    dim_ff: int = 128,
    cnn_kernel_size: int = 3,
    mlp_hidden_sizes=(512, 512, 512),
):
    """Return one-shot continuous and discrete policies for the selected architecture."""

    architecture = architecture.lower().replace("-", "")
    if architecture == "mlp":
        return MLPContinuousPolicy(hidden_sizes=mlp_hidden_sizes), MLPDiscretePolicy(hidden_sizes=mlp_hidden_sizes)

    if architecture not in {"cnn", "lstm", "bilstm", "transformer"}:
        raise ValueError(f"Unsupported architecture '{architecture}'")

    common = {
        "architecture": architecture,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "nhead": nhead,
        "dim_ff": dim_ff,
        "cnn_kernel_size": cnn_kernel_size,
    }
    return SequenceContinuousPolicy(**common), SequenceDiscretePolicy(**common)
