"""Architecture-adapted baselines for the same complete CS2 event law.

These are task-specific MLP, GRU, causal TCN and Transformer baselines, not
implementations of DeepHit or a named temporal point-process method.
Every forward returns logits for bin * 511 + mark - 1, followed by one tail.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# Matched information: frozen neural 272 + A/B site summaries 192 + tree-only 18.
INPUT_DIM = 482
TIME_BINS = 20
MARK_COUNT = 511
EVENT_CLASSES = TIME_BINS * MARK_COUNT
OUTPUT_DIM = EVENT_CLASSES + 1
TAIL_CLASS = EVENT_CLASSES


def _validate(x: Tensor, lengths: Tensor, input_dim: int = INPUT_DIM) -> None:
    if x.ndim != 3 or x.shape[-1] != input_dim:
        raise ValueError(f"Expected [B,L,{input_dim}], got {tuple(x.shape)}")
    if lengths.ndim != 1 or len(lengths) != len(x):
        raise ValueError("lengths must have one entry per sequence")
    if len(x) == 0 or x.shape[1] == 0:
        raise ValueError("Empty input batches and prefixes are not supported")
    if torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
        raise ValueError("Every length must be in [1, padded sequence length]")


def _last(x: Tensor, lengths: Tensor) -> Tensor:
    position = lengths.to(device=x.device, dtype=torch.long) - 1
    return x[torch.arange(len(x), device=x.device), position]


class JointBaseline(nn.Module):
    """Common direct joint-softmax output interface."""

    head: nn.Linear

    def initialize_class_prior(self, probabilities: Tensor) -> None:
        p = torch.as_tensor(probabilities, dtype=self.head.bias.dtype,
                            device=self.head.bias.device)
        if p.shape != (OUTPUT_DIM,) or not torch.isfinite(p).all() or (p <= 0).any():
            raise ValueError("Class initialization requires 10221 positive probabilities")
        p = p / p.sum()
        with torch.no_grad():
            self.head.bias.copy_(p.log())


class MLPJoint(JointBaseline):
    """Current matched causal summary, including site and lag features."""

    def __init__(self, width: int = 128, dropout: float = .1,
                 input_dim: int = INPUT_DIM, **_: Any):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.GELU(),
            nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, width),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(width, OUTPUT_DIM)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        _validate(x, lengths, self.input_dim)
        return self.head(self.encoder(_last(x, lengths)))


class GRUJoint(JointBaseline):
    """Packed complete-prefix recurrent baseline; padding cannot affect state."""

    def __init__(self, width: int = 128, dropout: float = .1,
                 num_layers: int = 2, use_cudnn: bool = False,
                 input_dim: int = INPUT_DIM, **_: Any):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GRU num_layers must be positive")
        # The native kernel avoids an independently reproduced Windows/cuDNN
        # GRU process-exit crash. This backend option does not alter the GRU.
        self.use_cudnn = bool(use_cudnn)
        self.input_dim = input_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.gru = nn.GRU(input_dim, width, num_layers=num_layers,
                          dropout=dropout if num_layers > 1 else 0., batch_first=True)
        self.output_norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, OUTPUT_DIM)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        _validate(x, lengths, self.input_dim)
        packed = pack_padded_sequence(self.input_norm(x), lengths.detach().cpu(),
                                     batch_first=True, enforce_sorted=False)
        with torch.backends.cudnn.flags(
            enabled=self.use_cudnn, benchmark=torch.backends.cudnn.benchmark,
            deterministic=torch.backends.cudnn.deterministic,
            allow_tf32=torch.backends.cudnn.allow_tf32,
        ):
            _, state = self.gru(packed)
        return self.head(self.dropout(self.output_norm(state[-1])))


class _CausalBlock(nn.Module):
    def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv1 = nn.Conv1d(width, width, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(width, width, kernel_size, dilation=dilation)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(F.pad(x, (self.left_padding, 0)))
        h = self.dropout(F.gelu(self.norm1(h.transpose(1, 2)))).transpose(1, 2)
        h = self.conv2(F.pad(h, (self.left_padding, 0)))
        h = self.dropout(F.gelu(self.norm2(h.transpose(1, 2)))).transpose(1, 2)
        return x + h


def tcn_levels_for_prefix(max_length: int, kernel_size: int = 3,
                          minimum: int = 4) -> int:
    """Two convolutions per level: RF=1+2*(k-1)*(2**levels-1)."""
    if max_length < 1 or kernel_size < 2:
        raise ValueError("Invalid prefix length or TCN kernel")
    levels = math.ceil(math.log2(1 + (max_length - 1) / (2 * (kernel_size - 1))))
    return max(minimum, levels)


class TCNJoint(JointBaseline):
    """Causal dilated convolutions with a full-prefix last-state readout.

    The study runner must choose levels from the observed cache maximum. A
    receptive-field check refuses inputs that would silently discard old frames.
    LayerNorm operates per frame, so future padded frames do not alter the past.
    """

    def __init__(self, width: int = 128, dropout: float = .1,
                 num_layers: int = 8, kernel_size: int = 3,
                 input_dim: int = INPUT_DIM, **_: Any):
        super().__init__()
        if num_layers < 1 or kernel_size < 2:
            raise ValueError("TCN needs positive levels and kernel_size >= 2")
        self.input_dim = input_dim
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** num_layers - 1)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([
            _CausalBlock(width, kernel_size, 2 ** i, dropout) for i in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, OUTPUT_DIM)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        _validate(x, lengths, self.input_dim)
        if int(lengths.max().item()) > self.receptive_field:
            raise ValueError(
                f"TCN RF {self.receptive_field} does not cover full prefix "
                f"{int(lengths.max().item())}; increase num_layers, never truncate input"
            )
        h = self.input_projection(self.input_norm(x)).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.head(self.output_norm(_last(h.transpose(1, 2), lengths)))


class TransformerJoint(JointBaseline):
    """Causal complete-prefix Transformer with unbounded sinusoidal positions."""

    def __init__(self, width: int = 128, dropout: float = .1,
                 num_layers: int = 4, num_heads: int = 4,
                 feedforward_multiplier: int = 4, input_dim: int = INPUT_DIM, **_: Any):
        super().__init__()
        if width % num_heads or width % 2:
            raise ValueError("Transformer width must be even and divisible by num_heads")
        self.input_dim = input_dim
        self.width = width
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=num_heads, dim_feedforward=width * feedforward_multiplier,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, OUTPUT_DIM)

    def _positions(self, count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        position = torch.arange(count, device=device, dtype=torch.float32)[:, None]
        scale = torch.exp(torch.arange(0, self.width, 2, device=device, dtype=torch.float32)
                          * (-math.log(10000.) / self.width))
        angles = position * scale[None, :]
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1).to(dtype)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        _validate(x, lengths, self.input_dim)
        count = x.shape[1]
        h = self.input_projection(self.input_norm(x))
        h = self.dropout(h + self._positions(count, x.device, h.dtype)[None, :, :])
        padding_mask = (torch.arange(count, device=x.device)[None, :]
                        >= lengths.to(x.device)[:, None])
        causal_mask = torch.ones(count, count, device=x.device, dtype=torch.bool).triu(1)
        h = self.encoder(h, mask=causal_mask, src_key_padding_mask=padding_mask,
                         is_causal=True)
        return self.head(_last(h, lengths))


def make_model(name: str, config: Mapping[str, Any] | None = None) -> JointBaseline:
    options = dict(config or {})
    key = name.lower().replace("_", "").replace("-", "")
    classes = {"mlp": MLPJoint, "mlpjoint": MLPJoint, "gru": GRUJoint,
               "grujoint": GRUJoint, "tcn": TCNJoint, "tcnjoint": TCNJoint,
               "transformer": TransformerJoint, "transformerjoint": TransformerJoint}
    if key not in classes:
        raise ValueError(f"Unknown baseline {name!r}; choose MLP/GRU/TCN/Transformer")
    return classes[key](**options)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
