"""
NeuralSentinel-IDS — GAN Generator
Generates synthetic adversarial network traffic that mimics benign traffic
while embedding attack patterns, fooling the IDS (Discriminator).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class AttentionBlock(nn.Module):
    """Self-attention over feature dimensions for better adversarial synthesis."""

    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim // 4)
        self.k = nn.Linear(dim, dim // 4)
        self.v = nn.Linear(dim, dim)
        self.scale = (dim // 4) ** -0.5
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim)
        q = self.q(x)                                                          # (B, dim//4)
        k = self.k(x)                                                          # (B, dim//4)
        gate = torch.sigmoid((q * k * self.scale).sum(dim=-1, keepdim=True))  # (B, 1)
        v = self.v(x)                                                          # (B, dim)
        return self.out(gate * v) + x                                          # residual


class Generator(nn.Module):
    """
    Conditional Generator — takes a noise vector z + optional attack type
    one-hot label and outputs a fake traffic sample in feature space.

    Architecture: Noise → FC blocks (BN + GELU) + Attention → Tanh output
    The output is in [-1, 1] (scaled feature space matching StandardScaler output).
    """

    def __init__(
        self,
        n_features: int,
        latent_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        n_classes: int = 0,          # 0 = unconditional
        use_attention: bool = True,
    ):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.use_attention = use_attention

        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        # Conditional embedding
        cond_dim = n_classes if n_classes > 0 else 0
        in_dim = latent_dim + cond_dim

        blocks: List[nn.Module] = []
        for h_dim in hidden_dims:
            blocks.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim

        self.body = nn.Sequential(*blocks)

        if use_attention:
            self.attn = AttentionBlock(hidden_dims[-1])

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], n_features),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, z: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z      : (B, latent_dim) noise vector
            labels : (B,) long tensor of class indices (optional)
        Returns:
            fake   : (B, n_features) synthetic traffic sample
        """
        if self.n_classes > 0 and labels is not None:
            one_hot = F.one_hot(labels, num_classes=self.n_classes).float()
            x = torch.cat([z, one_hot], dim=-1)
        else:
            x = z

        x = self.body(x)
        if self.use_attention:
            x = self.attn(x)
        return self.head(x)

    # ── Convenience helpers ───────────────────────────────────────────────────
    def sample(
        self,
        n: int,
        device: torch.device,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample n fake traffic vectors."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self(z, labels)

    def sample_numpy(
        self, n: int, device: torch.device, labels: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self.sample(n, device, labels).cpu().numpy()
