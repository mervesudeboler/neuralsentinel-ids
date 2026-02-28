"""
NeuralSentinel-IDS — Adversarial Training Loop
Implements WGAN-GP training between Generator and NeuralIDS (Discriminator).
After GAN convergence, uses generated adversarial samples to harden the IDS
via Adversarial Training (the core research contribution of this project).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Dict, List, Optional
import logging

from src.gan.generator import Generator
from src.ids.models import NeuralIDS

logger = logging.getLogger(__name__)


# ─── WGAN-GP Gradient Penalty ─────────────────────────────────────────────────
def gradient_penalty(
    discriminator: NeuralIDS,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    B = real.size(0)
    alpha = torch.rand(B, 1, device=device)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = discriminator(interpolates)
    grad = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_norm = grad.view(B, -1).norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1) ** 2).mean()


# ─── Adversarial Trainer ──────────────────────────────────────────────────────
class AdversarialTrainer:
    """
    Two-phase trainer:
      Phase 1 — WGAN-GP: Generator learns to fool the IDS.
      Phase 2 — Hardening: IDS is retrained on real + adversarial samples.
    """

    def __init__(
        self,
        generator: Generator,
        ids: NeuralIDS,
        device: torch.device,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        latent_dim: int = 64,
        lambda_gp: float = 10.0,
        n_critic: int = 5,
        save_dir: str = "checkpoints",
        epoch_callback: Optional[Callable] = None,  # real-time dashboard update
    ):
        self.G = generator.to(device)
        self.D = ids.to(device)
        self.device = device
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic
        self.save_dir = save_dir
        self.epoch_callback = epoch_callback
        os.makedirs(save_dir, exist_ok=True)

        self.opt_G = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.0, 0.9))
        self.opt_D = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.0, 0.9))
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.opt_G, T_max=50)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.opt_D, T_max=50)

        self.history: Dict[str, List[float]] = {
            "loss_G": [], "loss_D": [], "evasion_rate": [],
            "ids_f1_before": [], "ids_f1_after": [],
        }

    # ── Phase 1: GAN Training ─────────────────────────────────────────────────
    def train_gan(
        self,
        X_real: np.ndarray,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> None:
        logger.info(f"Phase 1: GAN training — {epochs} epochs …")
        dataset = TensorDataset(
            torch.tensor(X_real, dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            epoch_loss_D, epoch_loss_G = [], []

            for step, (real_batch,) in enumerate(loader):
                real_batch = real_batch.to(self.device)
                B = real_batch.size(0)

                # ─ Train Discriminator (n_critic times) ─
                for _ in range(self.n_critic):
                    z = torch.randn(B, self.latent_dim, device=self.device)
                    fake_batch = self.G(z).detach()

                    d_real = self.D(real_batch)
                    d_fake = self.D(fake_batch)
                    gp = gradient_penalty(self.D, real_batch, fake_batch, self.device, self.lambda_gp)
                    loss_D = d_fake.mean() - d_real.mean() + gp

                    self.opt_D.zero_grad()
                    loss_D.backward()
                    self.opt_D.step()
                    epoch_loss_D.append(loss_D.item())

                # ─ Train Generator ─
                z = torch.randn(B, self.latent_dim, device=self.device)
                fake_batch = self.G(z)
                loss_G = -self.D(fake_batch).mean()

                self.opt_G.zero_grad()
                loss_G.backward()
                self.opt_G.step()
                epoch_loss_G.append(loss_G.item())

            self.scheduler_G.step()
            self.scheduler_D.step()

            mean_D = np.mean(epoch_loss_D)
            mean_G = np.mean(epoch_loss_G)
            self.history["loss_D"].append(mean_D)
            self.history["loss_G"].append(mean_G)

            evasion = self._evasion_rate(X_real[:512])
            self.history["evasion_rate"].append(evasion)

            # ── Real-time dashboard update ──────────────────────────────────
            if self.epoch_callback is not None:
                self.epoch_callback(float(mean_G), float(mean_D), float(evasion))

            elapsed = time.time() - t0
            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    f"Epoch {epoch:>4}/{epochs} | "
                    f"D: {mean_D:+.4f} | G: {mean_G:+.4f} | "
                    f"Evasion: {evasion:.2%} | {elapsed:.1f}s"
                )

            if epoch % 25 == 0:
                self._save_checkpoint(epoch)

        logger.info("Phase 1 complete.")

    # ── Phase 2: IDS Hardening ────────────────────────────────────────────────
    def harden_ids(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 30,
        batch_size: int = 256,
        n_adversarial: int = 5000,
    ) -> None:
        """
        Augment training set with adversarial (GAN-generated) samples
        labelled as attacks, then fine-tune the IDS.
        This is the core Adversarial Training step.
        """
        logger.info(f"Phase 2: Hardening IDS with {n_adversarial} adversarial samples …")

        # Generate adversarial samples
        X_adv = self.G.sample_numpy(n_adversarial, self.device)
        y_adv = np.ones(n_adversarial, dtype=np.int64)  # labelled as attacks

        # Combine
        X_aug = np.vstack([X_train, X_adv])
        y_aug = np.concatenate([y_train, y_adv])

        idx = np.random.permutation(len(X_aug))
        X_aug, y_aug = X_aug[idx], y_aug[idx]

        dataset = TensorDataset(
            torch.tensor(X_aug, dtype=torch.float32),
            torch.tensor(y_aug, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        criterion = nn.BCEWithLogitsLoss()

        self.opt_D = optim.Adam(self.D.parameters(), lr=5e-5, betas=(0.9, 0.999))

        for epoch in range(1, epochs + 1):
            self.D.train()
            total_loss = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                # Label smoothing: attacks→0.9, normal→0.1 (reduces overconfidence)
                y_smooth = y_b * 0.9 + (1 - y_b) * 0.1
                logits = self.D(X_b).squeeze(-1)
                loss = criterion(logits, y_smooth)
                self.opt_D.zero_grad()
                loss.backward()
                self.opt_D.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  Hardening epoch {epoch}/{epochs} | loss: {total_loss/len(loader):.4f}")

        logger.info("Phase 2 complete. IDS hardened.")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _evasion_rate(self, X_real: np.ndarray) -> float:
        """Fraction of adversarial samples that bypass the IDS (predicted as normal)."""
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            z = torch.randn(len(X_real), self.latent_dim, device=self.device)
            fake = self.G(z)
            preds = self.D.predict(fake, threshold=0.5)
            evaded = (preds == 0).float().mean().item()
        return evaded

    def _save_checkpoint(self, epoch: int) -> None:
        torch.save({
            "epoch": epoch,
            "generator_state": self.G.state_dict(),
            "ids_state": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "history": self.history,
        }, os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt"))
        logger.info(f"Checkpoint saved → epoch {epoch}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.G.load_state_dict(ckpt["generator_state"])
        self.D.load_state_dict(ckpt["ids_state"])
        self.history = ckpt.get("history", self.history)
        logger.info(f"Checkpoint loaded ← {path}")
