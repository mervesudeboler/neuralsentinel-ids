"""
NeuralSentinel-IDS — IDS Model Definitions
Combines a classical Random Forest baseline with a deep neural IDS.
The neural IDS shares its Discriminator role inside the GAN loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
)
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


# ─── Neural IDS (also the GAN Discriminator) ─────────────────────────────────
class NeuralIDS(nn.Module):
    """
    A deep residual network that acts as both the IDS and the GAN Discriminator.
    Input: network traffic feature vector of dimension `n_features`.
    Output: probability [0,1] that the sample is malicious (attack).
    """

    def __init__(self, n_features: int, hidden_dims: Optional[List[int]] = None,
                 dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64, 32]

        self.n_features = n_features
        layers: List[nn.Module] = []

        in_dim = n_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Skip connection (residual) from input to pre-logit
        self.skip = nn.Linear(n_features, hidden_dims[-1])

        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logit (use sigmoid externally for probability)."""
        feat = self.backbone(x)
        skip = self.skip(x)
        out = feat + skip
        return self.head(out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (self.predict_proba(x) >= threshold).long().squeeze(-1)


# ─── Ensemble IDS ─────────────────────────────────────────────────────────────
class EnsembleIDS:
    """
    Production-grade IDS: combines NeuralIDS with Random Forest and
    Gradient Boosting via soft-voting ensemble.
    """

    def __init__(self, n_features: int, device: str = "cpu",
                 neural_weight: float = 0.5):
        self.device = torch.device(device)
        self.neural_weight = neural_weight
        self.rf_weight = (1 - neural_weight) * 0.6
        self.gb_weight = (1 - neural_weight) * 0.4

        self.neural = NeuralIDS(n_features).to(self.device)
        self.rf = RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=42,
        )
        self._rf_fitted = False
        self._gb_fitted = False

    def fit_classical(self, X: np.ndarray, y: np.ndarray) -> None:
        logger.info("Training Random Forest …")
        self.rf.fit(X, y)
        self._rf_fitted = True
        logger.info("Training Gradient Boosting …")
        self.gb.fit(X, y)
        self._gb_fitted = True

    def predict_ensemble(
        self, X: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        self.neural.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            p_neural = self.neural.predict_proba(X_t).cpu().numpy().flatten()

        scores = self.neural_weight * p_neural
        if self._rf_fitted:
            p_rf = self.rf.predict_proba(X)[:, 1]
            scores += self.rf_weight * p_rf
        if self._gb_fitted:
            p_gb = self.gb.predict_proba(X)[:, 1]
            scores += self.gb_weight * p_gb

        return (scores >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        preds = self.predict_ensemble(X)
        report = classification_report(y, preds, output_dict=True)
        cm = confusion_matrix(y, preds)
        try:
            auc = roc_auc_score(y, preds)
        except Exception:
            auc = float("nan")
        f1 = f1_score(y, preds, average="weighted")

        metrics = {
            "f1_weighted": round(f1, 4),
            "roc_auc": round(auc, 4),
            "accuracy": round(report["accuracy"], 4),
            "precision": round(report.get("1", {}).get("precision", 0), 4),
            "recall": round(report.get("1", {}).get("recall", 0), 4),
            "confusion_matrix": cm.tolist(),
        }
        logger.info(f"Evaluation → {metrics}")
        return metrics
