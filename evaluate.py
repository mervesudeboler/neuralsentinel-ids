"""
NeuralSentinel-IDS — Evaluation Script

Generates a comprehensive evaluation report comparing:
  • Baseline IDS performance
  • Hardened IDS performance (post adversarial training)
  • GAN evasion rates before/after hardening

Usage:
    python evaluate.py --checkpoint checkpoints/checkpoint_epoch_100.pt
    python evaluate.py --report-dir reports/
"""

import argparse
import os
import json
import torch
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from src.data.preprocessor import NSLKDDPreprocessor
from src.ids.models import NeuralIDS
from src.gan.generator import Generator
from src.utils.logger import setup_logger

logger = setup_logger("neuralsentinel.evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--device", default=None)
    return p.parse_args()


def plot_roc(y_true, y_scores, title: str, path: str):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5), facecolor="#161b22")
    ax = plt.gca()
    ax.set_facecolor("#0d1117")
    ax.plot(fpr, tpr, color="#58a6ff", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#444", linestyle="--")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title(title, color="white")
    ax.legend(loc="lower right", facecolor="#161b22", labelcolor="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"ROC curve → {path}")


def plot_confusion(cm: np.ndarray, path: str):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#161b22")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Attack"], color="white")
    ax.set_yticklabels(["Normal", "Attack"], color="white")
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("True", color="white")
    ax.set_title("Confusion Matrix", color="white")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white", fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix → {path}")


def main():
    args = parse_args()
    os.makedirs(args.report_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # ── Load data ─────────────────────────────────────────────────────────────
    prep = NSLKDDPreprocessor.load(
        os.path.join(cfg["data"]["dir"], "preprocessor.pkl")
    )
    _, _, splits_df = prep.transform(
        os.path.join(cfg["data"]["dir"], "KDDTest+.txt")
    )
    X_test, y_test, _ = prep.transform(
        os.path.join(cfg["data"]["dir"], "KDDTest+.txt")
    )
    n_features = X_test.shape[1]

    # ── Load models ───────────────────────────────────────────────────────────
    ids = NeuralIDS(n_features, hidden_dims=cfg["ids"]["hidden_dims"]).to(device)
    gen = Generator(n_features, cfg["gan"]["latent_dim"],
                    cfg["gan"]["generator_dims"]).to(device)

    ckpt_path = args.checkpoint or os.path.join(
        cfg["training"]["checkpoint_dir"], "neural_ids_final.pt"
    )
    if os.path.exists(ckpt_path):
        ids.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen_path = os.path.join(cfg["training"]["checkpoint_dir"], "generator_final.pt")
    if os.path.exists(gen_path):
        gen.load_state_dict(torch.load(gen_path, map_location=device))

    ids.eval(); gen.eval()

    # ── Evaluate IDS ──────────────────────────────────────────────────────────
    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        proba = ids.predict_proba(X_t).cpu().numpy().flatten()
        preds = (proba >= 0.5).astype(int)

    report = classification_report(y_test, preds, output_dict=True, target_names=["Normal", "Attack"])
    cm = confusion_matrix(y_test, preds)

    logger.info("\n" + classification_report(y_test, preds, target_names=["Normal", "Attack"]))

    # ── GAN Evasion Rate ──────────────────────────────────────────────────────
    n_test = 2000
    z = torch.randn(n_test, cfg["gan"]["latent_dim"], device=device)
    fake = gen(z)
    with torch.no_grad():
        fake_preds = ids.predict(fake).cpu().numpy()
    evasion_rate = (fake_preds == 0).mean()
    logger.info(f"GAN Evasion Rate (post-hardening): {evasion_rate:.2%}")

    # ── Save plots ────────────────────────────────────────────────────────────
    plot_roc(y_test, proba, "IDS ROC Curve",
             os.path.join(args.report_dir, "roc_curve.png"))
    plot_confusion(cm, os.path.join(args.report_dir, "confusion_matrix.png"))

    # ── Save JSON report ──────────────────────────────────────────────────────
    result = {
        "ids_report": report,
        "confusion_matrix": cm.tolist(),
        "gan_evasion_rate": round(float(evasion_rate), 4),
        "n_test_samples": int(len(y_test)),
        "n_adversarial_tested": n_test,
    }
    report_path = os.path.join(args.report_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
