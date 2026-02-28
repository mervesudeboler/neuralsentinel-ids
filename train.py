"""
NeuralSentinel-IDS — Main Training Entry Point

Usage:
    python train.py                        # default config
    python train.py --config config/config.yaml
    python train.py --phase gan            # Phase 1 only
    python train.py --phase harden        # Phase 2 only
    python train.py --device cuda
"""

import argparse
import os
import sys
import torch
import yaml
import numpy as np

from src.data.preprocessor import NSLKDDPreprocessor
from src.ids.models import NeuralIDS, EnsembleIDS
from src.gan.generator import Generator
from src.gan.trainer import AdversarialTrainer
from src.utils.logger import setup_logger, MetricsTracker

logger = setup_logger("neuralsentinel.train")


def parse_args():
    parser = argparse.ArgumentParser(description="NeuralSentinel-IDS Training")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--phase", choices=["all", "gan", "harden", "classical"],
                        default="all")
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto-detected if omitted)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Device: {device}")

    tracker = MetricsTracker(cfg["dashboard"]["state_file"])

    # ── Data ──────────────────────────────────────────────────────────────────
    prep = NSLKDDPreprocessor(
        data_dir=cfg["data"]["dir"],
        binary=cfg["data"]["binary"],
    )
    prep.download()

    logger.info("Loading and preprocessing dataset …")
    splits = prep.get_train_val_test(
        train_path=os.path.join(cfg["data"]["dir"], "KDDTrain+.txt"),
        test_path=os.path.join(cfg["data"]["dir"], "KDDTest+.txt"),
        val_size=cfg["data"]["val_size"],
    )
    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]
    X_test,  y_test  = splits["test"]
    n_features = X_train.shape[1]

    prep.save(os.path.join(cfg["data"]["dir"], "preprocessor.pkl"))
    logger.info(f"Features: {n_features} | Train: {len(X_train)} | "
                f"Val: {len(X_val)} | Test: {len(X_test)}")

    # ── Models ────────────────────────────────────────────────────────────────
    neural_ids = NeuralIDS(
        n_features=n_features,
        hidden_dims=cfg["ids"]["hidden_dims"],
        dropout=cfg["ids"]["dropout"],
    ).to(device)

    generator = Generator(
        n_features=n_features,
        latent_dim=cfg["gan"]["latent_dim"],
        hidden_dims=cfg["gan"]["generator_dims"],
        use_attention=cfg["gan"]["use_attention"],
    ).to(device)

    trainer = AdversarialTrainer(
        generator=generator,
        ids=neural_ids,
        device=device,
        lr_g=cfg["gan"]["lr_g"],
        lr_d=cfg["gan"]["lr_d"],
        latent_dim=cfg["gan"]["latent_dim"],
        lambda_gp=cfg["gan"]["lambda_gp"],
        n_critic=cfg["gan"]["n_critic"],
        save_dir=cfg["training"]["checkpoint_dir"],
    )

    # ── Classical IDS baseline ────────────────────────────────────────────────
    if args.phase in ("all", "classical"):
        ensemble = EnsembleIDS(n_features=n_features, device=device_str,
                               neural_weight=cfg["ids"]["neural_weight"])
        logger.info("Training classical ensemble baselines …")
        ensemble.fit_classical(X_train, y_train)
        baseline_metrics = ensemble.evaluate(X_test, y_test)
        logger.info(f"Baseline metrics: {baseline_metrics}")
        tracker.update_ids_metrics(baseline_metrics)

    # ── Phase 1: GAN ──────────────────────────────────────────────────────────
    if args.phase in ("all", "gan"):
        # Train on benign-only samples to teach Generator to mimic normal traffic
        X_normal = X_train[y_train == 0]
        logger.info(f"GAN training on {len(X_normal)} normal samples …")
        trainer.train_gan(
            X_real=X_normal,
            epochs=cfg["gan"]["epochs"],
            batch_size=cfg["gan"]["batch_size"],
        )
        # Push history to dashboard
        for i, (lg, ld, er) in enumerate(zip(
            trainer.history["loss_G"],
            trainer.history["loss_D"],
            trainer.history["evasion_rate"],
        )):
            tracker.update_gan(lg, ld, er)

    # ── Phase 2: Hardening ────────────────────────────────────────────────────
    if args.phase in ("all", "harden"):
        logger.info("Hardening IDS with adversarial samples …")
        trainer.harden_ids(
            X_train=X_train,
            y_train=y_train,
            epochs=cfg["training"]["harden_epochs"],
            batch_size=cfg["training"]["batch_size"],
            n_adversarial=cfg["training"]["n_adversarial"],
        )

        # Evaluate hardened IDS
        neural_ids.eval()
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = neural_ids.predict(X_t).cpu().numpy()
        from sklearn.metrics import f1_score, roc_auc_score
        f1 = f1_score(y_test, preds, average="weighted")
        auc = roc_auc_score(y_test, preds)
        hardened_metrics = {"f1_weighted": round(f1, 4), "roc_auc": round(auc, 4)}
        logger.info(f"Hardened IDS metrics: {hardened_metrics}")
        tracker.update_ids_metrics(hardened_metrics)

    # ── Save final models ─────────────────────────────────────────────────────
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    torch.save(neural_ids.state_dict(),
               os.path.join(cfg["training"]["checkpoint_dir"], "neural_ids_final.pt"))
    torch.save(generator.state_dict(),
               os.path.join(cfg["training"]["checkpoint_dir"], "generator_final.pt"))
    logger.info("Training complete. Models saved.")


if __name__ == "__main__":
    main()
