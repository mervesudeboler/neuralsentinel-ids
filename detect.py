"""
NeuralSentinel-IDS — Real-Time Detection Script
Simulates a live packet stream from the test set and feeds it to the IDS.
Updates the dashboard state in real time.

Usage:
    python detect.py
    python detect.py --speed 0.05   # seconds between packets
"""

import argparse
import time
import os
import random
import torch
import numpy as np
import yaml

from src.data.preprocessor import NSLKDDPreprocessor
from src.ids.models import NeuralIDS
from src.utils.logger import setup_logger, MetricsTracker

logger = setup_logger("neuralsentinel.detect")

FAKE_IPS = [f"192.168.{random.randint(1,5)}.{random.randint(1,254)}" for _ in range(200)]
ATTACK_CATS = ["DoS", "Probe", "R2L", "U2R", "Adversarial"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--speed", type=float, default=0.1,
                   help="Seconds between packets (lower = faster stream)")
    p.add_argument("--loop", action="store_true",
                   help="Loop the test set indefinitely")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tracker = MetricsTracker(cfg["dashboard"]["state_file"])

    # Load preprocessor and model
    prep = NSLKDDPreprocessor.load(
        os.path.join(cfg["data"]["dir"], "preprocessor.pkl")
    )
    X_test, y_test, df_test = prep.transform(
        os.path.join(cfg["data"]["dir"], "KDDTest+.txt")
    )
    n_features = X_test.shape[1]

    ids = NeuralIDS(n_features, hidden_dims=cfg["ids"]["hidden_dims"]).to(device)
    ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "neural_ids_final.pt")
    if os.path.exists(ckpt):
        ids.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f"Loaded IDS checkpoint ← {ckpt}")
    ids.eval()

    logger.info(f"Starting live detection stream — {len(X_test)} packets …")
    idx = 0

    while True:
        if idx >= len(X_test):
            if args.loop:
                idx = 0
            else:
                break

        x = torch.tensor(X_test[idx:idx+1], dtype=torch.float32).to(device)
        with torch.no_grad():
            proba = ids.predict_proba(x).item()
            pred = int(proba >= 0.5)

        tracker.increment_packets(1)

        if pred == 1:
            cat = df_test.iloc[idx].get("attack_category", random.choice(ATTACK_CATS))
            ip = random.choice(FAKE_IPS)
            tracker.add_alert(cat, ip, proba)
            logger.warning(f"🚨 ATTACK | category={cat} | src={ip} | confidence={proba:.2%}")
        else:
            if idx % 100 == 0:
                logger.info(f"  packets processed: {idx}")

        idx += 1
        time.sleep(args.speed)

    logger.info("Detection stream complete.")


if __name__ == "__main__":
    main()
