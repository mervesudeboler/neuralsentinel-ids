"""
NeuralSentinel-IDS — Structured Logging + Metrics Tracker
"""

import logging
import sys
import json
import os
from datetime import datetime
from typing import Any, Dict


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        level = f"{color}{self.BOLD}{record.levelname:<8}{self.RESET}"
        name = f"\033[90m{record.name}\033[0m"
        msg = record.getMessage()
        return f"[{ts}] {level} {name} · {msg}"


def setup_logger(
    name: str = "neuralsentinel", level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    # File
    os.makedirs("logs", exist_ok=True)
    fh = logging.FileHandler(
        f"logs/neuralsentinel_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(fh)

    logger.propagate = False
    return logger


class MetricsTracker:
    """Lightweight metric tracker that writes to dashboard_state.json."""

    def __init__(self, state_file: str = "data/dashboard_state.json"):
        self.state_file = state_file
        _default: Dict[str, Any] = {
            "history": {"loss_G": [], "loss_D": [], "evasion_rate": []},
            "ids_metrics": {},
            "recent_alerts": [],
            "attack_counts": {},
            "total_packets": 0,
            "attacks_detected": 0,
        }
        os.makedirs("data", exist_ok=True)
        # Load existing state so detect.py doesn't overwrite training results
        if os.path.exists(state_file):
            try:
                with open(state_file) as f:
                    loaded = json.load(f)
                _default.update(loaded)
            except Exception:
                pass
        self._state = _default

    def reset_history(self) -> None:
        """Yeni training başlarken GAN geçmişini sıfırla."""
        self._state["history"] = {"loss_G": [], "loss_D": [], "evasion_rate": []}
        self._state["ids_metrics"] = {}
        self._flush()

    def update_gan(self, loss_G: float, loss_D: float, evasion: float) -> None:
        self._state["history"]["loss_G"].append(round(loss_G, 5))
        self._state["history"]["loss_D"].append(round(loss_D, 5))
        self._state["history"]["evasion_rate"].append(round(evasion, 5))
        self._flush()

    def update_ids_metrics(self, metrics: Dict[str, Any]) -> None:
        self._state["ids_metrics"] = metrics
        self._flush()

    def add_alert(self, category: str, src_ip: str, confidence: float) -> None:
        alert = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "category": category,
            "src_ip": src_ip,
            "confidence": confidence,
        }
        self._state["recent_alerts"].append(alert)
        self._state["recent_alerts"] = self._state["recent_alerts"][-100:]
        cat = self._state["attack_counts"]
        cat[category] = cat.get(category, 0) + 1
        self._state["attacks_detected"] += 1
        self._flush()

    def increment_packets(self, n: int = 1) -> None:
        self._state["total_packets"] += n
        self._flush()

    def _flush(self) -> None:
        merged = dict(self._state)
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file) as f:
                    on_disk = json.load(f)
                # History: keep the longer list (training data must not be lost)
                merged["history"] = {}
                for key in ["loss_G", "loss_D", "evasion_rate"]:
                    disk_list = on_disk.get("history", {}).get(key, [])
                    mem_list = self._state.get("history", {}).get(key, [])
                    merged["history"][key] = (
                        disk_list if len(disk_list) > len(mem_list) else mem_list
                    )
                # ids_metrics: disk wins if memory is empty
                if on_disk.get("ids_metrics") and not self._state.get("ids_metrics"):
                    merged["ids_metrics"] = on_disk["ids_metrics"]
                # Packet counts: memory wins (detect.py owns these)
            except Exception:
                pass
        self._state = merged
        # Atomic write: write to temp file then rename → dashboard never reads partial JSON
        tmp = self.state_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._state, f)
        os.replace(tmp, self.state_file)
