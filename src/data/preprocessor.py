"""
NeuralSentinel-IDS — Data Preprocessing Module
NSL-KDD dataset loading, cleaning, and feature engineering.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, List
import logging
import pickle

logger = logging.getLogger(__name__)

# ─── NSL-KDD Column Names ────────────────────────────────────────────────────
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]
NUMERIC_FEATURES = [
    f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES + ["label", "difficulty"]
]

# Attack label → category mapping
ATTACK_MAP: Dict[str, str] = {
    "normal": "NORMAL",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "worm": "DoS", "mailbomb": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}

NSL_KDD_URLS = {
    "train": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
    "test":  "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
}


class NSLKDDPreprocessor:
    """
    End-to-end preprocessing pipeline for NSL-KDD dataset.
    Supports binary (normal vs attack) and multi-class classification.
    """

    def __init__(self, data_dir: str = "data", binary: bool = True):
        self.data_dir = data_dir
        self.binary = binary
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.label_encoder = LabelEncoder()
        self.feature_names: List[str] = []
        self.n_features: int = 0
        self._fitted = False

    # ── Download ──────────────────────────────────────────────────────────────
    def download(self) -> None:
        os.makedirs(self.data_dir, exist_ok=True)
        for split, url in NSL_KDD_URLS.items():
            dest = os.path.join(self.data_dir, f"KDD{split.capitalize()}+.txt")
            if not os.path.exists(dest):
                logger.info(f"Downloading {split} split → {dest}")
                urllib.request.urlretrieve(url, dest)
                logger.info("Done.")
            else:
                logger.info(f"{split} split already exists, skipping download.")

    # ── Load raw CSV ──────────────────────────────────────────────────────────
    def _load_raw(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=FEATURE_NAMES)
        df.drop(columns=["difficulty"], inplace=True)
        df["label"] = df["label"].str.rstrip(".")
        return df

    # ── Feature Engineering ───────────────────────────────────────────────────
    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_FEATURES:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                # Handle unseen labels gracefully
                df[col] = df[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        return df

    def _map_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        df["attack_category"] = df["label"].map(ATTACK_MAP).fillna("Unknown")
        if self.binary:
            df["target"] = (df["attack_category"] != "NORMAL").astype(int)
        else:
            target_labels = df["attack_category"].tolist()
            if not self._fitted:
                self.label_encoder.fit(target_labels)
            df["target"] = self.label_encoder.transform(target_labels)
        return df

    def _scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    # ── Public API ────────────────────────────────────────────────────────────
    def fit_transform(
        self, path: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        df = self._load_raw(path)
        df = self._encode_categoricals(df, fit=True)
        df = self._map_labels(df)

        feature_cols = [c for c in df.columns if c not in ["label", "attack_category", "target"]]
        self.feature_names = feature_cols
        self.n_features = len(feature_cols)

        X = self._scale(df[feature_cols].values.astype(np.float32), fit=True)
        y = df["target"].values.astype(np.int64)
        self._fitted = True
        return X, y, df

    def transform(
        self, path: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        assert self._fitted, "Call fit_transform first."
        df = self._load_raw(path)
        df = self._encode_categoricals(df, fit=False)
        df = self._map_labels(df)

        X = self._scale(df[self.feature_names].values.astype(np.float32), fit=False)
        y = df["target"].values.astype(np.int64)
        return X, y, df

    def get_train_val_test(
        self,
        train_path: str,
        test_path: str,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        X_train_full, y_train_full, _ = self.fit_transform(train_path)
        X_test, y_test, _ = self.transform(test_path)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=val_size, random_state=random_state, stratify=y_train_full,
        )
        return {
            "train": (X_train, y_train),
            "val":   (X_val, y_val),
            "test":  (X_test, y_test),
        }

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: str = "data/preprocessor.pkl") -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str = "data/preprocessor.pkl") -> "NSLKDDPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Preprocessor loaded ← {path}")
        return obj
