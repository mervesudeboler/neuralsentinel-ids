"""Tests for IDS models."""

import pytest
import torch
import numpy as np
from src.ids.models import NeuralIDS, EnsembleIDS

N_FEATURES = 41
BATCH = 32


@pytest.fixture
def ids():
    return NeuralIDS(N_FEATURES)


def test_neural_ids_forward(ids):
    x = torch.randn(BATCH, N_FEATURES)
    out = ids(x)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"


def test_neural_ids_predict_proba(ids):
    x = torch.randn(BATCH, N_FEATURES)
    proba = ids.predict_proba(x)
    assert proba.shape == (BATCH, 1)
    assert (proba >= 0).all() and (proba <= 1).all(), "Probabilities must be in [0, 1]"


def test_neural_ids_predict_binary(ids):
    x = torch.randn(BATCH, N_FEATURES)
    preds = ids.predict(x)
    assert preds.shape == (BATCH,)
    assert set(preds.tolist()).issubset({0, 1}), "Predictions must be binary"


def test_neural_ids_different_hidden_dims():
    ids2 = NeuralIDS(N_FEATURES, hidden_dims=[64, 32])
    x = torch.randn(BATCH, N_FEATURES)
    out = ids2(x)
    assert out.shape == (BATCH, 1)


def test_ensemble_ids_evaluate():
    ens = EnsembleIDS(N_FEATURES)
    X = np.random.randn(200, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, 200)
    ens.fit_classical(X, y)
    metrics = ens.evaluate(X, y)
    assert "f1_weighted" in metrics
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
