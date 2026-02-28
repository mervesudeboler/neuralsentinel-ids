"""Tests for GAN Generator."""

import pytest
import torch
from src.gan.generator import Generator

N_FEATURES = 41
LATENT_DIM = 64
BATCH = 16


@pytest.fixture
def gen():
    return Generator(N_FEATURES, LATENT_DIM)


def test_generator_output_shape(gen):
    z = torch.randn(BATCH, LATENT_DIM)
    out = gen(z)
    assert out.shape == (BATCH, N_FEATURES), f"Expected ({BATCH}, {N_FEATURES}), got {out.shape}"


def test_generator_output_range(gen):
    z = torch.randn(BATCH, LATENT_DIM)
    out = gen(z)
    assert out.min() >= -1.01 and out.max() <= 1.01, "Output should be in [-1, 1] (Tanh)"


def test_generator_sample_helper(gen):
    device = torch.device("cpu")
    samples = gen.sample(BATCH, device)
    assert samples.shape == (BATCH, N_FEATURES)


def test_generator_sample_numpy(gen):
    device = torch.device("cpu")
    arr = gen.sample_numpy(BATCH, device)
    assert arr.shape == (BATCH, N_FEATURES)
    assert arr.dtype.kind == "f"


def test_generator_no_nan(gen):
    z = torch.randn(BATCH, LATENT_DIM)
    out = gen(z)
    assert not torch.isnan(out).any(), "Generator output contains NaN"


def test_generator_conditional():
    gen_cond = Generator(N_FEATURES, LATENT_DIM, n_classes=5)
    z = torch.randn(BATCH, LATENT_DIM)
    labels = torch.randint(0, 5, (BATCH,))
    out = gen_cond(z, labels)
    assert out.shape == (BATCH, N_FEATURES)
