"""Numerical helper utilities shared across ART modules."""

from __future__ import annotations

import numpy as np


EPS = 1e-9


def normalize(x: np.ndarray) -> np.ndarray:
    """Return L2-normalized vector, or zeros when norm is near zero."""

    norm = np.linalg.norm(x)
    if norm < EPS:
        return np.zeros_like(x, dtype=float)
    return x.astype(float) / norm


def cosine_similarity(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between input vector and weight matrix rows."""

    x_norm = normalize(x)
    w_norm = weights / (np.linalg.norm(weights, axis=1, keepdims=True) + EPS)
    return w_norm @ x_norm


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute numerically-stable softmax with temperature scaling."""

    scaled = x / max(temperature, EPS)
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / (np.sum(exp) + EPS)


def compact_vector(x: np.ndarray, precision: int = 2) -> str:
    """Format a vector for concise debug output."""

    return np.array2string(x, precision=precision, suppress_small=True)
