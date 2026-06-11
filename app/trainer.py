"""
Training wrapper for the Streamlit app.

Uses src.model and src.loss without modifying them.
Supports custom weight initialization (not available in model.fit()).
"""

from collections.abc import Callable

import numpy as np

from src.loss import compute_gradients, mse
from src.model import LinearRegressionGD

# Called after each epoch: (epoch, loss, w, b, history)
EpochCallback = Callable[[int, float, float, float, dict[str, list]], None]


def init_model(
    lr: float = 0.01,
    n_epochs: int = 200,
    init_w: float = 0.0,
    init_b: float = 0.0,
    random_init: bool = False,
    convergence_tol: float = 1e-6,
) -> LinearRegressionGD:
    """Create a model with initialized weights and empty history."""
    model = LinearRegressionGD(
        lr=lr,
        n_epochs=n_epochs,
        convergence_tol=convergence_tol,
    )
    if random_init:
        model.w = float(np.random.randn() * 0.01)
        model.b = 0.0
    else:
        model.w = float(init_w)
        model.b = float(init_b)
    model.history = {"loss": [], "w": [], "b": []}
    return model


def run_one_epoch(
    model: LinearRegressionGD,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
) -> tuple[float, bool]:
    """
    Run a single gradient-descent epoch.

    Returns (loss, converged_early).
    """
    y_pred = model.predict(X)
    loss = mse(y, y_pred)
    dw, db = compute_gradients(X, y, model.w, model.b)

    model.w -= lr * dw
    model.b -= lr * db

    model.history["loss"].append(loss)
    model.history["w"].append(model.w)
    model.history["b"].append(model.b)

    converged = False
    if len(model.history["loss"]) > 1:
        prev = model.history["loss"][-2]
        converged = abs(loss - prev) < model.convergence_tol

    return loss, converged


def train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    n_epochs: int = 200,
    init_w: float = 0.0,
    init_b: float = 0.0,
    random_init: bool = False,
    convergence_tol: float = 1e-6,
    on_epoch: EpochCallback | None = None,
) -> LinearRegressionGD:
    """
    Train linear regression (batch loop — use run_one_epoch for Streamlit live UI).

    If on_epoch is provided, it is called after each epoch.
    """
    model = init_model(
        lr=lr,
        n_epochs=n_epochs,
        init_w=init_w,
        init_b=init_b,
        random_init=random_init,
        convergence_tol=convergence_tol,
    )

    for epoch in range(n_epochs):
        loss, converged = run_one_epoch(model, X, y, lr)

        if on_epoch is not None:
            on_epoch(epoch, loss, model.w, model.b, model.history)

        if converged:
            break

    return model
