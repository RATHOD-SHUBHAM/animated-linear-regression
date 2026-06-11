import numpy as np


# Loss and gradients for linear regression: y_hat = w * x + b
# Used by model.py during training — no weight updates happen here.


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error with 1/(2n) factor.

    L = (1 / 2n) * sum((y_pred - y_true)^2)

    The 1/2 is a convention so the factor of 2 cancels when differentiating
    the squared term (see compute_gradients).
    """
    n = len(y_true)
    # Sum of squared errors — NOT (sum of errors)^2
    return (1 / (2 * n)) * np.sum((y_pred - y_true) ** 2)


def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float,
) -> tuple[float, float]:
    """
    Gradients of MSE w.r.t. w and b for y_hat = w*x + b.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, 1) or (n_samples,).
    y : np.ndarray
        Target vector, shape (n_samples,).
    w : float
        Weight (slope).
    b : float
        Bias (intercept).

    Returns
    -------
    dw : float
        Gradient of loss w.r.t. w.
    db : float
        Gradient of loss w.r.t. b.
    """
    n = X.shape[0]
    y_pred = _predict(X, w, b)

    # Residual: positive when we over-predict
    error = y_pred - y

    # dL/dw = (1/n) * sum(error * x)   — note 1/n, not 1/(2n)
    # dL/db = (1/n) * sum(error)
    dw = (1 / n) * np.dot(X.flatten(), error)
    db = (1 / n) * np.sum(error)

    return float(dw), float(db)


def _predict(X: np.ndarray, w: float, b: float) -> np.ndarray:
    """
    Forward pass: y_hat = w*x + b.

    Flattens X to (n_samples,) so this works whether X is (n, 1) or (n,).
    """
    return X.flatten() * w + b
