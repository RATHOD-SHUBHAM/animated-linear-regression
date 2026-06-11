import numpy as np

from src.loss import _predict, compute_gradients, mse


class LinearRegressionGD:
    """
    Linear regression trained with batch gradient descent.

    Uses mse() and compute_gradients() from loss.py — no autograd.
    Records loss and (w, b) each epoch in self.history for visualization.
    """

    def __init__(
        self,
        lr: float = 0.01,
        n_epochs: int = 200,
        convergence_tol: float = 1e-6,
    ):
        # lr (learning rate α): step size for each weight update.
        # Too large → loss oscillates or diverges; too small → slow convergence.
        self.lr = lr

        # n_epochs: how many full passes over the training data (batch GD).
        self.n_epochs = n_epochs

        # convergence_tol: stop training when loss change between epochs is tiny.
        self.convergence_tol = convergence_tol

        # w (weight/slope) and b (bias/intercept) — learned parameters.
        # None until fit() runs.
        self.w: float | None = None
        self.b: float | None = None

        # Populated each epoch during fit() — used by visualize.py
        self.history: dict[str, list] = {"loss": [], "w": [], "b": []}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions y_hat = w*x + b for all rows in X."""
        return _predict(X, self.w, self.b)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """
        Train the model on (X, y) using batch gradient descent.

        Each epoch: forward pass → MSE → gradients → manual weight update.
        """
        # Small random w breaks symmetry; b starts at 0
        self.w = np.random.randn() * 0.01
        self.b = 0.0
        self.history = {"loss": [], "w": [], "b": []}

        for epoch in range(self.n_epochs):
            # --- Forward pass: compute predictions with current w, b ---
            y_pred = self.predict(X)

            # --- Cost: measure how far predictions are from true y ---
            loss = mse(y, y_pred)

            # --- Backward pass: compute dL/dw and dL/db (no autograd) ---
            dw, db = compute_gradients(X, y, self.w, self.b)

            # --- Parameter update: move w, b opposite to the gradient ---
            # w <- w - α * dL/dw   (α = self.lr)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Save state for loss curve, regression snapshots, weight trajectory
            self.history["loss"].append(loss)
            self.history["w"].append(self.w)
            self.history["b"].append(self.b)

            # Stop early if loss barely changes between consecutive epochs
            if epoch > 0 and abs(self.history["loss"][-1] - self.history["loss"][-2]) < self.convergence_tol:
                break

        return self
