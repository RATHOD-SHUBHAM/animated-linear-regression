from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# Resolve paths relative to this file so the module works from any working directory.
# PROJECT_ROOT -> animated_linear_regression/
# DATA_DIR     -> animated_linear_regression/data/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Dataset:
    """
    Container for prepared train/test splits.

    All X arrays are shaped (n_samples, 1) for compatibility with the training loop.
    y arrays are shaped (n_samples,) — we do not standardize the target for this project.
    """
    X_train: np.ndarray          # shape (n_train, 1)
    y_train: np.ndarray          # shape (n_train,)
    X_test: np.ndarray           # shape (n_test, 1)
    y_test: np.ndarray           # shape (n_test,)
    feature_scaler: "StandardScaler"


class StandardScaler:
    """
    Z-score standardization (same idea as sklearn.preprocessing.StandardScaler).

    Formula per feature:
        x_scaled = (x - mean) / std

    Stored after fit():
        mean_  -> μ, the mean computed from training data
        scale_ -> σ, the standard deviation computed from training data
                 (named "scale_" because it is the divisor used in transform)
    """

    def __init__(self, eps: float = 1e-8):
        # Tiny threshold: if std is smaller than eps, we treat it as zero.
        # Prevents division-by-zero when a feature is (near) constant.
        self.eps = eps

        # Populated by fit(); remain None until then.
        self.mean_: np.ndarray | None = None   # μ, shape (n_features,)
        self.scale_: np.ndarray | None = None  # σ, shape (n_features,)

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """
        Learn μ and σ from the provided data.

        IMPORTANT: Call this on X_train only to avoid data leakage.
        """
        X = _as_2d(X)  # ensure shape (n_samples, n_features)

        # μ — mean of each feature column
        self.mean_ = X.mean(axis=0)

        # σ — standard deviation of each feature column
        # ddof=0 gives population std, matching sklearn's StandardScaler
        std = X.std(axis=0)

        # If σ ≈ 0 the feature is constant; use 1.0 so (x - mean) / 1 = 0
        self.scale_ = np.where(std < self.eps, 1.0, std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply z-score scaling using statistics learned during fit().

        x_scaled = (x - μ) / σ

        Never recompute mean/std from X here — always reuse values from fit().
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before transform().")

        X = _as_2d(X)
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Undo z-score scaling to recover original feature units.

        x_original = x_scaled * σ + μ

        Useful when plotting the regression line on the original x-axis.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        X = _as_2d(X)
        return X * self.scale_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience method: fit on X, then transform the same X. Used for X_train."""
        return self.fit(X).transform(X)


def _as_2d(X: np.ndarray) -> np.ndarray:
    """
    Ensure X has shape (n_samples, n_features).

    Gradient descent and matrix operations expect 2D feature matrices,
    even when there is only one feature.
    """
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        return X.reshape(-1, 1)

    return X


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and drop rows with missing x or y values."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["x", "y"])
    return df


def load_and_prepare_data(
    data_dir: Path = DATA_DIR,
    standardize_x: bool = True,
) -> Dataset:
    """
    Load train/test CSVs and prepare NumPy arrays for gradient descent.

    Parameters
    ----------
    data_dir : Path
        Directory containing train.csv and test.csv.
    standardize_x : bool
        If True, fit the scaler on X_train and transform both train and test.
        y is left in original units so loss and plots stay interpretable.

    Returns
    -------
    Dataset
        Prepared arrays plus the fitted feature scaler.
    """
    train_df = _load_csv(data_dir / "train.csv")
    test_df = _load_csv(data_dir / "test.csv")

    # Extract features and targets as float64 NumPy arrays
    X_train = train_df["x"].to_numpy(dtype=np.float64)
    y_train = train_df["y"].to_numpy(dtype=np.float64)
    X_test = test_df["x"].to_numpy(dtype=np.float64)
    y_test = test_df["y"].to_numpy(dtype=np.float64)

    scaler = StandardScaler()

    if standardize_x:
        # Fit on train only; apply the same μ and σ to both splits
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        # Skip scaling but still enforce (n_samples, 1) shape
        X_train = _as_2d(X_train)
        X_test = _as_2d(X_test)

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_scaler=scaler,
    )
