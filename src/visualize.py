from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.loss import _predict

# Save plots here by default (animated_linear_regression/outputs/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def plot_loss_curve(
    history: dict[str, list],
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot training loss (MSE) vs epoch.

    A decreasing curve means gradient descent is working.
    """
    losses = history["loss"]
    epochs = range(len(losses))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, losses, color="#41BEE9", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Cost vs Iteration")
    ax.grid(True, alpha=0.3)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def plot_regression_snapshots(
    X: np.ndarray,
    y: np.ndarray,
    history: dict[str, list],
    epochs: list[int] | None = None,
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Scatter plot of data with regression lines at selected epochs.

    Shows how the line rotates/shifts toward the data cloud over training.
    X and history w/b must be in the same space (standardized if data was scaled).
    """
    n_epochs = len(history["loss"])
    if epochs is None:
        # Default: first, early, mid, late, final
        candidates = [0, 10, 50, 100, n_epochs - 1]
        epochs = sorted({e for e in candidates if 0 <= e < n_epochs})

    x_flat = X.flatten()
    x_line = np.linspace(x_flat.min(), x_flat.max(), 100)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(x_flat, y, alpha=0.4, s=15, color="#888888", label="Data")

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(epochs)))
    for i, epoch in enumerate(epochs):
        w, b = history["w"][epoch], history["b"][epoch]
        y_line = _predict(x_line.reshape(-1, 1), w, b)
        ax.plot(
            x_line,
            y_line,
            color=colors[i],
            linewidth=2,
            label=f"Epoch {epoch}",
        )

    ax.set_xlabel("x (feature space used for training)")
    ax.set_ylabel("y")
    ax.set_title("Regression Line at Selected Epochs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def plot_weight_trajectory(
    history: dict[str, list],
    ax: plt.Axes | None = None,
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot the path of (w, b) in parameter space over training.

    Each point is one epoch; the path shows gradient descent moving toward a minimum.
    """
    ws = history["w"]
    bs = history["b"]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ws, bs, color="#41BEE9", linewidth=1.5, alpha=0.8)
    ax.scatter(ws[0], bs[0], color="orange", s=80, zorder=5, label="Start")
    ax.scatter(ws[-1], bs[-1], color="lime", s=80, zorder=5, label="End")
    ax.set_xlabel("w (weight)")
    ax.set_ylabel("b (bias)")
    ax.set_title("Gradient Descent Path in (w, b) Space")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def animate_regression(
    X: np.ndarray,
    y: np.ndarray,
    history: dict[str, list],
    interval: int = 50,
    frame_step: int = 1,
    save_path: Path | str | None = None,
    show: bool = True,
) -> FuncAnimation:
    """
    Animate the regression line fitting the data over epochs.

    Parameters
    ----------
    interval : ms between frames in the animation window
    frame_step : plot every Nth epoch (use 2–5 if training ran many epochs)
    save_path : if set, save as GIF (requires pillow) or MP4
    """
    x_flat = X.flatten()
    x_line = np.linspace(x_flat.min(), x_flat.max(), 100)

    epoch_indices = list(range(0, len(history["loss"]), frame_step))
    if epoch_indices[-1] != len(history["loss"]) - 1:
        epoch_indices.append(len(history["loss"]) - 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_flat, y, alpha=0.4, s=15, color="#888888")
    ax.set_xlim(x_flat.min() - 0.1, x_flat.max() + 0.1)
    ax.set_ylim(y.min() - 5, y.max() + 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)

    (line,) = ax.plot([], [], color="#41BEE9", linewidth=2)
    title = ax.set_title("")

    def init():
        line.set_data([], [])
        title.set_text("")
        return line, title

    def update(frame_idx: int):
        epoch = epoch_indices[frame_idx]
        w, b = history["w"][epoch], history["b"][epoch]
        y_line = _predict(x_line.reshape(-1, 1), w, b)
        line.set_data(x_line, y_line)
        loss = history["loss"][epoch]
        title.set_text(f"Epoch {epoch}  |  Loss = {loss:.4f}  |  w = {w:.4f}  b = {b:.4f}")
        return line, title

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(epoch_indices),
        interval=interval,
        blit=True,
    )

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".gif":
            anim.save(path, writer="pillow", dpi=100)
        elif suffix == ".mp4":
            anim.save(path, writer="ffmpeg", dpi=100)
        else:
            anim.save(path.with_suffix(".gif"), writer="pillow", dpi=100)

    if show:
        plt.show()

    return anim


def plot_all(
    X: np.ndarray,
    y: np.ndarray,
    history: dict[str, list],
    output_dir: Path | str | None = None,
    show: bool = True,
) -> None:
    """
    Generate all static plots and save them to outputs/.

    Convenience wrapper called from main.py.
    """
    out = Path(output_dir) if output_dir else _ensure_output_dir()

    plot_loss_curve(history, save_path=out / "loss_curve.png", show=show)
    plt.close()

    plot_regression_snapshots(
        X, y, history, save_path=out / "regression_snapshots.png", show=show
    )
    plt.close()

    plot_weight_trajectory(history, save_path=out / "weight_trajectory.png", show=show)
    plt.close()
