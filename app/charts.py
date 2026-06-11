"""Interactive Plotly charts for the Streamlit app (presentation layer only)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.loss import _predict, compute_gradients

ACCENT = "#41BEE9"
BG = "#0e1117"
GRID = "#2a2f3a"
MUTED = "#888888"


def _base_layout(
    title: str,
    x_title: str,
    y_title: str,
    *,
    live_update: bool = False,
) -> dict:
    layout = dict(
        title=dict(text=title, font=dict(color=ACCENT, size=16)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#e0e0e0"),
        xaxis=dict(title=x_title, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title=y_title, gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=40, r=20, t=50, b=40),
        height=420,
    )
    if live_update:
        # Same uirevision keeps Plotly from resetting the canvas each tick.
        layout["uirevision"] = "live"
        layout["datarevision"] = "live"
    return layout


def loss_figure(
    history: dict[str, list],
    up_to_epoch: int | None = None,
    highlight_epoch: bool = True,
    live_update: bool = False,
) -> go.Figure:
    """
    MSE loss vs epoch.

    If up_to_epoch is set, only plot loss up to that epoch (inclusive).
    Highlights the current epoch as a marker when highlight_epoch is True.
    """
    losses = history["loss"]
    end = len(losses) if up_to_epoch is None else min(up_to_epoch + 1, len(losses))
    x = list(range(end))
    y = losses[:end]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=ACCENT, width=2),
            name="Loss",
        )
    )
    if highlight_epoch and not live_update and end > 0:
        fig.add_trace(
            go.Scatter(
                x=[end - 1],
                y=[y[-1]],
                mode="markers",
                marker=dict(color="#ff6b6b", size=10),
                name="Current epoch",
            )
        )
    if live_update:
        title = "Cost vs Iteration (live)"
    elif up_to_epoch is not None:
        title = f"Cost vs Iteration (through epoch {end - 1})"
    else:
        title = "Cost vs Iteration"
    fig.update_layout(**_base_layout(title, "Epoch", "Loss (MSE)", live_update=live_update))
    return fig


def regression_figure(
    X: np.ndarray,
    y: np.ndarray,
    history: dict[str, list],
    epoch: int,
    live_update: bool = False,
) -> go.Figure:
    """Scatter + regression line at a specific epoch."""
    epoch = min(epoch, len(history["loss"]) - 1)
    w, b = history["w"][epoch], history["b"][epoch]
    loss = history["loss"][epoch]

    x_flat = X.flatten()
    x_line = np.linspace(x_flat.min(), x_flat.max(), 100)
    y_line = _predict(x_line.reshape(-1, 1), w, b)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_flat,
            y=y,
            mode="markers",
            marker=dict(color=MUTED, size=6, opacity=0.5),
            name="Data",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=ACCENT, width=3),
            name=f"Epoch {epoch}",
        )
    )
    if live_update:
        title = "Regression Line (live)"
    else:
        title = f"Regression Line — Epoch {epoch}  (loss={loss:.2f}, w={w:.3f}, b={b:.3f})"
    fig.update_layout(
        **_base_layout(title, "x (standardized)", "y", live_update=live_update)
    )
    return fig


def weight_trajectory_figure(history: dict[str, list]) -> go.Figure:
    """Path of (w, b) over training."""
    ws, bs = history["w"], history["b"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ws,
            y=bs,
            mode="lines+markers",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=4),
            name="GD path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ws[0]],
            y=[bs[0]],
            mode="markers",
            marker=dict(color="orange", size=12, symbol="circle"),
            name="Start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ws[-1]],
            y=[bs[-1]],
            mode="markers",
            marker=dict(color="lime", size=12, symbol="circle"),
            name="End",
        )
    )
    fig.update_layout(**_base_layout("Weight Trajectory in (w, b) Space", "w", "b"))
    return fig


def gradient_figure(
    X: np.ndarray,
    y: np.ndarray,
    w: float,
    b: float,
    lr: float,
) -> go.Figure:
    """Bar chart of current gradients and implied update direction."""
    dw, db = compute_gradients(X, y, w, b)
    delta_w = -lr * dw
    delta_b = -lr * db

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Gradients (dL/d param)", "Update step (-lr * gradient)"),
    )
    fig.add_trace(
        go.Bar(x=["dw", "db"], y=[dw, db], marker_color=ACCENT, name="Gradient"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=["Δw", "Δb"], y=[delta_w, delta_b], marker_color="#ff6b6b", name="Update"),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=dict(text="Gradients at Selected Epoch", font=dict(color=ACCENT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(color="#e0e0e0"),
        showlegend=False,
        height=380,
    )
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    return fig
