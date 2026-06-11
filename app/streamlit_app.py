"""
Interactive Streamlit demo for animated linear regression.

Imports src/ as a library — does not modify the ML core.
Run: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path (local dev, Docker, Hugging Face)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.charts import (
    gradient_figure,
    loss_figure,
    regression_figure,
    weight_trajectory_figure,
)
from app.styles import CUSTOM_CSS
from app.trainer import init_model, run_one_epoch, train
from src.data import load_and_prepare_data
from src.loss import compute_gradients, mse

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Animated Linear Regression",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Bump when session shape changes — clears stale browser state after crashes/restarts.
APP_STATE_VERSION = 2
if st.session_state.get("_state_version") != APP_STATE_VERSION:
    for key in ("train_job", "live_ui", "model", "epoch_view"):
        st.session_state.pop(key, None)
    st.session_state["_state_version"] = APP_STATE_VERSION

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS = {
    "Custom": {},
    "✅ Good (lr=0.01)": {"lr": 0.01, "n_epochs": 200, "random_init": True},
    "🚀 Fast lr (0.05)": {"lr": 0.05, "n_epochs": 150, "random_init": True},
    "⚠️ Too high lr (0.2)": {"lr": 0.2, "n_epochs": 100, "random_init": True},
    "🐢 Too low lr (0.001)": {"lr": 0.001, "n_epochs": 300, "random_init": True},
    "📍 Zero init": {"lr": 0.01, "n_epochs": 200, "init_w": 0.0, "init_b": 0.0, "random_init": False},
}


@st.cache_data
def get_dataset(standardize_x: bool):
    return load_and_prepare_data(standardize_x=standardize_x)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Animated Linear Regression")
st.markdown(
    '<p class="hero-text">'
    "Linear regression from scratch with <b>manual gradient descent</b> (no autograd). "
    "Tune hyperparameters and watch the loss curve, regression line, and weight trajectory change."
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Hyperparameters")

    preset = st.selectbox("Preset", list(PRESETS.keys()))
    preset_vals = PRESETS[preset]

    lr = st.slider(
        "Learning rate (α)",
        min_value=0.0001,
        max_value=0.3,
        value=float(preset_vals.get("lr", 0.01)),
        step=0.0001,
        format="%.4f",
        help="Step size for weight updates. Too high → divergence.",
    )
    n_epochs = st.slider(
        "Epochs",
        min_value=10,
        max_value=500,
        value=int(preset_vals.get("n_epochs", 200)),
        step=10,
    )
    standardize_x = st.checkbox(
        "Standardize X",
        value=True,
        help="Z-score features (recommended for stable GD).",
    )
    random_init = st.checkbox(
        "Random init for w",
        value=bool(preset_vals.get("random_init", True)),
        help="Small random w (like model.fit). Uncheck to set w, b manually.",
    )
    if not random_init:
        init_w = st.number_input("Initial w", value=float(preset_vals.get("init_w", 0.0)), format="%.4f")
        init_b = st.number_input("Initial b", value=float(preset_vals.get("init_b", 0.0)), format="%.4f")
    else:
        init_w, init_b = 0.0, 0.0

    st.divider()
    live_training = st.checkbox(
        "Live training animation",
        value=True,
        help="Update loss chart and regression line each epoch while training.",
    )
    train_btn = st.button("🚀 Train Model", type="primary", width="stretch")

    if "train_job" in st.session_state:
        if st.button("Cancel training", width="stretch"):
            st.session_state.pop("train_job", None)
            st.rerun()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
dataset = get_dataset(standardize_x)
X_train, y_train = dataset.X_train, dataset.y_train

if train_btn:
    if live_training:
        st.session_state["train_job"] = {
            "model": init_model(
                lr=lr,
                n_epochs=n_epochs,
                init_w=init_w,
                init_b=init_b,
                random_init=random_init,
            ),
            "epoch_done": 0,
            "n_epochs": n_epochs,
            "lr": lr,
            "converged": False,
        }
        st.session_state.pop("model", None)
    else:
        model = train(
            X_train,
            y_train,
            lr=lr,
            n_epochs=n_epochs,
            init_w=init_w,
            init_b=init_b,
            random_init=random_init,
        )
        st.session_state["model"] = model
        st.session_state["lr"] = lr
        st.session_state["epoch_view"] = len(model.history["loss"]) - 1
    st.rerun()


LIVE_REFRESH_SEC = 0.25
LIVE_FRAME_TARGET = 40


def _live_figure(fig, title: str):
    """Stable Plotly layout for in-place live updates (avoids chart flicker)."""
    fig.update_layout(
        title={"text": title, "font": {"color": "#41BEE9", "size": 16}},
        uirevision="live",
        datarevision="live",
    )
    return fig


@st.fragment(run_every=LIVE_REFRESH_SEC)
def _live_training_panel() -> None:
    """Advance training in small batches; redraw charts each tick."""
    job = st.session_state.get("train_job")
    if not job:
        return

    model = job["model"]
    n_epochs_job = job["n_epochs"]
    epoch_done = job["epoch_done"]
    epochs_per_tick = max(1, n_epochs_job // LIVE_FRAME_TARGET)

    if epoch_done < n_epochs_job and not job.get("converged", False):
        for _ in range(epochs_per_tick):
            if epoch_done >= n_epochs_job or job.get("converged", False):
                break
            _, converged = run_one_epoch(model, X_train, y_train, job["lr"])
            epoch_done += 1
            job["epoch_done"] = epoch_done
            job["converged"] = converged
        st.session_state["train_job"] = job

    st.subheader("Live training")
    progress_pct = min(epoch_done / n_epochs_job, 1.0)
    loss_val = model.history["loss"][-1] if model.history["loss"] else None
    progress_text = (
        f"Epoch {epoch_done} / {n_epochs_job}  ·  loss = {loss_val:.4f}"
        if loss_val is not None
        else f"Epoch {epoch_done} / {n_epochs_job}"
    )
    st.progress(progress_pct, text=progress_text)

    if model.history["loss"]:
        epoch_idx = len(model.history["loss"]) - 1
        st.markdown(
            f'<div class="live-stats">'
            f"<span>Epoch <b>{epoch_idx}</b></span>"
            f"<span>Loss <b>{loss_val:.4f}</b></span>"
            f"<span>w <b>{model.w:.4f}</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        col_loss, col_reg = st.columns(2)
        with col_loss:
            st.plotly_chart(
                _live_figure(
                    loss_figure(
                        model.history,
                        up_to_epoch=epoch_idx,
                        highlight_epoch=False,
                    ),
                    "Cost vs Iteration (live)",
                ),
                width="stretch",
                key="live_loss_chart",
            )
        with col_reg:
            st.plotly_chart(
                _live_figure(
                    regression_figure(X_train, y_train, model.history, epoch_idx),
                    "Regression Line (live)",
                ),
                width="stretch",
                key="live_reg_chart",
            )
    else:
        st.info("Running first epoch…")

    if epoch_done >= n_epochs_job or job.get("converged", False):
        st.session_state["model"] = model
        st.session_state["lr"] = job["lr"]
        st.session_state["epoch_view"] = len(model.history["loss"]) - 1
        del st.session_state["train_job"]
        st.rerun()


if "train_job" in st.session_state:
    _live_training_panel()
    st.stop()

if "model" not in st.session_state:
    st.info("👈 Adjust hyperparameters and click **Train Model** to start.")
    st.stop()

model = st.session_state["model"]
history = model.history
n_trained = len(history["loss"])
lr = st.session_state.get("lr", lr)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Initial Loss", f"{history['loss'][0]:.2f}")
col2.metric("Final Loss", f"{history['loss'][-1]:.2f}")
col3.metric("Final w", f"{model.w:.4f}")
col4.metric("Final b", f"{model.b:.4f}")
col5.metric("Epochs Run", n_trained)

reduction = history["loss"][0] / max(history["loss"][-1], 1e-9)
if history["loss"][-1] >= history["loss"][0]:
    st.warning("Loss did not decrease — try a lower learning rate.")
elif reduction > 10:
    st.success(f"Loss decreased {reduction:.1f}× — gradient descent is working.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_reg, tab_loss, tab_weights, tab_grad = st.tabs(
    ["📉 Regression Line", "📊 Loss Curve", "🎯 Weight Trajectory", "∇ Gradients"]
)

with tab_reg:
    epoch_view = st.slider(
        "Scrub epoch",
        min_value=0,
        max_value=n_trained - 1,
        value=st.session_state.get("epoch_view", n_trained - 1),
        key="epoch_slider",
    )
    st.session_state["epoch_view"] = epoch_view
    st.plotly_chart(
        regression_figure(X_train, y_train, history, epoch_view),
        width="stretch",
        key="tab_regression_chart",
    )

with tab_loss:
    loss_epoch = st.slider(
        "Show loss up to epoch",
        min_value=0,
        max_value=n_trained - 1,
        value=n_trained - 1,
        key="loss_epoch_slider",
        help="Scrub to see how the loss curve builds over training.",
    )
    st.plotly_chart(
        loss_figure(history, up_to_epoch=loss_epoch),
        width="stretch",
        key="tab_loss_chart",
    )
    st.caption(
        f"At epoch {loss_epoch}: loss = {history['loss'][loss_epoch]:.4f}  ·  "
        f"w = {history['w'][loss_epoch]:.4f}  ·  b = {history['b'][loss_epoch]:.4f}"
    )

with tab_weights:
    st.plotly_chart(
        weight_trajectory_figure(history),
        width="stretch",
        key="tab_weight_chart",
    )

with tab_grad:
    grad_epoch = st.slider(
        "Epoch for gradient view",
        min_value=0,
        max_value=n_trained - 1,
        value=n_trained - 1,
        key="grad_epoch",
    )
    w_e, b_e = history["w"][grad_epoch], history["b"][grad_epoch]
    dw, db = compute_gradients(X_train, y_train, w_e, b_e)
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("dw", f"{dw:.6f}")
    gc2.metric("db", f"{db:.6f}")
    gc3.metric("Loss at epoch", f"{history['loss'][grad_epoch]:.4f}")
    st.plotly_chart(
        gradient_figure(X_train, y_train, w_e, b_e, lr),
        width="stretch",
        key="tab_gradient_chart",
    )
    st.caption(
        "Update rule: w ← w − α·dw,  b ← b − α·db  "
        f"(α = {lr}). Bars on the right show the actual step size."
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "ML core: `src/` (NumPy, manual MSE + gradients) · UI: `app/` · "
    "Try preset **Too high lr** to see divergence."
)
