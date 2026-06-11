# Visualization — Concepts and Interview Answers

Conceptual reference for `src/visualize.py` in the animated linear regression project.

Implementation lives in **`src/visualize.py`**. This document explains what each plot shows and why it matters for understanding gradient descent.

Companion to [05_linear_regression_and_training.md](./05_linear_regression_and_training.md) — `model.history` powers all plots here.

---

## Why visualize training?

Numbers in a terminal tell you loss went down. **Plots show you how and why.**

| Plot | Question it answers |
|------|---------------------|
| **Loss curve** | Is the model converging? Is learning rate too high? |
| **Regression snapshots** | How does the line move toward the data? |
| **Weight trajectory** | What path does gradient descent take in $(w, b)$ space? |
| **Animation** | Same as snapshots, but over time — the "gradient descent movie" |

This is the core deliverable of the project: **animate gradient descent**.

---

## What data we plot

Training uses **standardized** $X$ (from `data.py`) and **raw** $y$.

| Axis | Space |
|------|-------|
| x-axis (scatter) | Standardized feature values |
| y-axis | Original target values |
| Regression line | $y = wx + b$ in that same space |

The learned $w$ and $b$ apply to standardized $x$. To plot on original $x$ units, use `feature_scaler.inverse_transform()` from `data.py`.

---

## 1. Loss curve

Plots `history["loss"]` vs epoch.

$$
\text{x-axis: epoch} \quad \text{y-axis: } J(w, b)
$$

### What to look for

| Shape | Meaning |
|-------|---------|
| Smooth decrease | Healthy training, reasonable learning rate |
| Flat line at top | Not learning — check gradients, lr, or data |
| Oscillating / spiking | Learning rate too large |
| Plateau | Converged (or stuck in local minimum — rare for linear regression) |

### Function

`plot_loss_curve(history)` in `src/visualize.py`

Same idea as your reference code's Plotly cost chart — we use matplotlib for consistency with other plots.

---

## 2. Regression line snapshots

Scatter of $(x, y)$ plus the fitted line at **selected epochs** (e.g. 0, 10, 50, 100, final).

### What to look for

- **Epoch 0:** random line — poor fit
- **Middle epochs:** line rotating/shifting toward the cloud
- **Final epoch:** line through the center of the data

### Function

`plot_regression_snapshots(X, y, history, epochs=[...])`

Uses `history["w"][epoch]` and `history["b"][epoch]` to draw each line:

$$
\hat{y} = w_t \cdot x + b_t
$$

---

## 3. Weight trajectory

Plots $(w, b)$ at each epoch: `history["w"]` vs `history["b"]`.

### What to look for

- **Start point** (orange) — random initialization
- **Path** — how parameters move each epoch
- **End point** (green) — converged values

This is a 2D slice of the **loss landscape**. Gradient descent walks downhill in $(w, b)$ space to minimize MSE.

### Function

`plot_weight_trajectory(history)`

---

## 4. Animation

Same as regression snapshots, but the line updates **every frame** (one epoch per frame, or every Nth epoch).

Shows:
- Line moving toward data
- Epoch number, loss, $w$, $b$ in the title

### Function

`animate_regression(X, y, history)` — uses matplotlib `FuncAnimation`

Saved to `outputs/regression_animation.gif` when running `main.py`.

---

## Output files

Running `python -m src.main` saves to `outputs/`:

| File | Content |
|------|---------|
| `loss_curve.png` | Loss vs epoch |
| `regression_snapshots.png` | Static multi-epoch lines |
| `weight_trajectory.png` | $(w, b)$ path |
| `regression_animation.gif` | Animated line fitting |

---

## How visualization connects to the pipeline

```
model.fit()  →  history = {loss, w, b}
                      │
                      ├── plot_loss_curve()
                      ├── plot_regression_snapshots()
                      ├── plot_weight_trajectory()
                      └── animate_regression()
```

`visualize.py` never trains the model — it only **reads** `history`.

---

## Interview answers

### "How would you debug a model that isn't learning?"

> Plot the loss curve first. If loss is flat, check learning rate, gradient implementation, and data preprocessing. If loss oscillates, reduce lr. Regression snapshots show whether the line is moving toward the data at all.

### "What does the weight trajectory plot tell you?"

> It shows the path gradient descent takes in parameter space from random init to the converged $(w, b)$. Zig-zags can indicate a high learning rate; a smooth curve toward one point suggests stable convergence.

### "Why plot in standardized x-space?"

> Because that's the space the model trained in. The line $y = wx + b$ is correct relative to standardized inputs. For presentation in original units, inverse-transform $x$ with the fitted scaler.

---

## Self-check questions

1. Where does `history` come from?
   → **`model.fit()`** in `model.py`.

2. What three keys are in `history`?
   → **`loss`**, **`w`**, **`b`**.

3. Does `visualize.py` call `compute_gradients()`?
   → **No** — it only plots stored history.

4. If the loss curve spikes upward mid-training, what hyperparameter do you suspect?
   → **Learning rate too high.**

5. How do you run the full pipeline?
   → **`python -m src.main`** from `animated_linear_regression/`.

---

## Next step

Add **tests** (`tests/test_integration.py`, `tests/test_e2e.py`) and optionally **ipywidgets** sliders for interactive learning rate — see [03_testing.md](./03_testing.md).
