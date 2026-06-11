# Linear Regression and Training — Concepts and Interview Answers

Conceptual reference for `src/model.py` in the animated linear regression project.

Implementation lives in **`src/model.py`**, **`src/loss.py`**, and **`src/data.py`**. This document explains the full picture — what linear regression is, what training means, and how gradient descent learns the parameters.

Companions:
- [04_loss_and_gradients.md](./04_loss_and_gradients.md) — cost function and gradient math in depth
- [06_autograd_manual_vs_automatic.md](./06_autograd_manual_vs_automatic.md) — what "no autograd" means
- [01_standardization.md](./01_standardization.md) — why we scale features before training

---

## What is linear regression?

**Linear regression** is a supervised learning algorithm for predicting a **continuous** target ($y$) from input features ($x$).

It assumes the relationship between $x$ and $y$ is approximately **linear**:

$$
\hat{y} = wx + b
$$

Where:
- $\hat{y}$ — predicted value
- $x$ — input feature
- $w$ — **weight** (slope — how much $y$ changes when $x$ increases by 1)
- $b$ — **bias** (intercept — predicted $y$ when $x = 0$)

**Example:** predicting house price from square footage. Each extra sq ft adds roughly $w$ dollars; $b$ is the base price.

### What we are learning

The model does not "memorize" data. It learns two numbers $(w, b)$ that define a **line** (in 2D) best aligned with the data cloud.

| Term | Meaning |
|------|---------|
| **Parameters** | $w$ and $b$ — learned from data |
| **Hyperparameters** | Learning rate, number of epochs — set by us before training |
| **Training** | The process of finding good $w$ and $b$ |
| **Inference / prediction** | Using learned $w$, $b$ on new $x$ values |

In this project: one feature ($x$), so the model is a single line. With many features, $w$ becomes a vector and the model is a hyperplane.

---

## Supervised learning in one sentence

> We have labeled examples $(x, y)$. The model learns a function that maps $x \rightarrow \hat{y}$ so that $\hat{y}$ is close to the true $y$.

Linear regression is the simplest form of this for continuous targets.

---

## The full pipeline in this project

```
data.py          loss.py           model.py          visualize.py
────────         ───────           ────────          ────────────
Load CSV    →    mse()         ←   fit() loop   →   plots from
Standardize X    compute_gradients()   predict()       model.history
Return X, y
```

| File | Role |
|------|------|
| `data.py` | Load and prepare $(X, y)$; standardize features |
| `loss.py` | Compute cost and gradients (math only) |
| `model.py` | Run gradient descent; update $w$, $b$; record history |
| `visualize.py` | Plot loss curve, regression line, weight trajectory |

---

## What is training?

**Training** is the iterative process of improving model parameters so predictions get closer to true labels.

We start with random (or zero) $w$ and $b$. Each **epoch** (one full pass over the training data):

1. Make predictions with current $w$, $b$
2. Measure error (cost / loss)
3. Compute gradients — which direction increases error
4. Update $w$, $b$ in the **opposite** direction to reduce error

After many epochs, $w$ and $b$ should describe a line that fits the data well.

In `src/model.py`, this is the `fit(X, y)` method on `LinearRegressionGD`.

---

## Gradient descent — how the model learns

We cannot try every possible $(w, b)$. Instead we use **gradient descent**: follow the slope of the loss function downhill.

### The update rule

After computing gradients $\frac{\partial J}{\partial w}$ and $\frac{\partial J}{\partial b}$:

$$
w \leftarrow w - \alpha \frac{\partial J}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial J}{\partial b}
$$

Where $\alpha$ is the **learning rate** (`lr` in code).

**Intuition:** the gradient points **uphill** (direction of steepest increase in loss). We step **downhill** by subtracting the gradient.

### Batch gradient descent

Our project uses **batch** gradient descent: each epoch uses **all** training samples to compute one gradient and one update.

| Variant | What it uses per update |
|---------|-------------------------|
| **Batch GD** (our project) | Entire training set |
| **Stochastic GD (SGD)** | One sample |
| **Mini-batch GD** | Small batch of samples |

Batch GD is stable and simple — ideal for learning and animation on small datasets.

---

## Key hyperparameters

### Learning rate ($\alpha$ / `lr`)

The **learning rate** controls how big each parameter update step is.

| Value | Effect |
|-------|--------|
| **Too large** | Loss oscillates or diverges — overshoots the minimum |
| **Too small** | Convergence is very slow — many epochs needed |
| **Just right** | Loss decreases smoothly to a low value |

Default in our project: `lr=0.01`. With standardized features, this usually works well.

**Analogy:** walking downhill in the dark. Small steps = safe but slow. Large steps = fast but you might overshoot the valley.

### Number of epochs (`n_epochs`)

An **epoch** is one complete pass through the training data.

- Too few epochs → model underfits (line not fitted well)
- Too many epochs → usually fine for linear regression (loss plateaus); early stopping helps

Default: `n_epochs=200`.

### Convergence tolerance (`convergence_tol`)

If the loss change between consecutive epochs is smaller than this threshold, training stops early.

$$
|J_{\text{new}} - J_{\text{old}}| < \text{convergence\_tol}
$$

Default: `1e-6`. Saves time when the model has already converged.

---

## One training epoch — step by step

What happens inside `fit()` each epoch:

| Step | Name | What happens | Code |
|------|------|--------------|------|
| 1 | **Forward pass** | Compute $\hat{y} = wx + b$ for all samples | `predict(X)` |
| 2 | **Cost** | Measure average squared error | `mse(y, y_pred)` |
| 3 | **Backward pass** | Compute $\frac{\partial J}{\partial w}$, $\frac{\partial J}{\partial b}$ | `compute_gradients(...)` |
| 4 | **Update** | $w \leftarrow w - \alpha \cdot dw$; same for $b$ | `self.w -= lr * dw` |
| 5 | **Record** | Save loss, $w$, $b$ to history | `self.history[...]` |

See [04_loss_and_gradients.md](./04_loss_and_gradients.md) for the cost and gradient formulas.

---

## Parameters vs hyperparameters

| Type | Examples | Who sets them | Change during training? |
|------|----------|---------------|-------------------------|
| **Parameters** | $w$, $b$ | Learned from data | Yes |
| **Hyperparameters** | `lr`, `n_epochs`, `convergence_tol` | You, before training | No |

Interview tip: hyperparameters are not updated by gradient descent in basic linear regression.

---

## Weight initialization

In `fit()`:

- $w$ starts as small random: `np.random.randn() * 0.01`
- $b$ starts at `0.0`

**Why small random $w$?** Breaks symmetry — if all parameters start at exactly zero in multi-feature models, all features update identically. For one feature, zero init also works, but small random is a good habit.

**Why $b = 0$?** Bias can move in either direction once training starts.

---

## History — why we record every epoch

```python
self.history = {"loss": [], "w": [], "b": []}
```

Each epoch appends the current loss, $w$, and $b$. This powers:

| Plot | Uses |
|------|------|
| **Loss curve** | `history["loss"]` — did training converge? |
| **Regression line animation** | `history["w"]`, `history["b"]` — how the line moves toward data |
| **Weight trajectory** | `history["w"]` vs `history["b"]` — path in parameter space |

Without history, you only see the final line — not **how** gradient descent got there.

---

## How training connects to standardized data

From [01_standardization.md](./01_standardization.md): we standardize $X$ before training.

| Effect | Why it matters |
|--------|----------------|
| Features near mean 0, std 1 | Gradients are balanced; learning rate behaves predictably |
| $y$ stays in original units | Loss and plots stay interpretable |

The learned $w$ applies to **standardized** $x$. To plot on the original axis, use `feature_scaler.inverse_transform()` from `data.py`.

---

## Underfitting vs overfitting (brief)

| | Underfitting | Good fit | Overfitting |
|---|-------------|----------|-------------|
| **Loss** | High on train | Low on train | Very low on train, higher on test |
| **Line** | Too flat / wrong slope | Follows trend | Chases noise (less common in plain linear regression) |

Plain linear regression with one feature rarely overfits badly on reasonable data. More relevant when you add many features or polynomial terms.

---

## Closed-form vs gradient descent

Linear regression has a **closed-form solution** (normal equation):

$$
w = (X^T X)^{-1} X^T y
$$

| Approach | When to use |
|----------|-------------|
| **Closed-form** | Small data, exact solution, no iteration |
| **Gradient descent** | Large data, deep learning, animation, learning |

We use gradient descent because this project is about **understanding optimization** — the same loop used in neural networks.

---

## Interview answers

### "What is linear regression?"

> A supervised algorithm that models a continuous target as a linear function of features: $\hat{y} = wx + b$. We learn $w$ and $b$ to minimize prediction error, typically MSE.

### "What happens during training?"

> We iteratively compute predictions, measure loss, compute gradients of the loss w.r.t. $w$ and $b$, and update parameters in the direction that reduces loss — gradient descent.

### "What is the learning rate?"

> The step size $\alpha$ in the update rule $w \leftarrow w - \alpha \cdot \nabla J$. It controls how aggressively we move toward lower loss. Too high diverges; too low is slow.

### "What is an epoch?"

> One full pass through the entire training dataset. In batch GD, one epoch = one gradient computation using all samples.

### "What's the difference between parameters and hyperparameters?"

> Parameters ($w$, $b$) are learned from data during training. Hyperparameters (learning rate, epochs) are set beforehand and control how training runs.

### "Why record history during training?"

> To visualize convergence — loss curve shows the model learning; weight trajectory shows the path in $(w, b)$ space. Essential for debugging and for the animation in this project.

### "Batch GD vs SGD?"

> Batch GD uses all samples per update — stable, deterministic. SGD uses one sample — noisier but faster per step on huge datasets. Mini-batch is the practical middle ground in deep learning.

---

## Code map

```
src/model.py — LinearRegressionGD
├── __init__(lr, n_epochs, convergence_tol)
├── predict(X)              → forward pass at inference
└── fit(X, y)               → training loop
    ├── initialize w, b
    ├── loop n_epochs:
    │   ├── predict → mse → compute_gradients
    │   ├── update w, b
    │   └── append to history
    └── early stop if converged
```

---

## Self-check questions

1. What does $w$ represent geometrically?
   → **Slope** of the regression line.

2. What does the learning rate control?
   → **Step size** of each parameter update.

3. Does `loss.py` update weights?
   → **No** — only `model.py` does.

4. What is stored in `model.history`?
   → **loss**, **w**, and **b** at each epoch.

5. Why standardize $X$ before calling `fit()`?
   → **Stable gradient descent** with a reasonable learning rate.

6. Will final loss always be exactly zero?
   → **No** — real data has noise; we minimize but rarely hit zero.

7. What comes after `model.py` in this project?
   → **`visualize.py`** — plot loss curve, regression line, weight trajectory.

---

## Next step

Build **`src/visualize.py`** using `model.history` and data from `load_and_prepare_data()`. Then wire everything in **`src/main.py`**.
