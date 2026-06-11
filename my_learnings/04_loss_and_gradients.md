# Loss and Gradients — Concepts and Interview Answers

Conceptual reference for `src/loss.py` in the animated linear regression project.

Implementation lives in **`src/loss.py`** — this document explains the math and intuition only.

Companion to [01_standardization.md](./01_standardization.md). Training (weight updates) is covered in [05_linear_regression_and_training.md](./05_linear_regression_and_training.md). See [06_autograd_manual_vs_automatic.md](./06_autograd_manual_vs_automatic.md) for what "no autograd" means.

---

## What `loss.py` implements

Linear regression needs three mathematical pieces before training can begin:

| Concept | What it does | Function in `src/loss.py` |
|---------|--------------|---------------------------|
| **Model / forward pass** | Compute predictions from current $w$, $b$ | `_predict()` |
| **Cost function** | Measure how wrong predictions are | `mse()` |
| **Backward pass** | Compute gradients for updating $w$, $b$ | `compute_gradients()` |

`loss.py` does **not** update weights — that happens in `model.py`.

---

## The linear regression model

Linear regression predicts a continuous target from input features using a linear function.

For one feature:

$$
f_{w,b}(x) = wx + b
$$

Where:
- $f_{w,b}(x)$ — predicted output ($\hat{y}$)
- $w$ — weight (slope)
- $b$ — bias (intercept)
- $x$ — input feature

**Goal:** find $(w, b)$ that best fit the data. `loss.py` provides the tools to measure "best" (cost) and compute how to improve (gradients). The actual search happens in `model.py`.

**Array shapes in this project:**

| Symbol | Shape |
|--------|-------|
| $X$ | `(n_samples, 1)` |
| $w$, $b$ | scalar |
| $\hat{y}$ | `(n_samples,)` |

---

## Forward pass

The **forward pass** applies the model to input data using the current parameters:

$$
\hat{y} = w \cdot x + b
$$

For a batch of $n$ samples, compute one prediction per row of $X$.

**Purpose:** produces $\hat{y}$, which is needed by both the cost function and the gradient computation.

In `src/loss.py`, this is handled by `_predict(X, w, b)`.

---

## Cost function (MSE)

The **cost function** (also called **loss function**) measures how far predictions are from the true values. We use **Mean Squared Error (MSE)**:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2
$$

Where:
- $J(w, b)$ — cost (scalar)
- $m$ — number of training examples ($n$ in our code)
- $x^{(i)}$ — input for the $i$-th example
- $y^{(i)}$ — true output for the $i$-th example

Equivalently, using $\hat{y}_i = f_{w,b}(x^{(i)})$:

$$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

In `src/loss.py`, this is `mse(y_true, y_pred)`.

### Why squared errors?

- Penalizes large mistakes more than small ones
- Smooth and differentiable everywhere (required for gradient descent)
- Under Gaussian noise assumptions, minimizing MSE equals maximum likelihood

### Why the $\frac{1}{2}$ factor?

When you differentiate a squared term, a factor of **2** appears from the power rule. The $\frac{1}{2}$ in the cost cancels it, giving cleaner gradient formulas.

| Version | Cost | Gradient w.r.t. $w$ |
|---------|------|---------------------|
| With $\frac{1}{2m}$ (our project) | $\frac{1}{2m}\sum(\hat{y}-y)^2$ | $\frac{1}{m}\sum(\hat{y}-y)\,x$ |
| Without $\frac{1}{2}$ (sklearn-style) | $\frac{1}{m}\sum(\hat{y}-y)^2$ | $\frac{2}{m}\sum(\hat{y}-y)\,x$ |

Both find the same optimal $(w, b)$ — only the loss scale and gradient magnitude differ. Our project uses **$\frac{1}{2m}$** in the cost and **$\frac{1}{m}$** in the gradients.

### Common bug

```
WRONG:  (sum of errors)²
CORRECT: sum of (error²)
```

Squaring the sum of errors is a different (incorrect) formula entirely.

---

## Backward pass (gradient computation)

The **backward pass** computes how the cost changes when we nudge $w$ or $b$. These **gradients** tell us which direction to adjust each parameter.

Define the error (residual) for one sample:

$$
e = \hat{y} - y
$$

### Single sample (chain rule)

For $L = \frac{1}{2}(\hat{y} - y)^2$ with $\hat{y} = wx + b$:

$$
\frac{\partial J}{\partial w} = (\hat{y} - y) \cdot x = e \cdot x
$$

$$
\frac{\partial J}{\partial b} = (\hat{y} - y) = e
$$

**Intuition:** if we over-predict ($\hat{y} > y$), the error is positive, and the gradient pushes $w$ down (when $x > 0$) after the update step in `model.py`.

### Full batch ($m$ samples)

Average over all training examples:

$$
\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right) x^{(i)}
$$

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left(f_{w,b}(x^{(i)}) - y^{(i)}\right)
$$

Note: gradients use $\frac{1}{m}$, not $\frac{1}{2m}$ — the 2 cancels during differentiation.

In `src/loss.py`, this is `compute_gradients(X, y, w, b)` which returns `(dw, db)`.

---

## How the three pieces connect

```
Input: X, y, current w and b
         │
         ▼
   Forward pass (_predict)
   ŷ = w*x + b
         │
         ├──────────────────┐
         ▼                  ▼
   Cost (mse)        Backward (compute_gradients)
   J(w, b)            dw, db
```

Both branches need $\hat{y}$. The cost tells us **how bad** we are; the gradients tell us **how to improve**. Weight updates happen next in `model.py`.

---

## Gradient intuition

| Situation | Error ($\hat{y} - y$) | Effect on $w$ (when $x > 0$) |
|-----------|------------------------|------------------------------|
| Over-predicted | positive | Gradient positive → update reduces $w$ |
| Under-predicted | negative | Gradient negative → update increases $w$ |

**Sanity check:** with $w=0$, $b=0$ and all $y > 0$:
- Predictions are 0 → error is negative → $db < 0$
- In `model.py`, $b \leftarrow b - \alpha \cdot db$ will **increase** $b$ toward the data

---

## How to test (no code duplication — see `tests/`)

| Test | What it verifies |
|------|------------------|
| `mse(y, y) == 0` | Perfect predictions → zero cost |
| `mse([0], [1]) == 0.5` | Single sample with $\frac{1}{2m}$ formula |
| Gradient sign at $w=0, b=0$ | Updates will move in the right direction |
| Numerical gradient check | Analytical gradients match finite-difference approximation |

The numerical gradient check is the gold standard for ML interviews — it proves your derivation and code are both correct.

---

## Interview answers

### "Walk me through the forward pass and cost function."

> The model is $f_{w,b}(x) = wx + b$. The forward pass computes predictions $\hat{y}$ for all inputs. The cost is MSE: $J(w,b) = \frac{1}{2m}\sum(\hat{y}_i - y_i)^2$, which averages squared prediction errors.

### "Derive the gradient of MSE w.r.t. w."

> For one sample, $L = \frac{1}{2}(\hat{y}-y)^2$. By the chain rule, $\partial L / \partial w = (\hat{y}-y) \cdot x$. Averaging over $m$ samples: $\partial J / \partial w = \frac{1}{m}\sum(\hat{y}_i - y_i)\,x_i$.

### "Why MSE instead of MAE?"

> MSE is smooth everywhere — gradient descent works cleanly. MAE uses absolute value, which has an undefined derivative at zero. MAE is more robust to outliers, but MSE is standard for linear regression.

### "Why does the cost use 1/(2m) but gradients use 1/m?"

> The $\frac{1}{2}$ in the cost cancels the factor of 2 from differentiating the squared term. The optimum is identical either way.

### "Does loss.py train the model?"

> No. It computes cost and gradients. `model.py` performs the parameter updates using gradient descent.

---

## Common mistakes checklist

- [ ] Cost uses $\frac{1}{2m}$; gradients use $\frac{1}{m}$
- [ ] Sum of **squares**, not square of **sum**
- [ ] Parameter name consistency (`X`, not `x` in one place and `X` in another)
- [ ] `X.flatten()` so dot products work with shape `(n, 1)`

---

## Self-check questions

1. What are the three pieces `loss.py` provides?
   → Forward pass, cost function, gradients.

2. What is `mse([0], [1])` with our $\frac{1}{2m}$ formula?
   → **0.5**

3. Does `loss.py` update $w$ and $b$?
   → **No** — that is `model.py`.

4. Why do we need gradients if we already have the cost?
   → Cost is a single number; gradients tell us **which direction** to change $w$ and $b$ to reduce it.

5. What comes after `loss.py`?
   → **`model.py`** — the training loop that uses these functions every epoch.

---

## Code map

```
src/loss.py
├── mse()                 → cost function J(w, b)
├── compute_gradients()   → backward pass: dw, db
└── _predict()            → forward pass: ŷ = wx + b
```

---

## Next step

Build **`src/model.py`** — iteratively call `mse()` and `compute_gradients()`, update $w$ and $b$, and record history for visualization.
