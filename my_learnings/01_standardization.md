# Standardization — Concepts, Code, and Interview Answers

Notes from building `src/data.py` for the animated linear regression project.

---

## What is standardization?

**Standardization** is a preprocessing technique that rescales features so they have **mean 0** and **standard deviation 1**. It is also called **z-score normalization** or **z-score scaling**.

For a feature value $x$:

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

Where:
- $\mu$ = mean of the feature
- $\sigma$ = standard deviation of the feature

### How to standardize (two steps)

1. Compute $\mu$ and $\sigma$ for each feature (from training data only).
2. For each value: subtract $\mu$, then divide by $\sigma$.

### Example

If `X_train = [10, 20, 30]`:
- $\mu = 20$
- $\sigma \approx 8.16$
- Standardized value of 20 → $(20 - 20) / 8.16 = 0$
- Standardized value of 30 → $(30 - 20) / 8.16 \approx 1.22$

After standardization, the feature is **centered** (mean ≈ 0) and **scaled** (std ≈ 1).

---

## Why use standardization in machine learning?

| Reason | Explanation |
|--------|-------------|
| **Mean centering** | Subtracting $\mu$ shifts each feature to mean 0, so the model focuses on *relative* variation around the center rather than raw magnitude. |
| **Comparable scales** | Dividing by $\sigma$ puts features on similar scales so large-magnitude features do not dominate learning. |
| **Faster convergence** | Gradient descent (and similar optimizers) converges faster with standardized features; steps are more balanced across dimensions. |
| **Stable training** | Large raw values can cause huge gradients and numerical instability (overflow/underflow). |
| **Fair regularization** | In Ridge/Lasso, penalties apply fairly only when all features are on comparable scales. |
| **Interpretable coefficients** | In multi-feature regression, standardized coefficients reflect relative feature importance. |

### Deeper intuition

**Mean centering** — Standardization subtracts the mean from each feature, so the transformed data is centered at 0. That helps the model capture how far each point deviates from the average, not just its absolute value.

**Comparable scales** — Standardization divides by standard deviation, so features measured in different units (e.g. age vs. income) contribute on a similar scale. Without this, high-variance or large-unit features can dominate the loss and gradients.

For our **1-feature** linear regression animation, the main benefits are **faster, stable gradient descent** with a reasonable learning rate.

---

## How our code implements it

In `src/data.py`, the `StandardScaler` class stores two values learned during `fit()`:

| Attribute | Symbol | Meaning |
|-----------|--------|---------|
| `mean_` | $\mu$ | Mean of the feature, computed from training data |
| `scale_` | $\sigma$ | Standard deviation of the feature, computed from training data |

`scale_` is the standard deviation. It is named `scale_` (following sklearn convention) because it is the **divisor** in the transform step:

```python
x_scaled = (x - mean_) / scale_
```

### What is `eps`?

`eps` is a tiny constant (default `1e-8`) used as a safety check.

If a feature is **constant** (every value is the same), then $\sigma = 0$ and we would divide by zero:

```python
(x - mean) / 0  # → inf or nan
```

Our fix:

```python
scale_ = 1.0 if std < eps else std
```

When std is zero, `(x - mean) / 1 = 0` for every point — a safe, correct result.

---

## Fit on train, transform test — the correct workflow

### The rule

1. **`fit(X_train)`** — compute $\mu$ and $\sigma$ from training data only
2. **`transform(X_train)`** — apply $(x - \mu) / \sigma$ to training data
3. **`transform(X_test)`** — apply the **same** $\mu$ and $\sigma$ to test data

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # learn μ, σ then transform train
X_test = scaler.transform(X_test)        # reuse μ, σ; never refit on test
```

### Interview answer: "Walk me through standardization for train vs test."

> I compute the mean and standard deviation on the **training set only** during `fit()`. Then I transform both train and test using those same statistics. The test set never influences $\mu$ or $\sigma$. At inference time, I apply the same scaler that was fit during training — I never recompute statistics from new data inside the training pipeline.

---

## Why must we fit the scaler on X_train only?

At **training time**, the model must only see information that would realistically be available **before** deployment.

- $\mu$ and $\sigma$ are **preprocessing parameters**, just like learned weights.
- They describe the **training distribution**.
- Test data simulates **unseen future data**. Using it to compute $\mu$ or $\sigma$ gives the pipeline information it should not have.

If you fit on test data:
- Your preprocessing **knows** something about the test distribution.
- Reported test metrics become **optimistically biased**.
- The pipeline fails in production, where true future data was never part of fit.

---

## What goes wrong if you fit on train + test combined?

This is **data leakage** through preprocessing.

### Concrete example

```python
# WRONG — do not do this
X_all = np.concatenate([X_train, X_test])
scaler.fit(X_all)
```

Problems:

1. **Test statistics leak into preprocessing** — $\mu$ and $\sigma$ are influenced by test samples the model should treat as unseen.
2. **Overly optimistic evaluation** — test metrics no longer reflect real-world generalization.
3. **Broken production behavior** — in deployment you only have training history to compute $\mu$ and $\sigma$; the combined fit is not reproducible.

### Analogy

Fitting on train + test is like letting a student **see exam questions** while studying, then reporting their exam score as proof they generalize.

---

## What we standardize in this project (and what we don't)

| Array | Standardized? | Why |
|-------|---------------|-----|
| `X_train`, `X_test` | Yes | Stabilizes gradient descent |
| `y_train`, `y_test` | No | Keeps loss and plots in original units (easier to interpret) |

If you plot the regression line in **original x units**, use `inverse_transform()`:

```python
x_line_orig = scaler.inverse_transform(x_line_scaled)
```

---

## Quick reference — interview follow-ups

**Q: Standardization vs normalization?**
- Standardization → z-score: $(x - \mu) / \sigma$, mean 0, std 1
- Normalization (min-max) → $(x - \min) / (\max - \min)$, scales to [0, 1]

**Q: `np.std` ddof=0 vs ddof=1?**
- `ddof=0` → population std (sklearn `StandardScaler` default)
- `ddof=1` → sample std (Bessel correction)
- Be consistent; mention which you use if asked.

**Q: Do you standardize before or after train/test split?**
- **After split.** Fit on train only, then transform both splits.

**Q: Should you standardize y in linear regression?**
- Optional. For this project we keep y raw for interpretability. Standardizing y can help optimization but changes how you read coefficients and loss.

---

## Code map (where things live)

```
animated_linear_regression/
├── data/
│   ├── train.csv
│   └── test.csv
└── src/
    └── data.py
        ├── StandardScaler   # fit / transform / inverse_transform
        ├── load_and_prepare_data()
        └── Dataset          # dataclass holding prepared arrays
```

---

## Self-check questions

1. If `fit()` is called on test data by mistake, is that leakage?
   → **Yes.**

2. After `fit(X_train)`, can test points have values outside the train range when standardized?
   → **Yes.** Test points can be > 1 or < -1 in z-score space; that is expected.

3. Should you call `fit()` again on production inference data?
   → **No.** Reuse the scaler saved from training.
