# Standardization vs Normalization — Differences and When to Use What

Companion to [01_standardization.md](./01_standardization.md). Covers the two most common feature-scaling methods and how to choose between them.

---

## Terminology warning (say this in interviews)

**"Normalization" is overloaded.**

- Some people use *normalization* to mean **any** rescaling (including z-score).
- In ML interviews, *normalization vs standardization* usually means:

| Term | What it usually means |
|------|------------------------|
| **Standardization** | Z-score scaling → mean 0, std 1 |
| **Normalization** | Min-max scaling → range [0, 1] |

If the question is ambiguous, clarify:

> "By normalization, do you mean min-max scaling to [0, 1], or z-score standardization?"

---

## Side-by-side comparison

| | **Standardization (z-score)** | **Normalization (min-max)** |
|---|---|---|
| **Formula** | $x_{\text{scaled}} = \dfrac{x - \mu}{\sigma}$ | $x_{\text{scaled}} = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}}$ |
| **Uses from data** | Mean $\mu$, std $\sigma$ | Min $x_{\min}$, max $x_{\max}$ |
| **Typical result** | Mean ≈ 0, std ≈ 1 | Values in **[0, 1]** |
| **Bounded output?** | No — values can be any real number | Yes — bounded by train min/max |
| **Also called** | Z-score normalization, z-score scaling | Min-max scaling, feature scaling to [0, 1] |

---

## Standardization (z-score) — recap

$$
x_{\text{scaled}} = \frac{x - \mu}{\sigma}
$$

**What it does:** Centers data around 0, then scales by spread (standard deviation).

**Example:** `X_train = [10, 20, 30]`
- $\mu = 20$, $\sigma \approx 8.16$
- 20 → 0, 30 → ≈ 1.22

See [01_standardization.md](./01_standardization.md) for fit/transform workflow, data leakage, and code details.

---

## Normalization (min-max) — recap

$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

**What it does:** Maps the smallest training value to 0 and the largest to 1; everything else lands in between.

**Example:** `X_train = [10, 20, 30]`
- $x_{\min} = 10$, $x_{\max} = 30$
- 10 → 0, 20 → 0.5, 30 → 1

---

## Key differences (intuition)

### 1. What they preserve

| Method | Preserves |
|--------|-----------|
| **Standardization** | *How many standard deviations* a point is from the mean |
| **Min-max** | *Relative position* between min and max |

### 2. Output range

- **Standardization** → unbounded. A test point can be 3.5 or -2.1 in z-score space.
- **Min-max** → nominally [0, 1] on training data. Test points can fall **outside** [0, 1] if they exceed train min/max.

### 3. Sensitivity to outliers

Both are sensitive to outliers, but in different ways:

| Method | Outlier effect |
|--------|----------------|
| **Standardization** | Outliers inflate $\sigma$; most points get compressed |
| **Min-max** | Outliers stretch $x_{\min}$ or $x_{\max}$; most values cluster in a narrow band |

For heavy outliers, consider **RobustScaler** (median + IQR) instead.

---

## When to use standardization

Use **z-score standardization** when:

| Scenario | Why |
|----------|-----|
| **Gradient descent** (linear regression, logistic regression, neural nets) | Balanced gradients, faster convergence |
| **Distance-based models** (k-NN, SVM, K-Means, PCA) | Distance should not be dominated by large-scale features |
| **Regularized models** (Ridge, Lasso) | Penalty applies fairly across features |
| **Features have different units** (age, income, temperature) | Puts them on comparable scales |
| **Data is roughly unbounded** | No natural [0, 1] range |

**Our project:** We standardize `X` in `src/data.py` because we train linear regression with gradient descent.

---

## When to use min-max normalization

Use **min-max scaling to [0, 1]** when:

| Scenario | Why |
|----------|-----|
| **Algorithm expects bounded inputs** | Some neural nets, custom pipelines |
| **Natural bounded features** | Pixel values [0, 255] → [0, 1] |
| **Activation functions need [0, 1]** | Legacy sigmoid-heavy architectures |
| **Relative rank within range matters** | "Where between min and max is this value?" |

---

## When to skip scaling entirely

| Scenario | Why |
|----------|-----|
| **Tree-based models** (Random Forest, XGBoost, decision trees) | Splits depend on order, not absolute scale |
| **All features already on same scale** | Rare in practice, but possible |
| **You will scale inside the model** | Some frameworks handle it internally |

---

## Decision flowchart

```
Do you need inputs strictly in [0, 1]?
├── Yes → Min-max normalization
└── No
    ├── Using gradient descent or distance-based methods?
    │   └── Yes → Standardization (z-score)
    ├── Using tree-based models?
    │   └── Usually skip scaling
    └── Heavy outliers?
        └── RobustScaler or preprocess first
```

---

## Fit on train only — applies to BOTH methods

The same data-leakage rule from [01_standardization.md](./01_standardization.md) applies here:

```python
# Standardization — fit on train
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Min-max — same pattern
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Never compute $\mu$, $\sigma$, $x_{\min}$, or $x_{\max}$ using test data during training.

---

## Interview answers

### "What's the difference between standardization and normalization?"

> Standardization (z-score) transforms features to mean 0 and standard deviation 1 using $(x - \mu) / \sigma$. Normalization (min-max) scales features to a fixed range, usually [0, 1], using $(x - \min) / (\max - \min)$. Standardization is better for gradient-based and distance-based methods; min-max is better when you need bounded inputs.

### "Can test data fall outside [0, 1] after min-max scaling?"

> Yes. Min-max uses training min and max. If a test point is below the training minimum or above the training maximum, its scaled value will be below 0 or above 1. That is expected — we do not refit on test data.

### "Can standardized test values be greater than 1 or less than -1?"

> Yes. Z-scores measure distance from the mean in units of standard deviation. A test point far from the training mean can easily be 2, 3, or -2. There is no fixed bound.

---

## Quick reference table

| Question | Standardization | Min-max |
|----------|-----------------|---------|
| Output mean | ≈ 0 | Not fixed (often ~0.5) |
| Output std | ≈ 1 | Not fixed |
| Output range | Unbounded | [0, 1] on train |
| Best for GD | ✅ Yes | Rarely |
| Best for k-NN / SVM | ✅ Yes | Sometimes |
| Best for trees | Skip scaling | Skip scaling |
| Outlier sensitive | Yes | Yes |

---

## Self-check questions

1. You are building k-NN on features [age, annual_income]. Scale or not? Which method?
   → **Yes, standardization.** Different units and scales; k-NN uses distance.

2. You are training XGBoost on raw tabular data. Scale or not?
   → **Usually no.** Tree splits are scale-invariant.

3. You are feeding pixel values (0–255) into a neural net. Which scaling?
   → **Min-max to [0, 1]** is the natural choice.

4. Your linear regression loss oscillates with lr=0.1 on raw features. What do you try first?
   → **Standardize X** (and possibly lower learning rate).

5. Someone says "always normalize your data." What's wrong with that?
   → **Too vague.** Method depends on algorithm, data distribution, and whether you need bounded outputs. Trees often need no scaling at all.
