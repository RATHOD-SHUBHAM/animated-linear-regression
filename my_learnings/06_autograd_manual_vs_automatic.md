# Autograd — Manual Gradients vs Automatic Differentiation

Why comments in `src/loss.py` and `src/model.py` say **"no autograd"**, and how that compares to PyTorch / TensorFlow.

Companion to [04_loss_and_gradients.md](./04_loss_and_gradients.md) and [05_linear_regression_and_training.md](./05_linear_regression_and_training.md).

---

## What does "no autograd" mean?

**Autograd** = **automatic differentiation**. A framework computes gradients for you.

**No autograd** (this project) = **you** derive and implement the gradients by hand in NumPy.

When we write in `loss.py` / `model.py`:

> *"no autograd"*

we mean: gradient descent is implemented **from first principles** — manual gradients, manual weight updates — not using PyTorch, TensorFlow, or JAX.

---

## Two ways to train a model

### 1. Manual (what this project does)

In `src/loss.py` — you wrote the math:

```python
dw = (1 / n) * np.dot(X.flatten(), error)
db = (1 / n) * np.sum(error)
```

In `src/model.py` — you update weights yourself:

```python
self.w -= self.lr * dw
self.b -= self.lr * db
```

**You are responsible for:**
- Defining the loss (MSE)
- Deriving $\frac{\partial L}{\partial w}$ and $\frac{\partial L}{\partial b}$
- Implementing `compute_gradients()`
- Running the training loop

**NumPy only** — no gradient engine.

---

### 2. With autograd (PyTorch, TensorFlow, JAX)

```python
import torch

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * X + b
loss = ((y_pred - y) ** 2).mean()

loss.backward()   # framework computes dw, db automatically

w -= lr * w.grad
b -= lr * b.grad
```

**You define:** forward pass + loss.

**The framework:**
- Tracks operations on tensors
- Builds a computation graph
- Applies the chain rule automatically
- Fills `.grad` on each parameter

You do **not** write `compute_gradients()` by hand.

---

## Side-by-side comparison

| | **This project (NumPy)** | **PyTorch / TensorFlow** |
|---|--------------------------|---------------------------|
| Gradients | Hand-derived in `loss.py` | `loss.backward()` |
| Forward pass | `_predict()` | Same idea, on tensors |
| Weight updates | `w -= lr * dw` | `optimizer.step()` |
| Learning value | Understand the math deeply | Build models faster |
| Interview signal | "I can derive gradients" | "I use PyTorch in production" |

Both run **gradient descent**. The difference is **who computes the derivatives**.

---

## What autograd actually does (intuition)

During the forward pass, the framework builds a **computation graph**:

```
x → *w → +b → y_pred → (y_pred - y)² → mean → loss
```

On `backward()`, it walks **backward** through the graph, applying the **chain rule** at each step:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \text{loss}} \cdot \frac{\partial \text{loss}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}
$$

For linear regression this is simple — a few lines of calculus.

For a deep neural network with millions of parameters, doing that by hand is impractical. **Autograd scales** to arbitrary computation graphs.

---

## Mapping this project to PyTorch concepts

| This project | PyTorch equivalent |
|--------------|-------------------|
| `_predict(X, w, b)` | `model(X)` forward pass |
| `mse(y, y_pred)` | `loss_fn(y_pred, y)` |
| `compute_gradients(...)` | `loss.backward()` |
| `self.w -= lr * dw` | `optimizer.step()` |
| `model.history` | Logged metrics / TensorBoard |

Same algorithm — different level of abstraction.

---

## Why build without autograd first?

| Reason | Explanation |
|--------|-------------|
| **Interview readiness** | Derive and implement gradients for linear/logistic regression and simple nets |
| **Debug PyTorch** | When `loss.backward()` gives NaN, you know where to look |
| **Understand `.backward()`** | Autograd is not magic — it's the chain rule, automated |
| **This project's goal** | Animate gradient descent — you need explicit $w$, $b$, and history |

Once the manual version clicks, PyTorch feels like a convenience layer on top of the same math.

---

## In our pipeline

```
loss.py     → YOU compute dw, db     (manual backward pass)
model.py    → YOU update w, b        (manual optimizer step)
```

PyTorch version would be:

```
model(X)           → forward
loss.backward()    → autograd
optimizer.step()   → update parameters
```

---

## Interview answers

### "What is autograd?"

> Automatic differentiation — a framework tracks operations during the forward pass and computes gradients via the chain rule when you call `backward()`. PyTorch and TensorFlow use it so you don't hand-derive gradients for every layer.

### "Why did you implement linear regression without autograd?"

> To prove I understand what gradient descent actually does — manual MSE, manual gradients, manual updates in NumPy. In production I'd use PyTorch, but the from-scratch version shows I know what `.backward()` and `optimizer.step()` are doing under the hood.

### "Manual gradients vs autograd — trade-off?"

> Manual: better for learning and simple models; doesn't scale to deep nets. Autograd: essential for large models; hides calculus but requires understanding when things break (vanishing gradients, NaN loss, wrong `requires_grad`).

---

## Self-check questions

1. Does our project use PyTorch autograd?
   → **No** — pure NumPy with hand-written gradients.

2. What function replaces `loss.backward()` in our code?
   → **`compute_gradients()`** in `loss.py`.

3. What replaces `optimizer.step()`?
   → **`self.w -= lr * dw`** in `model.py`.

4. Is the training algorithm different without autograd?
   → **No** — still gradient descent. Only gradient *computation* is manual.

5. Why can't you skip learning manual gradients for ML interviews?
   → Many interviews ask you to **derive** MSE gradients or debug training without a framework.
