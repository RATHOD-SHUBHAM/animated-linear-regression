# Testing — Smoke, Unit, Integration, and End-to-End

Quick reference for testing the animated linear regression pipeline. Use this when deciding **what kind of test to write** and **how many of each**.

Related modules: `src/data.py`, `src/loss.py`, `src/model.py`, `src/visualize.py`, `src/main.py`

---

## Quick summary table

| Type | Your understanding | Refinement |
|------|-------------------|------------|
| **Smoke** | Quick, short tests | ✅ Correct — "does it run / not crash?" |
| **Unit** | Testing a function | ✅ Correct — one function or class in isolation |
| **Integration** | 2+ files/modules together | ✅ Correct — e.g. `data.py` → `model.py` |
| **End-to-end (E2E)** | Test everything together | ✅ Correct — full pipeline, like running `main.py` |

---

## One-line definitions (memorize these)

| Type | One sentence |
|------|--------------|
| **Smoke test** | Does the system start and run without immediately crashing? |
| **Unit test** | Does this single function or class behave correctly in isolation? |
| **Integration test** | Do two or more modules work correctly when wired together? |
| **End-to-end (E2E) test** | Does the full pipeline work from entry point to final output? |

---

## Master comparison table

| | **Smoke** | **Unit** | **Integration** | **End-to-end (E2E)** |
|---|---|---|---|---|
| **Scope** | Whole app or module | One function / method / class | 2+ modules together | Full pipeline |
| **Depth** | Shallow | Deep | Medium | Medium–deep |
| **Speed** | Fastest (< 1 s) | Fast (1–5 s) | Medium (5–15 s) | Slowest (10–30+ s) |
| **How many?** | Few | **Most** | Some | Very few (1–2) |
| **Typical tool** | Script or pytest | pytest | pytest | pytest + `main.py` |
| **Catches** | Import errors, crashes, wrong shapes | Logic bugs, wrong math | Interface / data-flow bugs | Broken wiring across the stack |
| **Does NOT catch** | Subtle math errors | Bugs in other modules | Bugs inside one isolated function | Every edge case |

---

## The test pyramid

```
              /   E2E   \           ← 1–2 tests, slowest
             / Integration \        ← 2–4 tests
            /     Unit      \       ← 8–15 tests (most live here)
           /      Smoke       \     ← 2–3 tests, fastest
```

**Rule:** Many unit tests at the base, very few E2E tests at the top. Same pattern used in backend and ML engineering teams.

---

## Important clarification

> **All smoke tests are tests. Not all tests are smoke tests.**

- *"Run tests"* in the office usually means **pytest in CI** (mostly unit + integration).
- *"Run a smoke test"* means a **quick sanity check** before a bigger run or deploy.

---

## Mapped to this project

| Type | What to test | Example |
|------|--------------|---------|
| **Smoke** | Imports + basic load | `load_and_prepare_data()` runs; `X_train.shape[1] == 1` |
| **Unit** | `StandardScaler` | `inverse_transform(fit_transform(X)) ≈ X` |
| **Unit** | `mse()` | `mse(y, y) == 0` |
| **Unit** | `compute_gradients()` | Matches finite-difference gradient |
| **Unit** | `LinearRegressionGD` | Loss after training < loss at epoch 0 |
| **Integration** | data → model | `model.fit(ds.X_train, ds.y_train)` runs; predictions have correct shape |
| **Integration** | model → visualize | History dict produces a plot without error |
| **E2E** | Full pipeline | `main.py`: load → train → save/show loss curve + regression plot |

---

## Suggested folder layout

```
animated_linear_regression/
├── src/
│   ├── data.py
│   ├── loss.py
│   ├── model.py
│   ├── visualize.py
│   └── main.py
└── tests/
    ├── test_smoke.py         # smoke
    ├── test_data.py          # unit
    ├── test_loss.py          # unit
    ├── test_model.py         # unit
    ├── test_integration.py   # integration
    └── test_e2e.py           # end-to-end
```

Run all tests:

```bash
cd animated_linear_regression
source alr/bin/activate
pytest tests/ -v
```

---

## Code examples

### Smoke test

```python
# tests/test_smoke.py
def test_data_loads_without_error():
    from src.data import load_and_prepare_data
    ds = load_and_prepare_data()
    assert ds.X_train.shape[0] > 0
    assert ds.X_train.shape[1] == 1
```

### Unit test

```python
# tests/test_data.py
import numpy as np
from src.data import StandardScaler

def test_scaler_roundtrip():
    scaler = StandardScaler()
    X = np.array([[1.0], [2.0], [3.0]])
    X_back = scaler.inverse_transform(scaler.fit_transform(X))
    assert np.allclose(X, X_back)
```

### Integration test

```python
# tests/test_integration.py
def test_data_to_model():
    from src.data import load_and_prepare_data
    from src.model import LinearRegressionGD

    ds = load_and_prepare_data()
    model = LinearRegressionGD(lr=0.01, n_epochs=50)
    model.fit(ds.X_train, ds.y_train)
    preds = model.predict(ds.X_test)

    assert preds.shape == ds.y_test.shape
    assert model.history["loss"][-1] < model.history["loss"][0]
```

### End-to-end test

```python
# tests/test_e2e.py
def test_main_pipeline_runs():
    from src.main import main
    main()  # no exception = pass; optionally assert output files exist
```

---

## When to add each type (build order)

| Phase | Module built | Tests to add |
|-------|--------------|--------------|
| 1 | `data.py` | Smoke + unit tests for scaler and loader |
| 2 | `loss.py` | Unit tests for MSE and gradients |
| 3 | `model.py` | Unit tests + integration (data → model) |
| 4 | `visualize.py` + `main.py` | Integration + 1 E2E test |

**Do not wait until the end** to write all tests. Add them alongside each module.

---

## Interview answers

### "How do you test ML code?"

> I use a test pyramid: smoke tests for quick sanity checks, unit tests for loss functions and gradient correctness — including numerical gradient checks — integration tests for the data-to-training path, and one or two end-to-end tests that run the full pipeline. Unit tests are the most valuable because they catch math bugs early without needing a full training run.

### "What's the difference between a smoke test and a unit test?"

> A smoke test is a fast, shallow check that the system runs without crashing — for example, data loads and shapes look right. A unit test verifies a specific function is correct with precise assertions — for example, the inverse transform recovers the original values exactly.

### "Why not only E2E tests?"

> E2E tests are slow and when they fail, they don't tell you which layer broke. Unit tests pin down the exact function at fault. E2E tests confirm the wiring is correct at the end.

---

## Glossary (office terms)

| Term | Meaning |
|------|---------|
| **Smoke test** | Quick "does it run?" check |
| **Sanity check** | Informal smoke test |
| **Unit test** | Tests one isolated piece of code |
| **Integration test** | Tests interaction between modules |
| **E2E test** | Tests the full user / pipeline flow |
| **Regression test** | Re-running tests after changes to ensure nothing broke |
| **Test coverage** | How much of the code is exercised by tests (a metric, not a goal by itself) |
| **CI** | Continuous Integration — tests run automatically on every push/PR |

---

## Self-check questions

1. Is `assert X_train.mean() ≈ 0` a smoke test or a unit test?
   → **Smoke-ish** — it checks a property but not one isolated function deeply. A dedicated `test_standardizer_zero_mean()` in `test_data.py` is a proper **unit test**.

2. Testing `data.py` output as input to `model.fit()` — what type?
   → **Integration test.**

3. Running `python -m src.main` and checking a plot is saved — what type?
   → **End-to-end test.**

4. Should you have 20 E2E tests for this project?
   → **No.** 1–2 is enough. Most tests should be unit tests.

5. Your gradient test compares analytical vs numerical gradient — what type?
   → **Unit test** (and a strong one for ML interviews).
