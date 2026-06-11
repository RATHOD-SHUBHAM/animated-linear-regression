---
title: Animated Linear Regression
emoji: 📈
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: "1.28.0"
app_file: app/streamlit_app.py
pinned: false
license: mit
---

# Animated Linear Regression

Interactive demo of **linear regression trained from scratch** with manual gradient descent (NumPy, no autograd).

Tune the learning rate and epochs — watch the loss curve, regression line, and weight trajectory respond in real time.

## Try this

1. Click **Train Model** with preset **Good (lr=0.01)**
2. Scrub the **epoch slider** on the Regression tab
3. Switch to **Too high lr (0.2)** and retrain — see loss diverge

## Local setup (uv)

Requires [uv](https://docs.astral.sh/uv/).

```bash
cd animated_linear_regression
uv venv alr
uv pip install -r requirements.txt --python alr/bin/python
uv run --python alr/bin/python -m streamlit run app/streamlit_app.py
```

Opens http://localhost:8501

> **Note:** Use `-m streamlit`, not bare `streamlit` — `uv run streamlit` fails if Streamlit isn't in uv's default env. Point `--python` at your `alr` venv after installing deps.

**Alternative (with venv activated):**

```bash
source alr/bin/activate
python -m streamlit run app/streamlit_app.py
```

## CLI pipeline (no UI)

```bash
cd animated_linear_regression
uv run --python alr/bin/python -m src.main
```

Or with venv activated: `python -m src.main`

Outputs saved to `outputs/`.

## Docker

```bash
docker build -t animated-linear-regression .
docker run -p 8501:8501 animated-linear-regression
```

Open http://localhost:8501

## Deploy to Hugging Face Spaces

1. Copy this folder to a new GitHub repo (exclude `alr/`, `outputs/`)
2. Create a [Hugging Face Space](https://huggingface.co/new-space) → SDK: **Streamlit**
3. Connect the repo — HF reads `app_file` from this README frontmatter
4. Space builds automatically

## Project structure

```
├── src/          # ML core (frozen) — data, loss, model, visualize, main
├── app/          # Streamlit UI — imports src/ as library
├── data/         # train.csv, test.csv
├── Dockerfile
└── requirements.txt
```

## ML core (`src/`)

- Manual MSE and gradients in `loss.py`
- Batch gradient descent in `model.py`
- No PyTorch autograd — built for learning and interviews
