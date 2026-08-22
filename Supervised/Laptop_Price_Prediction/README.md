# Laptop / Desktop Price Predictor

Predicts the price (USD) of a laptop or desktop from its hardware and
software specs. Trained on the [`all-computer-prices`](https://www.kaggle.com/datasets/paperxd/all-computer-prices)
Kaggle dataset (~100k rows).

**Final model**: CatBoost, tuned via `RandomizedSearchCV` — MAE ≈ $136,
RMSE ≈ $195, R² ≈ 0.884 on held-out test data (see `artifacts/comparison_results.csv`
for the full leaderboard: Linear/Ridge/Lasso, Decision Tree, Random Forest,
Gradient Boosting, XGBoost, CatBoost, LightGBM, ExtraTrees, AdaBoost, SVR).

## Project layout

```
laptop_app/
├── app.py                  # Streamlit UI: predict, batch predict, model comparison, explainability
├── api.py                  # FastAPI REST service (/health, /predict)
├── utils.py                # Shared loading / validation / prediction / logging logic
├── requirements.txt
├── artifacts/
│   ├── laptop_price_model.pkl
│   ├── laptop_price_preprocessor.pkl
│   ├── feature_meta.json    # min/max/allowed values per feature, used for form UI + validation
│   └── comparison_results.csv
├── logs/
│   └── predictions.csv      # append-only audit log, written at runtime
└── tests/
    └── test_predictions.py
```

The notebook that trains and saves `laptop_price_model.pkl` /
`laptop_price_preprocessor.pkl` lives separately; this folder only *serves*
those artifacts. Re-run the notebook and re-copy the two `.pkl` files into
`artifacts/` whenever you retrain.

## Running it

```bash
pip install -r requirements.txt

# UI
streamlit run app.py

# API (separate process)
uvicorn api:app --reload --port 8000   # docs at /docs

# Tests
pytest -v
```

## What was added to make this production-ready

The notebook alone proves the modeling works. These are the pieces that turn
it into something you'd actually hand to a user or another service:

1. **Serving layer decoupled from the notebook.** `utils.py` is the single
   source of truth for loading artifacts and running a prediction — the UI
   and the API both call into it instead of duplicating logic (and drifting
   out of sync with each other).
2. **Input validation, not just `handle_unknown='ignore'`.** The
   `OneHotEncoder` would silently zero-out an unrecognized category, which
   hides typos and bad client data. `validate_input()` checks required
   fields, rejects out-of-training-range numerics (extrapolation warning),
   and rejects unknown categoricals *before* they reach the model.
3. **A real API, not just a UI.** `api.py` exposes `/predict` with a Pydantic
   schema (so bad requests get a 422 with a clear message, not a 500) and
   `/health` for readiness checks — the shape a real deployment (load
   balancer, another microservice, a mobile client) expects.
4. **Batch prediction.** Upload a CSV of many devices, get predictions back
   as a CSV, with per-row errors instead of one failure killing the whole
   batch.
5. **Prediction logging / audit trail.** Every prediction is appended to
   `logs/predictions.csv` with inputs, output, latency, and a request id —
   the minimum needed to later check for data drift or debug "why did it
   predict that."
6. **Model versioning hook.** `model_version` is surfaced in both the UI and
   API responses. Right now it's the artifact's file timestamp; swap in a
   `model_card.json` written at training time (name, date, metrics, git
   commit) for real traceability.
7. **Explainability.** The "About" tab shows CatBoost's global feature
   importances, so the app doesn't just spit out a number with no reasoning
   behind it.
8. **Tests that exercise the actual saved artifacts.** `tests/test_predictions.py`
   loads the real `.pkl` files and checks validation, prediction sanity
   bounds, and both API endpoints — these are the tests that catch "the
   preprocessor and model were trained on different schemas."
10. **Error handling throughout.** Missing artifacts, bad CSV uploads, and
    malformed requests all produce a specific message instead of a stack
    trace in the user's face.

## Recently added

**Prediction intervals.** Every prediction now returns `price_low` /
`price_high` (an 80% interval) alongside the point estimate, in both the UI
and the API. This is *not* a made-up +/- percentage — it's built from
`artifacts/residual_quantiles.json`, the 10th/90th percentile of
`(actual − predicted)` in log-price space measured on the real held-out test
set (same split as the notebook: `test_size=0.2, random_state=42`). Practical
effect: the interval automatically reflects where the model is confident
(common configs) vs. uncertain (rare spec combinations), instead of being a
flat number slapped on every prediction.

To regenerate it after retraining (the file will go stale if the model or
preprocessor changes): reload `computer_prices_all.csv`, reapply the same
feature engineering as the notebook (gpu_generation/gpu_suffix extraction,
resolution_pixels, log1p target), re-split with the same `random_state`,
transform the test set, compute `y_test - model.predict(X_test)`, and take
the 10th/90th percentiles. See git history for the exact script, or ask
for it again — it's about 30 lines.

**CI.** `.github/workflows/tests.yml` runs `pytest` (and checks the model
artifacts are actually present) on every push and PR to any branch. Push
this to GitHub and you'll get a pass/fail badge for free — cheap, visible
proof the project isn't just "worked once on my laptop."

## Suggested next steps (didn't build these in, but worth knowing about)

- **Drift monitoring**: a scheduled job comparing `logs/predictions.csv`
  input distributions against the training distribution.
- **SHAP explanations per-prediction** (not just global importance) — "why
  did *this* laptop get priced at $1,500."
- **Auth/rate-limiting** on the API if it's ever exposed publicly.
- **A proper `model_card.json`** written by the notebook at save time
  instead of relying on file mtime for versioning.
- **CI badge in this README** and auto-deploy step (e.g. to Streamlit
  Community Cloud or a container registry) once you push this to GitHub.
