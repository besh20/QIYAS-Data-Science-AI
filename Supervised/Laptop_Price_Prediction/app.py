"""
Laptop / Desktop Price Predictor — Streamlit app.

Run locally:
    streamlit run app.py

Tabs:
    1. Predict          — single-device form -> price estimate
    2. Batch Predict     — upload a CSV of specs -> CSV of predictions
    3. Model Comparison  — baseline vs tuned model leaderboard from the notebook
    4. About / Explainability — feature importance + data/model card
"""

from __future__ import annotations

import io
import textwrap

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    ALL_COLS,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    ModelLoadError,
    PredictionError,
    load_artifacts,
    predict_batch,
    predict_price,
    validate_input,
)


def render_html(html: str) -> None:
    """st.markdown(..., unsafe_allow_html=True) for multi-line HTML.

    Streamlit's markdown parser treats 4-space-indented lines as a code
    block (standard Markdown behavior) — and Python triple-quoted strings
    carry their source indentation right through. Without dedenting first,
    every multi-line HTML block above renders as literal text instead of
    being parsed as HTML. Single-line HTML (no leading whitespace) doesn't
    need this.
    """
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)

st.set_page_config(page_title="Laptop Price Predictor", page_icon="◆", layout="wide")

# --------------------------------------------------------------------------- #
# Visual theme — a "spec sheet / datasheet" look: graphite surface, monospace
# numerals for anything measured (price, specs), a single copper accent used
# only for the number that matters. Pure presentation; no logic below this
# block was touched.
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg: #12161c;
        --surface: #1a212b;
        --surface-2: #212a36;
        --border: #2b3543;
        --text: #e8eaed;
        --text-dim: #8b95a5;
        --accent: #e0a458;
        --accent-dim: #7a6142;
        --good: #4bc98f;
    }

    .stApp { background-color: var(--bg); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }

    /* header banner */
    .app-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 22px 28px; margin-bottom: 6px;
        background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
        border: 1px solid var(--border); border-radius: 10px;
    }
    .app-header h1 {
        font-size: 1.55rem; margin: 0; color: var(--text);
    }
    .app-header .subtitle { color: var(--text-dim); font-size: 0.85rem; margin-top: 4px; }
    .badge-row { display: flex; gap: 10px; }
    .badge {
        font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
        color: var(--accent); background: rgba(224,164,88,0.10);
        border: 1px solid var(--accent-dim); border-radius: 6px;
        padding: 6px 12px; white-space: nowrap;
    }

    /* section labels inside the form */
    .section-label {
        font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.82rem;
        letter-spacing: 0.06em; text-transform: uppercase; color: var(--accent);
        border-bottom: 1px solid var(--border); padding-bottom: 6px; margin-bottom: 12px;
    }

    /* the result card (replaces st.success/st.info blocks) */
    .price-card {
        background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
        padding: 24px 28px; margin-top: 8px;
    }
    .price-card .label {
        font-size: 0.78rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em;
    }
    .price-card .price {
        font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 2.6rem;
        color: var(--accent); line-height: 1.15; margin: 4px 0 14px 0;
    }
    .interval-row { display: flex; align-items: center; gap: 14px; margin-top: 6px; }
    .interval-track {
        flex: 1; height: 6px; border-radius: 3px; background: var(--border); position: relative;
    }
    .interval-fill {
        position: absolute; top: 0; bottom: 0; left: 18%; right: 18%;
        background: var(--accent-dim); border-radius: 3px;
    }
    .interval-labels {
        display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem; color: var(--text-dim); margin-top: 6px;
    }
    .interval-note { font-size: 0.82rem; color: var(--text-dim); margin-top: 10px; line-height: 1.5; }
    .meta-row {
        font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--text-dim);
        margin-top: 16px; padding-top: 12px; border-top: 1px dashed var(--border);
    }

    /* error card */
    .error-card {
        background: rgba(224,88,88,0.08); border: 1px solid #6b3a3a; border-radius: 10px;
        padding: 16px 20px; margin-top: 8px;
    }
    .error-card .label { color: #e08a8a; font-weight: 600; font-size: 0.85rem; margin-bottom: 6px; }
    .error-card li { color: var(--text-dim); font-size: 0.88rem; }

    /* tabs */
    button[data-baseweb="tab"] { font-family: 'Space Grotesk', sans-serif; font-weight: 600; }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
    [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }

    /* primary button */
    .stButton>button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: var(--accent) !important; color: #1a1206 !important; border: none !important;
        font-weight: 600 !important;
    }

    /* number/select inputs: subtle border tightening */
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        border-color: var(--border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    artifacts = get_artifacts()
except ModelLoadError as exc:
    st.error(f"⚠️ Could not start the app: {exc}")
    st.stop()

model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
feature_meta = artifacts["feature_meta"]
comparison_df = artifacts["comparison_df"]
residual_quantiles = artifacts["residual_quantiles"]
model_version = artifacts["model_version"]

render_html(f"""
    <div class="app-header">
    <div>
    <h1>◆ Laptop / Desktop Price Predictor</h1>
    <div class="subtitle">CatBoost regression · trained on ~100k device listings</div>
    </div>
    <div class="badge-row">
    <div class="badge">MAE ≈ $136</div>
    <div class="badge">R² ≈ 0.88</div>
    <div class="badge">saved {model_version}</div>
    </div>
    </div>
""")


tab_predict, tab_batch, tab_compare, tab_about = st.tabs(
    ["Predict", "Batch Predict", "Model Comparison", "About & Explainability"]
)

# --------------------------------------------------------------------------- #
# Tab 1: Single prediction
# --------------------------------------------------------------------------- #
with tab_predict:
    st.subheader("Enter device specs")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        payload: dict = {}

        cat_meta = feature_meta["categorical"]
        num_meta = feature_meta["numerical"]

        with col1:
            st.markdown('<div class="section-label">Identity</div>', unsafe_allow_html=True)
            payload["device_type"] = st.selectbox("Device type", cat_meta["device_type"])
            payload["brand"] = st.selectbox("Brand", cat_meta["brand"])
            payload["os"] = st.selectbox("Operating system", cat_meta["os"])
            payload["form_factor"] = st.selectbox("Form factor", cat_meta["form_factor"])
            payload["release_year"] = st.number_input(
                "Release year", int(num_meta["release_year"]["min"]),
                int(num_meta["release_year"]["max"]) + 1, int(num_meta["release_year"]["max"]),
            )
            payload["warranty_months"] = st.number_input(
                "Warranty (months)", int(num_meta["warranty_months"]["min"]),
                int(num_meta["warranty_months"]["max"]), 12,
            )

        with col2:
            st.markdown('<div class="section-label">CPU / GPU</div>', unsafe_allow_html=True)
            payload["cpu_brand"] = st.selectbox("CPU brand", cat_meta["cpu_brand"])
            payload["cpu_tier"] = st.select_slider("CPU tier (1=entry, 6=flagship)", cat_meta["cpu_tier"])
            payload["cpu_cores"] = st.number_input(
                "CPU cores", int(num_meta["cpu_cores"]["min"]), int(num_meta["cpu_cores"]["max"]), 8
            )
            payload["cpu_threads"] = st.number_input(
                "CPU threads", int(num_meta["cpu_threads"]["min"]), int(num_meta["cpu_threads"]["max"]), 16
            )
            payload["cpu_base_ghz"] = st.number_input(
                "CPU base clock (GHz)", float(num_meta["cpu_base_ghz"]["min"]),
                float(num_meta["cpu_base_ghz"]["max"]), 2.6, step=0.1,
            )
            payload["cpu_boost_ghz"] = st.number_input(
                "CPU boost clock (GHz)", float(num_meta["cpu_boost_ghz"]["min"]),
                float(num_meta["cpu_boost_ghz"]["max"]), 3.5, step=0.1,
            )
            payload["gpu_brand"] = st.selectbox("GPU brand", cat_meta["gpu_brand"])
            payload["gpu_tier"] = st.select_slider("GPU tier (1=integrated, 6=flagship)", cat_meta["gpu_tier"])
            payload["vram_gb"] = st.number_input(
                "VRAM (GB)", int(num_meta["vram_gb"]["min"]), int(num_meta["vram_gb"]["max"]), 6
            )
            gpu_gen_col, gpu_sfx_col = st.columns(2)
            payload["gpu_generation"] = gpu_gen_col.number_input("GPU generation (0 if none)", 0, 60, 40)
            payload["gpu_suffix"] = gpu_sfx_col.selectbox("GPU suffix", ["None", "Ti", "XT", "Super", "Limited"])

        with col3:
            st.markdown('<div class="section-label">Memory / Display / Build</div>', unsafe_allow_html=True)
            payload["ram_gb"] = st.number_input(
                "RAM (GB)", int(num_meta["ram_gb"]["min"]), int(num_meta["ram_gb"]["max"]), 16
            )
            payload["storage_type"] = st.selectbox("Storage type", cat_meta["storage_type"])
            payload["storage_gb"] = st.number_input(
                "Storage (GB)", int(num_meta["storage_gb"]["min"]), int(num_meta["storage_gb"]["max"]), 512
            )
            payload["storage_drive_count"] = st.number_input(
                "Number of drives", int(num_meta["storage_drive_count"]["min"]),
                int(num_meta["storage_drive_count"]["max"]), 1,
            )
            payload["display_type"] = st.selectbox("Display type", cat_meta["display_type"])
            payload["display_size_in"] = st.number_input(
                "Display size (in)", float(num_meta["display_size_in"]["min"]),
                float(num_meta["display_size_in"]["max"]), 15.6, step=0.1,
            )
            res_w = st.number_input("Resolution width (px)", 640, 7680, 1920, step=10)
            res_h = st.number_input("Resolution height (px)", 480, 4320, 1080, step=10)
            payload["resolution_pixels"] = int(res_w) * int(res_h)
            payload["refresh_hz"] = st.number_input(
                "Refresh rate (Hz)", int(num_meta["refresh_hz"]["min"]), int(num_meta["refresh_hz"]["max"]), 60
            )
            payload["battery_wh"] = st.number_input(
                "Battery (Wh, 0 for desktop)", int(num_meta["battery_wh"]["min"]),
                int(num_meta["battery_wh"]["max"]), 56,
            )
            payload["charger_watts"] = st.number_input(
                "Charger (W)", int(num_meta["charger_watts"]["min"]), int(num_meta["charger_watts"]["max"]), 65
            )
            payload["psu_watts"] = st.number_input(
                "PSU (W, 0 for laptop)", int(num_meta["psu_watts"]["min"]), int(num_meta["psu_watts"]["max"]), 0
            )
            payload["weight_kg"] = st.number_input(
                "Weight (kg)", float(num_meta["weight_kg"]["min"]), float(num_meta["weight_kg"]["max"]), 1.8, step=0.1
            )
            payload["wifi"] = st.selectbox("Wi-Fi", cat_meta["wifi"])
            payload["bluetooth"] = st.selectbox("Bluetooth", cat_meta["bluetooth"])

        submitted = st.form_submit_button("Predict price", type="primary", use_container_width=True)

    if submitted:
        errors = validate_input(payload, feature_meta)
        if errors:
            error_items = "".join(f"<li>{e}</li>" for e in errors)
            render_html(f"""
                <div class="error-card">
                <div class="label">⚠ Fix before predicting</div>
                <ul>{error_items}</ul>
                </div>
            """)
        else:
            try:
                result = predict_price(
                    payload, model, preprocessor, feature_meta,
                    residual_quantiles=residual_quantiles,
                )

                interval_html = ""
                if result["price_low"] is not None:
                    conf_pct = int(result["interval_confidence"] * 100)
                    interval_html = (
                        '<div class="interval-row">'
                        '<div class="interval-track"><div class="interval-fill"></div></div>'
                        '</div>'
                        '<div class="interval-labels">'
                        f'<span>${result["price_low"]:,.0f}</span>'
                        f'<span>{conf_pct}% prediction interval</span>'
                        f'<span>${result["price_high"]:,.0f}</span>'
                        '</div>'
                        '<div class="interval-note">'
                        "Built from the model's actual errors on held-out test data — the true "
                        f"price falls in this range {conf_pct}% of the time. A wide range means "
                        "this configuration sits in a part of the market the model is less "
                        "certain about."
                        '</div>'
                    )

                render_html(
                    '<div class="price-card">'
                    '<div class="label">Estimated price</div>'
                    f'<div class="price">${result["predicted_price"]:,.2f}</div>'
                    f'{interval_html}'
                    f'<div class="meta-row">request {result["request_id"]} · {result["latency_ms"]} ms</div>'
                    '</div>'
                )
            except PredictionError as exc:
                render_html(f"""
                    <div class="error-card">
                    <div class="label">⚠ Prediction failed</div>
                    <div style="color: var(--text-dim); font-size: 0.88rem;">{exc}</div>
                    </div>
                """)

# --------------------------------------------------------------------------- #
# Tab 2: Batch prediction
# --------------------------------------------------------------------------- #
with tab_batch:
    st.subheader("Batch prediction from CSV")
    st.caption(f"Required columns (order doesn't matter): `{', '.join(ALL_COLS)}`")

    template_df = pd.DataFrame([{c: "" for c in ALL_COLS}])
    st.download_button(
        "Download empty CSV template",
        template_df.to_csv(index=False).encode(),
        file_name="laptop_specs_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload specs CSV", type=["csv"])
    if uploaded is not None:
        try:
            # keep_default_na=False: pandas otherwise silently reads the literal
            # string "None" (a real gpu_suffix category, meaning "no suffix") as
            # a missing value, which then breaks the encoder downstream. Only
            # genuinely empty cells should be treated as missing.
            input_df = pd.read_csv(uploaded, keep_default_na=False, na_values=[""])
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file as CSV: {exc}")
            input_df = None

        if input_df is not None:
            with st.spinner(f"Predicting {len(input_df)} rows..."):
                try:
                    result_df = predict_batch(
                        input_df, model, preprocessor, feature_meta,
                        residual_quantiles=residual_quantiles,
                    )
                except PredictionError as exc:
                    st.error(str(exc))
                    result_df = None

            if result_df is not None:
                n_failed = (result_df["error"] != "").sum()
                n_ok = len(result_df) - n_failed
                status_color = "var(--good)" if n_failed == 0 else "var(--accent)"
                render_html(f"""
                    <div class="meta-row" style="border-top:none; padding-top:0; margin-top:0; margin-bottom:10px;">
                    <span style="color:{status_color}; font-weight:600;">{n_ok}/{len(result_df)} rows predicted</span>
                    {f' · {n_failed} failed validation, see the error column' if n_failed else ''}
                    </div>
                """)
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    result_df.to_csv(index=False).encode(),
                    file_name="laptop_price_predictions.csv",
                    mime="text/csv",
                )

# --------------------------------------------------------------------------- #
# Tab 3: Model comparison (from the notebook's saved results)
# --------------------------------------------------------------------------- #
with tab_compare:
    st.subheader("Baseline vs. tuned model leaderboard")
    if comparison_df is None:
        st.info("No comparison_results.csv found in artifacts/.")
    else:
        sort_col = st.selectbox("Sort by", ["R2", "MAE", "RMSE", "MSE"], index=0)
        ascending = sort_col != "R2"
        st.dataframe(
            comparison_df.sort_values(sort_col, ascending=ascending).reset_index(drop=True),
            use_container_width=True,
        )
        best_row = comparison_df.sort_values("R2", ascending=False).iloc[0]
        st.caption(
            f"Best model: **{best_row['Model']}** — "
            f"R²={best_row['R2']:.3f}, MAE=${best_row['MAE']:.1f}, RMSE=${best_row['RMSE']:.1f}"
        )
        st.bar_chart(comparison_df.set_index("Model")["R2"])

# --------------------------------------------------------------------------- #
# Tab 4: About + explainability
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("About this model")
    st.markdown(
        """
- **Task**: predict device price (USD) from hardware/software specs (regression).
- **Data**: [`all-computer-prices`](https://www.kaggle.com/datasets/paperxd/all-computer-prices) on Kaggle, ~100k rows.
- **Target transform**: `log1p(price)` during training, `expm1()` on predictions.
- **Preprocessing**: `StandardScaler` on 17 numerical features, `OneHotEncoder(handle_unknown="ignore")` on 13 categorical features.
- **Final model**: CatBoost, hyperparameter-tuned with `RandomizedSearchCV` (5-fold CV, scored on MSE).
- **Known limitation**: predictions on inputs far outside the training distribution
  (e.g. a 128-core CPU) are extrapolations and should not be trusted — the app
  validates against the training range before predicting.
        """
    )

    st.subheader("Global feature importance")
    if hasattr(model, "feature_importances_"):
        try:
            feature_names = preprocessor.get_feature_names_out()
            importances = pd.Series(model.feature_importances_, index=feature_names)
            top20 = importances.sort_values(ascending=False).head(20)
            st.bar_chart(top20)
        except Exception as exc:  # noqa: BLE001
            st.info(f"Could not compute feature importance: {exc}")
    else:
        st.info("Loaded model does not expose feature_importances_.")

    st.subheader("Recent prediction log")
    from utils import PREDICTION_LOG_PATH
    if PREDICTION_LOG_PATH.exists():
        log_df = pd.read_csv(PREDICTION_LOG_PATH).tail(20)
        st.dataframe(log_df, use_container_width=True)
    else:
        st.caption("No predictions logged yet this session.")