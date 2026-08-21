import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")

st.title("📈 Model Comparison")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE_DIR, "comparison_results.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    st.warning(
        "`comparison_results.csv` not found — showing placeholder data. "
        "In your notebook, run:\n\n"
        "`comparison_df.to_csv('comparison_results.csv', index=False)`\n\n"
        "then place that file in the app folder to show your real results."
    )
    df = pd.DataFrame({
        "Model": ["XGBoost (Tuned)", "Gradient Boosting (Tuned)", "XGBoost", "SVR (Tuned)",
                  "Ridge Regression", "Linear Regression", "Gradient Boosting", "SVR",
                  "Random Forest", "Lasso Regression", "Decision Tree"],
        "MAE": [137.58, 137.97, 141.13, 144.93, 148.06, 148.06, 152.03, 153.27, 157.07, 205.61, 231.03],
        "RMSE": [197.15, 198.01, 202.44, 205.92, 210.16, 210.16, 213.16, 216.37, 220.84, 277.08, 337.15],
        "R2": [0.8823, 0.8813, 0.8759, 0.8716, 0.8662, 0.8662, 0.8624, 0.8582, 0.8523, 0.7675, 0.6558],
        "Version": ["Tuned", "Tuned", "Baseline", "Tuned", "Baseline", "Baseline",
                     "Baseline", "Baseline", "Baseline", "Baseline", "Baseline"]
    })

df_sorted = df.sort_values("R2", ascending=False).reset_index(drop=True)

st.markdown("### Full Results Table")
st.dataframe(
    df_sorted.style.highlight_min(subset=["MAE", "RMSE", "MSE"], color="#046152")
                     .highlight_max(subset=["R2"], color="#046152"),
    use_container_width=True
)

st.markdown("### Visual Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**RMSE by Model** (lower is better)")
    st.bar_chart(df_sorted.set_index("Model")["RMSE"])

with col2:
    st.markdown("**R² Score by Model** (higher is better)")
    st.bar_chart(df_sorted.set_index("Model")["R2"])

st.divider()

st.markdown("""
### What each metric means

| Metric | What it measures | Better values are... |
|---|---|---|
| **MAE** | Average absolute dollar error between predicted and actual price | Lower |
| **MSE** | Average *squared* error — penalizes large mistakes more heavily | Lower |
| **RMSE** | Square root of MSE — back in dollar units, still sensitive to big misses | Lower |
| **R²** | Proportion of price variance explained by the model (1.0 = perfect) | Higher (closer to 1) |

**Note on SVR:** Due to its poor scalability (training time scales roughly
O(n²)–O(n³) with dataset size), SVR was trained on a random 8,000-row
subsample rather than the full training set — a standard, well-documented
workaround for this algorithm's known limitation on large datasets.
""")
