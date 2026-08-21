import streamlit as st

st.set_page_config(page_title="Model Information", page_icon="🧠", layout="wide")

st.title("🧠 Model Information")

st.markdown("""
### Algorithms Tested

Eight regression algorithms were trained and compared on the same
train/test split:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. Support Vector Regressor (SVR)
8. XGBoost Regressor *(bonus)*

Each model was evaluated using **MAE**, **MSE**, **RMSE**, and **R²** on a
held-out test set. The top-performing models were then fine-tuned using
**RandomizedSearchCV** (for the more expensive models) and
**GridSearchCV** (for Ridge, which only has one key hyperparameter) to
search for better hyperparameter settings.
""")

st.divider()

st.markdown("### 🏆 Final Selected Model: XGBoost Regressor (Tuned)")

col1, col2, col3, col4 = st.columns(4)
# ---- EDIT THESE TO MATCH YOUR FINAL TUNED XGBOOST RESULTS ----
col1.metric("MAE", "$137.58")
col2.metric("RMSE", "$197.15")
col3.metric("R² Score", "0.882")
col4.metric("MSE", "38,867.53")
# -----------------------------------------------------------------

st.markdown("""
**Why XGBoost performed best:**

- **Captures non-linear, tiered pricing** — component tiers (e.g. GPU/CPU
  tier) don't add a fixed amount to price at every step; top tiers command
  a disproportionate premium. Tree-based splits handle this naturally,
  while linear models assume a straight-line relationship.
- **Captures feature interactions automatically** — e.g. extra RAM only
  drives up price meaningfully when paired with a high-end CPU/GPU. Trees
  can split on one feature conditioned on another; linear models would
  need manually engineered interaction terms to see this.
- **Boosting corrects errors sequentially** — each new tree specifically
  targets the residual errors of the ensemble so far, which is why
  boosting methods (XGBoost, Gradient Boosting) outperformed both a
  single Decision Tree and Random Forest's simple averaging.
- **Handles discrete/spiky numeric features well** — RAM, storage, and
  core counts only take standard fixed values rather than being smoothly
  continuous; trees split cleanly at these breakpoints.
- **Hyperparameter tuning added a further, modest improvement** —
  R² rose from 0.876 (baseline) to 0.882 (tuned), suggesting the default
  configuration was already fairly close to optimal for this dataset.

Full metric comparisons for every model (baseline and tuned) are on the
**Model Comparison** page.
""")
