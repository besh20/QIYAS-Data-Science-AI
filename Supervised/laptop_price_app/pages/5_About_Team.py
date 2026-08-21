import streamlit as st

st.set_page_config(page_title="About Team", page_icon="👥", layout="wide")

st.title("👥 About the Team")

st.markdown("""
### Project Team

Kefyalew, Mihretab, Samuel 

### Course / Project Info

- **Course:** *(Data Science & Machine Learning)*
- **Project:** Laptop & Computer Price Prediction
- **Tools used:** Python, pandas, scikit-learn, XGBoost, Streamlit

### Acknowledgements

Built as part of a regression modeling project covering the full ML
pipeline: exploratory data analysis, feature engineering, preprocessing,
model training and comparison across 8 regression algorithms,
hyperparameter tuning, and deployment as an interactive web application.
""")
