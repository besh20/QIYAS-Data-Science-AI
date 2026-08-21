import streamlit as st

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

st.title("💻 Laptop & Computer Price Predictor")

st.markdown("""
## Project Description

This project builds a machine learning system that predicts the price of a
laptop or computer based on its technical specifications — things like CPU,
GPU, RAM, storage, display, and battery details.

The goal was to take a raw dataset of device specs and prices, explore it,
clean and engineer useful features, train and compare several regression
algorithms, tune the best-performing ones, and finally package the winning
model into this interactive web application.

### What you can do here
Use the sidebar to navigate between pages:

- **Dataset Information** — an overview of the data used to train the models
- **Model Information** — details on the algorithms tested and the final chosen model
- **Make Prediction** — enter a device's specs and get a predicted price
- **Model Comparison** — see how all 8 regression models performed side-by-side
- **About Team** — who built this project

### How it works (in short)
1. Raw specs (CPU, GPU, RAM, storage, etc.) are cleaned and encoded
2. The processed features are fed into a tuned **XGBoost Regressor**
   (the best-performing model out of 8 tested algorithms)
3. The model outputs a predicted price, trained on log-transformed prices
   to handle the natural right-skew of price data
""")

st.info("👈 Use the sidebar to get started — try **Make Prediction** to test the model yourself.")
