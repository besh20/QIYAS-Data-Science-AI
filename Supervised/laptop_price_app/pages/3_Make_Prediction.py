import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Make Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Predict a Device's Price")

# ------------------------------------------------------------------
# Load model + preprocessor once (cached so it doesn't reload on every click)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "laptop_price_model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "laptop_price_preprocessor.pkl"))
    return model, preprocessor

try:
    model, preprocessor = load_artifacts()
except FileNotFoundError:
    st.error(
        "Could not find `laptop_price_model.pkl` or `laptop_price_preprocessor.pkl` "
        "in the app folder. Make sure both files are placed alongside `Home.py`."
    )
    st.stop()

st.markdown("Fill in the specs below and click **Predict Price**.")

# ------------------------------------------------------------------
# Input form
# EDIT the dropdown options below to match the exact category values
# your training data used (check df['col'].unique() in your notebook)
# ------------------------------------------------------------------
with st.form("prediction_form"):

    st.subheader("Device")
    c1, c2, c3 = st.columns(3)
    device_type = c1.selectbox("Device Type", ["Laptop", "Desktop"])
    brand = c2.selectbox("Brand", ["Dell", "HP", "Lenovo", "Apple", "Asus", "Acer", "MSI", "Razer", "Samsung", "Gigabyte"])
    release_year = c3.number_input("Release Year", min_value=2015, max_value=2026, value=2025)

    c1, c2, c3 = st.columns(3)
    os_choice = c1.selectbox("Operating System", ["Windows", "macOS", "Linux", "ChromeOS"])
    form_factor = c2.selectbox("Form Factor", [ "2-in-1","ATX", "Gaming", "WorkStation", "Ultrabook","Mainstream", "Full-Tower", "Mini-ITX", "Micro-ATX", "SFF"])
    warranty_months = c3.number_input("Warranty (months)", min_value=0, max_value=60, value=12)

    st.subheader("CPU")
    c1, c2, c3 = st.columns(3)
    cpu_brand = c1.selectbox("CPU Brand", ["Intel", "AMD", "Apple"])
    cpu_tier = c2.selectbox("CPU Tier", [1, 2, 3, 4, 5, 6])
    cpu_cores = c3.number_input("CPU Cores", min_value=2, max_value=32, value=8)

    c1, c2, c3 = st.columns(3)
    cpu_threads = c1.number_input("CPU Threads", min_value=2, max_value=64, value=16)
    cpu_base_ghz = c2.number_input("CPU Base Clock (GHz)", min_value=1.0, max_value=5.0, value=2.8, step=0.1)
    cpu_boost_ghz = c3.number_input("CPU Boost Clock (GHz)", min_value=1.0, max_value=6.0, value=4.5, step=0.1)

    st.subheader("GPU")
    c1, c2, c3 = st.columns(3)
    gpu_brand = c1.selectbox("GPU Brand", ["Nvidia", "AMD", "Intel", "Apple"])
    gpu_tier = c2.selectbox("GPU Tier", [1, 2, 3, 4, 5, 6])
    vram_gb = c3.number_input("VRAM (GB)", min_value=0, max_value=32, value=6)

    c1, c2 = st.columns(2)
    gpu_generation = c1.number_input("GPU Generation (e.g. 40 for RTX 40-series, 1/2 for Arc A/B)", min_value=0, max_value=99, value=40)
    gpu_suffix = c2.selectbox("GPU Suffix", ["None", "Ti", "Super", "XT", "Limited"])

    st.subheader("Memory & Storage")
    c1, c2, c3 = st.columns(3)
    ram_gb = c1.selectbox("RAM (GB)", [4, 8, 16, 32, 64, 128])
    storage_type = c2.selectbox("Storage Type", ["SSD", "HDD", "Hybrid", "NVMe"])
    storage_gb = c3.selectbox("Storage (GB)", [256, 512, 1024, 2048, 4096])

    storage_drive_count = st.number_input("Number of Storage Drives", min_value=1, max_value=4, value=1)

    st.subheader("Display")
    c1, c2, c3 = st.columns(3)
    display_type = c1.selectbox("Display Type", ["IPS", "OLED", "TN", "VA","LED", "Mini-LED", "QLED"])
    display_size_in = c2.number_input("Display Size (in)", min_value=10.0, max_value=40.0, value=15.6, step=0.1)
    refresh_hz = c3.selectbox("Refresh Rate (Hz)", [60, 90, 120, 144, 165, 240])

    resolution_choice = st.selectbox(
        "Resolution",
        ["1920x1080", "2560x1440", "2560x1600", "2880x1800", "3440x1440", "3840x2160"]
    )
    w, h = map(int, resolution_choice.split("x"))
    resolution_pixels = w * h

    st.subheader("Power & Connectivity")
    c1, c2, c3 = st.columns(3)
    battery_wh = c1.number_input("Battery (Wh) — 0 for desktops", min_value=0, max_value=120, value=60)
    charger_watts = c2.number_input("Charger (W)", min_value=0, max_value=400, value=90)
    psu_watts = c3.number_input("PSU (W) — 0 for laptops", min_value=0, max_value=1500, value=0)

    c1, c2 = st.columns(2)
    wifi = c1.selectbox("WiFi", ["WiFi 5", "WiFi 6", "WiFi 6E", "WiFi 7"])
    bluetooth = c2.selectbox("Bluetooth", ["Yes", "No"])

    st.subheader("Physical")
    weight_kg = st.number_input("Weight (kg)", min_value=0.5, max_value=20.0, value=1.8, step=0.1)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# ------------------------------------------------------------------
# Run prediction
# ------------------------------------------------------------------
if submitted:
    input_df = pd.DataFrame([{
        "device_type": device_type, "brand": brand, "release_year": release_year,
        "os": os_choice, "form_factor": form_factor,
        "cpu_brand": cpu_brand, "cpu_tier": cpu_tier, "cpu_cores": cpu_cores,
        "cpu_threads": cpu_threads, "cpu_base_ghz": cpu_base_ghz, "cpu_boost_ghz": cpu_boost_ghz,
        "gpu_brand": gpu_brand, "gpu_tier": gpu_tier, "vram_gb": vram_gb,
        "ram_gb": ram_gb, "storage_type": storage_type, "storage_gb": storage_gb,
        "storage_drive_count": storage_drive_count,
        "display_type": display_type, "display_size_in": display_size_in,
        "refresh_hz": refresh_hz, "resolution_pixels": resolution_pixels,
        "battery_wh": battery_wh, "charger_watts": charger_watts, "psu_watts": psu_watts,
        "wifi": wifi, "bluetooth": bluetooth,
        "weight_kg": weight_kg, "warranty_months": warranty_months,
        "gpu_generation": gpu_generation, "gpu_suffix": gpu_suffix,
    }])

    try:
        processed = preprocessor.transform(input_df)
        pred_log = model.predict(processed)
        pred_price = np.expm1(pred_log)[0]

        st.success("Prediction complete!")
        st.metric("💰 Predicted Price", f"${pred_price:,.2f}")

        with st.expander("See the exact specs submitted"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means a column name or category value here doesn't "
            "exactly match what the preprocessor was trained on. Double check "
            "the column list and dropdown options against your training notebook."
        )
