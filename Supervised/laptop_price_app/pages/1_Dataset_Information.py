import streamlit as st

st.set_page_config(page_title="Dataset Information", page_icon="📊", layout="wide")

st.title("📊 Dataset Information")

# ---- EDIT THESE NUMBERS TO MATCH YOUR ACTUAL DATASET ----
N_ROWS = 99999
N_FEATURES = 31
# -----------------------------------------------------------

col1, col2 = st.columns(2)
col1.metric("Total Records", f"{N_ROWS:,}")
col2.metric("Original Features", N_FEATURES)

st.markdown("""
### Feature Overview

The dataset contains specifications for laptops and desktop computers,
covering:

| Category | Features |
|---|---|
| **Device Info** | device_type, brand, model, release_year, os, form_factor |
| **CPU** | cpu_brand, cpu_model, cpu_tier, cpu_cores, cpu_threads, cpu_base_ghz, cpu_boost_ghz |
| **GPU** | gpu_brand, gpu_model, gpu_tier, vram_gb |
| **Memory & Storage** | ram_gb, storage_type, storage_gb, storage_drive_count |
| **Display** | display_type, display_size_in, resolution, refresh_hz |
| **Power** | battery_wh, charger_watts, psu_watts |
| **Connectivity** | wifi, bluetooth |
| **Physical** | weight_kg, warranty_months |
| **Target** | price |

### Key Preprocessing Decisions

- **Dropped `model`, `cpu_model`, `gpu_model`** — near-unique identifier
  strings with no generalizable pattern; risk of overfitting/leakage if encoded.
- **Engineered `gpu_generation` and `gpu_suffix`** from `gpu_model`
  (e.g. "RTX 40 70" → generation 40; "RTX 40 70 Ti" → suffix "Ti") to
  recover the useful signal hidden inside the raw string.
- **Parsed `resolution`** (e.g. "1920x1080") into `resolution_pixels`
  (total pixel count) so the model can use it as a genuine numeric value.
- **Log-transformed `price`** (`np.log1p`) — the raw price distribution
  was right-skewed (skewness ≈ 0.99), so this improves performance for
  models sensitive to skewed targets.
- **`cpu_tier` / `gpu_tier`** were treated as **categorical** (not numeric)
  since the price difference between tiers isn't evenly spaced — a
  flagship tier commands a disproportionately higher premium.

### Key EDA Insights

- Price is right-skewed with a peak around $1,500–2,000 and a long tail
  toward premium/workstation devices.
- No single spec (RAM, CPU cores, storage) predicts price alone — each
  shows a rising price *ceiling* but heavy overlap, meaning price is
  driven by the *combination* of specs.
- `battery_wh` showed a large spike at 0 and `weight_kg` had extreme
  outliers, consistent with desktops being present alongside laptops in
  the dataset.
""")

