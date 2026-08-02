# Laptop Price Predictor — Streamlit App

## Setup

1. Place these two files (from your notebook) into this folder, next to `Home.py`:
   - `laptop_price_model.pkl`
   - `laptop_price_preprocessor.pkl`

2. (Optional but recommended) Export your results table for the Model
   Comparison page. In your notebook, run:
   ```python
   comparison_df.to_csv('comparison_results.csv', index=False)
   ```
   and place `comparison_results.csv` in this folder too. If it's missing,
   the app will show placeholder numbers instead so it still runs.

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run Home.py
   ```

## Before you present this

Go through each file in `pages/` and update the placeholder details to
match your actual project:

- **1_Dataset_Information.py** — row/feature counts, EDA bullet points
- **2_Model_Information.py** — your actual final metric numbers
- **3_Make_Prediction.py** — dropdown options (brand, os, etc.) should
  match the exact category values your training data used — check with
  `df['column_name'].unique()` in your notebook and update the lists here
- **4_Model_Comparison.py** — works automatically once `comparison_results.csv`
  is present
- **5_About_Team.py** — your actual team names and roles

## Folder structure

```
laptop_price_app/
├── Home.py
├── laptop_price_model.pkl            <- add this
├── laptop_price_preprocessor.pkl     <- add this
├── comparison_results.csv            <- add this (optional)
├── requirements.txt
└── pages/
    ├── 1_Dataset_Information.py
    ├── 2_Model_Information.py
    ├── 3_Make_Prediction.py
    ├── 4_Model_Comparison.py
    └── 5_About_Team.py
```
