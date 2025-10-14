import streamlit as st
import pandas as pd
import numpy as np



st.set_page_config(page_title="Data Introduction", layout="wide")

# --- Sidebar ---
st.sidebar.title("Data Introduction & Cleaning")
st.sidebar.info("Overview of both datasets, column harmonization, and merging steps.")

# --- Main Title ---
st.title("🚗 Vehicle Price Prediction — Data Introduction")

# --- Load Datasets ---
st.subheader("📂 Raw Datasets")
car_v4 = pd.read_csv("car details v4.csv")
vehicle_df = pd.read_csv("vehicle_price_prediction.csv")

st.write("**Car Details v4 Dataset**")
st.dataframe(car_v4.head())

st.write("**Vehicle Price Prediction Dataset**")
st.dataframe(vehicle_df.head())

# --- Harmonize Columns ---
car_v4_renamed = car_v4.rename(columns={
    'Make': 'make',
    'Model': 'model',
    'Year': 'year',
    'Kilometer': 'mileage',
    'Fuel Type': 'fuel_type',
    'Transmission': 'transmission',
    'Seller Type': 'seller_type',
    'Drivetrain': 'drivetrain',
    'Price': 'price'
})

# Add missing columns to car_v4 to match vehicle_df
for col in vehicle_df.columns:
    if col not in car_v4_renamed.columns:
        car_v4_renamed[col] = None

car_v4_renamed = car_v4_renamed[vehicle_df.columns]

# --- Merge ---
merged_df = pd.concat([vehicle_df, car_v4_renamed], ignore_index=True)
st.write("✅ Merged dataset preview:")
st.dataframe(merged_df.head())

# --- Drop Unnecessary Columns ---
drop_cols = ['exterior_color', 'interior_color', 'accident_history',
             'seller_type', 'trim', 'mileage_per_year', 'condition']
df_new = merged_df.drop(columns=drop_cols, errors='ignore')

# --- Convert Data Types ---
num_cols = ['engine_hp', 'owner_count', 'vehicle_age']
for col in num_cols:
    df_new[col] = pd.to_numeric(df_new[col], errors='coerce').fillna(0).astype(int)

# --- Add Transmission Class ---
df_new['Transmission Class'] = np.where(df_new['transmission'] == 'Automatic', 1, 0)

# --- Remove Duplicate Models (same year) ---
df_new = df_new.drop_duplicates(subset=['model', 'year', 'price'])

st.subheader("🧹 Cleaned Dataset Preview")
st.dataframe(df_new.head())

# --- Quick Summary ---
st.markdown("### Dataset Summary")
st.write(f"Total entries: {len(df_new):,}")
st.write(f"Columns: {', '.join(df_new.columns)}")
st.write("Data types summary:")
st.write(df_new.dtypes)

df_new.to_csv("df_new")



