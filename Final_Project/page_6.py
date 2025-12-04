import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

st.title("Car Price Estimator (Lasso vs Ridge)")

st.write("""
Enter details of the vehicle below.  
The app will estimate a **price range** using **Lasso** and **Ridge** regression.
""")

# -------------------------
# Load your final dataset
# -------------------------
df_new2 = pd.read_csv("Final_Project/df_final.csv")
df = df_new2.copy()

# -------------------------
# PREPARE ENCODING
# -------------------------

# Categorical variables
categorical_cols = ['fuel_type', 'make', 'fuel_type']

df_encoded = pd.get_dummies(df,columns=categorical_cols,drop_first=True)  
model_features = df_encoded.drop(columns=['price','drivetrain','body_type', 'transmission','model']).columns.tolist()

X = df_encoded.drop(columns=['price','drivetrain','body_type', 'transmission','model'])  # everything EXCEPT price
y = df_encoded['price']
# Save column list for model input
# -------------------------
# TRAIN MODELS
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

ridge = Ridge(alpha=1)
ridge.fit(X_scaled, y)

# -------------------------
# USER INPUT SECTION
# -------------------------

st.subheader("Vehicle Inputs")

# Text + numeric inputs
make = st.selectbox("Make", sorted(df["make"].unique()))
fuel = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
trans = st.selectbox("Transmission", sorted(df["Transmission Class"].unique()))

year = st.number_input("Year", min_value=1980, max_value=2025, value=2018)
mileage = st.number_input("Mileage", min_value=0, max_value=400000, value=60000)
engine_hp = st.number_input("Engine HP", min_value=50, max_value=1200, value=250)
owner_count = st.number_input("Owner Count", min_value=0, max_value=10, value=1)
vehicle_age = st.number_input("Vehicle Age", min_value=0, max_value=40, value=5)
brand_pop = st.number_input("Brand Popularity", min_value=0, max_value=100, value=50)

# -------------------------
# PACK INPUT INTO A ROW
# -------------------------

input_dict = {
    "year": year,
    "mileage": mileage,
    "engine_hp": engine_hp,
    "owner_count": owner_count,
    "vehicle_age": vehicle_age,
    "brand_popularity": brand_pop,
    "make": make,
    "Fuel Type": fuel,
    "Transmission Class": trans
}

input_df = pd.DataFrame([input_dict])

# One-hot encode using same structure as training
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# Add missing columns (ones that exist in training but not in user input)
for col in model_features:
    if col not in input_encoded:
        input_encoded[col] = 0

# Arrange columns in correct order
input_encoded = input_encoded[model_features]

# -------------------------
# SCALE USER INPUT
# -------------------------

input_scaled = scaler.transform(input_encoded)

# -------------------------
# PRICE PREDICTION
# -------------------------

lasso_pred = lasso.predict(input_scaled)[0]
ridge_pred = ridge.predict(input_scaled)[0]

# Create price ranges (±10% buffer)
lasso_low, lasso_high = lasso_pred * 0.9, lasso_pred * 1.1
ridge_low, ridge_high = ridge_pred * 0.9, ridge_pred * 1.1

# -------------------------
# DISPLAY RESULTS
# -------------------------

st.subheader("Predicted Price Ranges")

st.write(f"""
### **Lasso Estimate:**  
**${lasso_low:,.0f} → ${lasso_high:,.0f}**  
(central prediction: ${lasso_pred:,.0f})

### **Ridge Estimate:**  
**${ridge_low:,.0f} → ${ridge_high:,.0f}**  
(central prediction: ${ridge_pred:,.0f})
""")

# Comparison Box
st.subheader("Model Comparison")

diff = abs(lasso_pred - ridge_pred)
st.write(f"Difference between models: **${diff:,.0f}**")

if lasso_pred > ridge_pred:
    st.success("Lasso predicts a higher price.")
elif ridge_pred > lasso_pred:
    st.success("Ridge predicts a higher price.")
else:
    st.info("Both models predict the same value.")
