import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

st.title("Price Estimator")

st.write("""Enter what type of car you are looking for and the model will predict the price range
         for you dream car.
""")

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("Final_Project/df_final.csv")

# -------------------------
# PREPARE ENCODING
# -------------------------

# Correct categorical column names
categorical_cols = ['fuel_type', 'Transmission Class']

# Encode df
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# X and y
X = df_encoded.drop(columns=['price','drivetrain','body_type', 'transmission','model', 'make'])  # everything EXCEPT price
y = df_encoded['price']

model_features = X.columns.tolist()

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


fuel = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
st.write("For Transmission please select 0 for Manual and 1 for Automatic")
trans = st.selectbox("Transmission Class", sorted(df["Transmission Class"].unique()))
year = st.number_input("Year", 1980, 2025, 2018)
mileage = st.number_input("Mileage", 0, 400000, 60000)
engine_hp = st.number_input("Engine HP", 50, 1200, 250)
owner_count = st.number_input("Owner Count", 0, 10, 1)
vehicle_age = st.number_input("Vehicle Age", 0, 40, 5)
brand_pop = st.number_input("Brand Popularity",min_value=0.039318,max_value=0.040484,value=0.039800,step=0.00001)
# -------------------------
# PACK INPUT
# -------------------------

input_dict = {
    "year": year,
    "mileage": mileage,
    "engine_hp": engine_hp,
    "owner_count": owner_count,
    "vehicle_age": vehicle_age,
    "brand_popularity": brand_pop,
    "fuel_type": fuel,
    "Transmission Class": trans
}

input_df = pd.DataFrame([input_dict])

# One-hot encode user input
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# Add missing columns
for col in model_features:
    if col not in input_encoded:
        input_encoded[col] = 0

# Reorder
input_encoded = input_encoded[model_features]

# -------------------------
# OUTPUT
# -------------------------

if st.button("Estimate Price"):

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predictions
    lasso_pred = lasso.predict(input_scaled)[0]
    ridge_pred = ridge.predict(input_scaled)[0]

    # Price ranges
    lasso_low, lasso_high = lasso_pred * 0.9, lasso_pred * 1.1
    ridge_low, ridge_high = ridge_pred * 0.9, ridge_pred * 1.1

    st.subheader("Predicted Price Ranges")

    st.write(f"""
    ### **Lasso Regression Range Estimate:**  
    Minimum Value:{lasso_low:,.0f}
    Maximum Value:{lasso_high:,.0f}
    

    ### **Ridge Regression Range Estimate:**  
    Minimum Value: {ridge_low:,.0f}
    Maximum Value: {ridge_high:,.0f}
    
    """)

    st.subheader("Comparison")
    diff = abs(lasso_pred - ridge_pred)
    st.write(f"Difference: **${diff:,.0f}**")

    if lasso_pred > ridge_pred:
        st.success("Lasso predicts a higher price.")
    elif ridge_pred > lasso_pred:
        st.success("Ridge predicts a higher price.")
    else:
        st.info("Both models predict the same value.")

    st.write("""
            Obviously this model has some issues with getting the correct price range. 
            This is due to the fact that it requires a lot more data points to understand how each feature works, 
            and it also requires much more diversity using other brands. We also have not included the model into our dataset 
            because of the same reasons (there are no large variety). For an initial model estimation, however, it is a pretty good estimation.""")