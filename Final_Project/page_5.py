import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

st.title("Lasso vs Ridge Regression Explorer")

# ---------------------------
# Load your dataset
# ---------------------------
df =  pd.read_csv("Final_Project/df_final.csv") # <-- your dataset



categorical_cols = ['fuel_type']

df_encoded = pd.get_dummies(df,columns=categorical_cols,drop_first=True)      
X = df_encoded.drop(columns=['price','drivetrain','body_type', 'transmission','model','make'])  # everything EXCEPT price
y = df_encoded['price']

# ---------------------------
# Model selector
# ---------------------------
model_choice = st.radio(
    "Choose Regression Model:",
    ["Lasso Regression", "Ridge Regression"]
)

# ---------------------------
# Model Hyperparameters
# ---------------------------
if model_choice == "Lasso Regression":
    alpha = st.slider("Lasso Alpha (λ)", 0.01, 5.0, 0.1, 0.01)
else:
    alpha = st.slider("Ridge Alpha (λ)", 0.01, 5.0, 1.0, 0.01)

# ---------------------------
# Run Model Button
# ---------------------------
if st.button("Run Regression Model"):

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # Fit Selected Model
    # ---------------------------
    if model_choice == "Lasso Regression":
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)

    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    residuals = y_test - pred

    # ---------------------------
    # Output Results
    # ---------------------------
    st.subheader(f"Results: {model_choice}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean-Squared Error", f"{mse:,.2f}")
    col2.metric("Root Mean-Squared Error", f"{rmse:,.2f}")
    col3.metric("R² Score", f"{r2:.4f}")

    st.write("### Coefficients")
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    })
    st.dataframe(coef_df)

    # Plot Actual vs Predicted
    st.write("Actual vs Predicted")
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    sns.regplot(x=y_test, y=pred, ax = ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(model_choice)
    st.pyplot(fig)

    st.write("Residual Distribution")

    fig_res, ax_res = plt.subplots()
    ax_res.hist(residuals, bins=30, alpha=0.7)
    ax_res.set_title(f"Residual Distribution")
    ax_res.set_xlabel("Residual")
    ax_res.set_ylabel("Frequency")
    st.pyplot(fig_res)