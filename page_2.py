import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df_new = pd.read_csv("df_new")

st.subheader("Missing Values Overview")
missing_counts = df_new.isnull().sum()
st.write(missing_counts[missing_counts > 0])


st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(df_new.isnull(), cbar=False, yticklabels=False, cmap='viridis')
ax.set_title("Missing Values in Dataset")
st.pyplot(fig)


st.subheader("KNN Imputation on Numeric Columns")

numeric_columns = ['engine_hp', 'owner_count', 'vehicle_age', 'brand_popularity']
df_numeric = df_new[numeric_columns].copy()

# Split into sets with and without missing values
df_with_missing = df_numeric[df_numeric.isnull().any(axis=1)]
df_without_missing = df_numeric.dropna()

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_without_missing), columns=df_without_missing.columns)

# Initialize and fit KNN imputer
imputer = KNNImputer(n_neighbors=5)
imputer.fit(df_scaled)

def impute_and_inverse_transform(data):
    """Scales data, imputes missing values, and inversely transforms it back."""
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    imputed_scaled = imputer.transform(scaled_data)
    return pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=data.columns, index=data.index)


df_imputed = impute_and_inverse_transform(df_numeric)


st.subheader("Example: Distribution Before vs After Imputation (Engine HP)")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(df_numeric['engine_hp'].dropna(), kde=True, color='blue', alpha=0.5, label='Original (non-missing)', ax=ax2)
sns.histplot(df_imputed.loc[df_numeric['engine_hp'].isnull(), 'engine_hp'], kde=True, color='red', alpha=0.5, label='Imputed', ax=ax2)
ax2.set_title("Distribution of Original vs Imputed Engine HP")
ax2.legend()
st.pyplot(fig2)


st.subheader("Statistical Comparison")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Original Engine HP Stats**")
    st.write(df_numeric['engine_hp'].describe())

with col2:
    st.markdown("**Imputed Engine HP Stats**")
    st.write(df_imputed['engine_hp'].describe())


df_new[numeric_columns] = df_imputed
df_new.to_csv("df_new1")
