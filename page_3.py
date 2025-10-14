import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Correlation Analysis", layout="wide")
st.title("Correlation Analysis")
st.sidebar.title("Correlations")
st.sidebar.info("Explore how features relate to vehicle price and each other.")


df_new = pd.read_csv("df_new1")


numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()


st.subheader("Full Dataset Correlation Matrix")

fig, ax = plt.subplots(figsize=(10, 6))
corr_matrix = df_new[numeric_cols].corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap="viridis", vmin=-1, vmax=1, ax=ax)
ax.set_title("Full Dataset Correlation Matrix")
st.pyplot(fig)


st.subheader("Correlations with Price")
price_corr = corr_matrix['price'].sort_values(ascending=False)
st.write(price_corr)
