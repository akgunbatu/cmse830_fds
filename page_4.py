import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="EDA Visualization", layout="wide")
st.title("Exploratory Data Analysis (EDA)")
st.sidebar.title("EDA Visualization")

df_new = pd.read_csv("df_new1")

st.subheader("Engine Horsepower vs Price (Matplotlib)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df_new['engine_hp'], df_new['price'], alpha=0.5, color='teal')
ax.set_xlabel("Engine Horsepower (HP)")
ax.set_ylabel("Price ($)")
ax.set_title("Engine HP vs Price")
st.pyplot(fig)


st.subheader("Distribution of Engine Horsepower")
fig_hp = px.histogram(df_new, x="engine_hp", nbins=40, color_discrete_sequence=["#1f77b4"])
fig_hp.update_layout(width=1000, height=600, bargap=0.1)
st.plotly_chart(fig_hp, use_container_width=True)


st.subheader("Price Distribution by Fuel Type")
fig_price = px.histogram(df_new, x='price', color='fuel_type', nbins=40)
fig_price.update_layout(
    xaxis=dict(range=[0, 2000000]),
    width=1000,
    height=600,
    bargap=0.1
)
st.plotly_chart(fig_price, use_container_width=True)


st.subheader("3D Relationship Between Year, Mileage, and Price")

fig_3d = px.scatter_3d(
    df_new,
    x='year',
    y='mileage',
    z='price',
    color='drivetrain' if 'drivetrain' in df_new.columns else None,
    title="3D Scatter: Year vs Mileage vs Price",
    opacity=0.7,
    width=1000,
    height=700
)
fig_3d.update_layout(scene=dict(
    xaxis=dict(range=[1995,2025]),
    yaxis=dict(range=[0, 200000]),  
    zaxis=dict(range=[0, 2000000])))
st.plotly_chart(fig_3d, use_container_width=True)