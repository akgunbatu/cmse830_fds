import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Correlation Analysis", layout="wide")
st.title("Correlation Analysis")

st.sidebar.title("Correlations")#setting up a title for the side column
st.sidebar.info("Explore how features relate to vehicle price and each other.") # A quick summary of what to expect from the page
st.write("""
        - With a good correlation heatmap, we can observe how strong the relationship among columns.
        - Creating a correlation heatmap will show how price is influenced by other features.""")

df_new = pd.read_csv("Final_Project/df_final.csv") #loading the dataset
df_new = df_new.drop(columns = "Unnamed: 0") #dropping the unknown column from the dataset


numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist() #creating a list of all the numeric columns 


st.subheader("Full Dataset Correlation Matrix") 

fig, ax = plt.subplots(figsize=(10, 6))
corr_matrix = df_new[numeric_cols].corr(numeric_only=True) #creating a correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="viridis", vmin=-1, vmax=1, ax=ax) #setting up a heat map for the correlation matrix for better visualization
ax.set_title("Full Dataset Correlation Matrix") #title for the heatmap
st.pyplot(fig)


st.subheader("Correlations with Price")
price_corr = corr_matrix['price'].sort_values(ascending=False) #creating an descending list for price correlations
st.write(price_corr) #printing the list

st.write("""
- Brand Popularity has the lowest correlation with price of the vehicle.
- Year and mileage does not have a strong correlation with price as it was hoped.
- Vehicle Age, Owner Count and Engine Horsepower have the strongest correlations.
- Next steps will be looking how each of these variables affect price of the vehicle.
""") #A summary of what these correlations mean and how it affects the project.