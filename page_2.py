#importing all the necessary tools
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


df_new = pd.read_csv("df_new.csv") #loading the merged dataset

st.subheader("Missing Values Overview")
missing_counts = df_new.isnull().sum() #counting all the missing values in each column
st.write(missing_counts[missing_counts > 0]) 


st.subheader("Missing Values Heatmap")
fig, ax = plt.subplots(figsize=(10, 5)) #creating a heatmap for the missing values across the columns for better visualization
sns.heatmap(df_new.isnull(), cbar=False, yticklabels=False, cmap='viridis')
ax.set_title("Missing Values in Dataset") #setting the title for plot
st.pyplot(fig) #showing the plot


st.subheader("KNN Imputation on Numeric Columns")
#Brief explanation of why the KNN was used for the dataset
st.write("""
- The columns drivetrain and body type were strings and specific to the unique vehicle, so the imputation was impossible.
- K nearest neighbor is used for other numeric columns.
- KNN is the best choice for this data because it uses similar values from the neighbors to preserve the realistic relationship.
- Dataset is also not big enough which makes it a perfect enviroment for KNN.
""")

# I used the week 6 ICA KNN example for my imputation method. The function and the code are from the ICA I have adjusted them to make them fit my data.

numeric_columns = ['mileage','engine_hp', 'owner_count', 'vehicle_age', 'brand_popularity'] #defining a list of columns for imputation
df_numeric = df_new[numeric_columns] #pulling only those specific columns from the dataset

df_with_missing = df_numeric[df_numeric.isnull().any(axis=1)] #rows with missing values
df_without_missing = df_numeric.dropna() #rows with no missing value

scaler = StandardScaler() #setting up the scaler
df_scaled = pd.DataFrame(scaler.fit_transform(df_without_missing), columns=df_without_missing.columns) #fits the scaled dataset to non-missing data and then transforms it from using what it learned 

imputer = KNNImputer(n_neighbors=5) #setting up the KNN imputer using the 5 nearest neighbors
imputer.fit(df_scaled) #Fitting the imputer to the scaled data

def impute_and_inverse_transform(data):
    
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index) #scale the data and transform using the initial scaler
    imputed_scaled = imputer.transform(scaled_data) #imputting the missing values to the scaled data using the initial imputer
    return pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=data.columns, index=data.index) #returns the inverted imputed data to get back the original data. (converts the z-score)

df_imputed = impute_and_inverse_transform(df_numeric) #using the impute and inverse function for the numeric columns to get rid of the missing values in columns if they have any


st.subheader("Distribution Before vs After Imputation") 

select = st.selectbox("Select a column to view", numeric_columns, index=1) #creating a select box for users to select which numeric column from the list they want to view, defaut is engine_hp


fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.histplot(df_numeric[select].dropna(), kde=True, color='blue', alpha=0.5, label='Original (non-missing)', ax=ax2) #histogram plot for the original data
sns.histplot(df_imputed.loc[df_numeric[select].isnull(), select], kde=True, color='red', alpha=0.5, label='Imputed', ax=ax2) #histogram plot for the imputed data
ax2.set_title("Distribution of Original vs Imputed") #title for the plot
ax2.legend() #showing the legend
st.pyplot(fig2)


st.subheader("Comparison") 
col1, col2 = st.columns(2) #creating two columns for comparison

with col1: #left column is the original data
    st.markdown("**Original Stats**")
    st.write(df_numeric[select].describe())
    #Showing the statistics of the original data

with col2: #right column is the imputed data
    st.markdown("**Imputed Stats**")
    st.write(df_imputed[select].describe())
    #showing the statistics of the imputed data



