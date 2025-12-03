import streamlit as st
import pandas as pd
import numpy as np



st.set_page_config(page_title="Data Introduction", layout="wide") 


st.sidebar.title("Data Introduction") #setting up a title for the side column
st.sidebar.info("Overview of both datasets, and merged datasets cleaned.") # A quick summary of what to expect from the page


st.title("Vehicle Price Prediction — Data Introduction") #title of the page

#loading the necessary datasets
ford_df = pd.read_csv("Final_Project/ford.csv")
vehicle_df = pd.read_csv("Final_Project/vehicle_price_prediction.zip")
car_df = pd.read_csv("Final_Project/car_price_prediction_.csv")
df_new = pd.read_csv("Final_Project/df_final_na.csv")
df_new = df_new.drop(columns = "Unnamed: 0") #dropping the first unknown column

st.markdown('Select a Dataset to View') #title for the button selection

dataset = st.radio("Choose a dataset:", ('Ford Vehicles', 'Vehicle Price Prediction', 'Car Price', 'Merged')) #setting up a button selection for each dataset

if dataset == 'Ford Vehicles':
    st.markdown("Ford Vehicle Dataset")
    st.dataframe(ford_df.head())
    #introducing the dataset and showing the first 5 rows

elif dataset == 'Vehicle Price Prediction':
    st.markdown("Vehicle Price Prediction Dataset")
    st.dataframe(vehicle_df.head())
    #introducing the dataset and showing the first 5 rows

elif dataset == 'Car Price':
    st.markdown("Car Price Dataset")
    st.dataframe(car_df.head())
    #introducing the dataset and showing the first 5 rows

elif dataset == 'Merged':
    col1, col2 = st.columns([2, 1]) #splitting the page into two columns
    with col1:
        st.markdown("Merged Dataset")
        st.dataframe(df_new.head())
        #for the left column, the data is introduced and then the first five rows is shown.
    with col2:
        st.write("""
        - Two dataset merged together using the vehicle price prediction columns.
        - Cleaned and removed duplicates.
        - Removed the cars with same model and same year assuming the price does not change.
        - Added a binary Transmission Class; 0 for Manual and 1 for Automatic.
        - Some columns are converted into a float for visualization""")
        #for the right columns, a brief explanation of what has changed with merging.





