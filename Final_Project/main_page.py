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
st.markdown("Introduction")
st.write("""
        In this project, we will be looking at a dataset that contains vehicles with their unique features and how they correlate with their price. 
        Hopefully by the end of this, we will be able to estimate a car's price just from picking out the desired feature inputs. 
        First, let's look at which data we will be using.""") #brief introduction

st.markdown('Select a Dataset to View') #title for the button selection

dataset = st.radio("Choose a dataset:", ('Ford Vehicles', 'Vehicle Price Prediction', 'Car Price', 'Merged')) #setting up a button selection for each dataset

if dataset == 'Ford Vehicles':
    st.markdown("Ford Vehicle Dataset")
    st.dataframe(ford_df.head())
    #introducing the dataset and showing the first 5 rows
    st.write("""
             - A Ford Dataset that contains all the available vehicles from the Ford company.
             """) #brief explanation of the dataset

elif dataset == 'Vehicle Price Prediction':
    st.markdown("Vehicle Price Prediction Dataset")
    st.dataframe(vehicle_df.head())#introducing the dataset and showing the first 5 rows
    st.write("""
             - A very large dataset that contains engine horsepower which is something important in price.
             - To be able to use it in the app without any errors, the sample size is reduced.
             """) #brief explanation of the dataset


elif dataset == 'Car Price':
    st.markdown("Car Price Dataset")
    st.dataframe(car_df.head())
    #introducing the dataset and showing the first 5 rows
    st.write("""
             - Another Price Prediction Dataset to add more diversity to our dataset.
             """) #brief explanation of the dataset

elif dataset == 'Merged':
    
    st.markdown("Merged Dataset")
    st.dataframe(df_new.head())
    #for the left column, the data is introduced and then the first five rows is shown.
    st.dataframe(df_new.info()) #showing the initial info for the merged dataset
    st.write("""
    - Three dataset merged together using the vehicle price prediction columns.
    - Cleaned the dataset.
    - Cars with same model, year, and some other variables were kept assuming their price changes across the dataset.
    - Added a binary Transmission Class; 0 for Manual and 1 for Automatic.
    - Some columns are converted into a float for visualization.""")
    #for the right columns, a brief explanation of what has changed with merging.





