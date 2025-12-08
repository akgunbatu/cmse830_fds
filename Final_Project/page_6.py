import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge

st.title("Price Estimator") #page title

st.write("""Enter what type of car you want and the model will predict the price range
         for the car you are looking for. This might take a little bit of time to run because building your dream takes time.
""")
#brief introduction to the page
df = pd.read_csv("Final_Project/df_final.csv") #loading the merged dataset

@st.cache_resource #caching the regression models so we don't run it everytime we change our inputs
def regression_model(df): #defining the regression function to model it
        
    df_encoded = pd.get_dummies(df, columns=['fuel_type'], drop_first=True) #to include fuel types into our model we one-hot encode the fuel types 

    X = df_encoded.drop(columns=['price','drivetrain','body_type', 'transmission','model', 'make'])  #in our new dataset we take our numerical columns for the values to observe in price
    y = df_encoded['price'] #for our target column we take the price since we are trying to predict the price

    scaler = StandardScaler() #creating a scaler
    X_scaled = scaler.fit_transform(X) # scaling the X

    lasso = Lasso(alpha=0.1) #creating the lasso regression with 0.1 alpha
    lasso.fit(X_scaled, y) #training the data to fit the Lasso model

    ridge = Ridge(alpha=1) #creating the ridge regression with 1 alpha
    ridge.fit(X_scaled, y) #predicting the model using the test data

    return scaler, lasso, ridge, list(X.columns) #Return the scaler, lasso model, ridge model and list of x feature names so we can compare it later

scaler, lasso, ridge, model_features = regression_model(df) #calling the function

st.subheader("Vehicle Inputs") #title for Inputs


fuel = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique())) #Selecting the fuel type
st.write("For Transmission please select 0 for Manual and 1 for Automatic") #remainder on what binary number represents for transmission
trans_class = st.selectbox("Transmission Class", sorted(df["Transmission Class"].unique())) #selecting the transmission class
year = st.number_input("Year", 1980, 2025, 2018) #picking the year
mileage = st.number_input("Mileage", 0, 400000, 60000) #picking the mileage
engine_hp = st.number_input("Engine HP", 50, 1200, 250) #picking the engine horsepower
owner_count = st.number_input("Owner Count", 0, 10, 1) #picking the owner count
vehicle_age = st.number_input("Vehicle Age", 0, 40, 5) #picking the vehicle age
brand_popularity = st.number_input("Brand Popularity",min_value=0.039318,max_value=0.040484,value=0.039800,step=0.00001) #picking the brand popularity

input_dict = {"year": year,
              "mileage": mileage,
              "engine_hp": engine_hp,
              "owner_count": owner_count,
              "vehicle_age": vehicle_age,
              "brand_popularity": brand_popularity,
              "fuel_type": fuel,
              "Transmission Class": trans_class} #create a dictionary for all the inputs

input_df = pd.DataFrame([input_dict]) #creating a dataframe from the dictionary

input_encoded = pd.get_dummies(input_df, columns=['fuel_type']) #one-hot encoding the fuel type

for i in model_features: #iterating through the X columns list
    if i not in input_encoded: #if the column name is not in the input dataframe 
        input_encoded[i] = 0 #then it creates the column and sets the value to zero
#this makes the dataset to have all the features that we are looking for

input_encoded = input_encoded[model_features] #reorders the columns so we don't run into issues in our regression models
#for rearranging and ensuring we all have the same columns as the regression model function, I asked chatgpt and it gave me the idea
#OpenAI. (2025). ChatGPT (Dec 1 version) [Large language model]. https://chatgpt.com/share/69363bad-6904-8009-a8cd-e21d7f804304

if st.button("Estimate Price"):

    input_scaled = scaler.transform(input_encoded) #scaling the input dataframe

    lasso_pred = lasso.predict(input_scaled)[0] #retrieving the lasso price prediction
    ridge_pred = ridge.predict(input_scaled)[0] #retrieving the ridge price prediction


    st.subheader("Predicted Prices") #title for predictions

    st.write(f"""
    ### **Lasso Regression Estimate:**  
    Value: {lasso_pred:,.0f}\n

    ### **Ridge Regression Estimate:**  
    Value: {ridge_pred:,.0f}\n
    """) #displaying the predictions for each regression model

    st.subheader("Comparison") #title for comparison
    diff = abs(lasso_pred - ridge_pred) #taking the absolute value of the difference
    st.write(f"Difference: **${diff:,.2f}**") #displaying the difference

    if lasso_pred > ridge_pred and diff > 1:
        st.success("Lasso predicts a higher price.") #if the lasso prediction is higher display message
    elif ridge_pred > lasso_pred and diff > 1:
        st.success("Ridge predicts a higher price.") #if the ridge prediction is higher display message
    else:
        st.info("Both models predict the same value.") #if they are the same valeu display message

    st.write("""
            Obviously this model has some issues with getting the correct price range. 
            This is due to the fact that it requires a lot more data points to understand how each feature works, 
            and it also requires much more diversity using other brands. We also have not included the model or make into our dataset 
            because of the same reasons (there are no large variety) and it also takes a very long time to create the model. 
            For an initial model estimation, however, it is a pretty good estimation.""")
    #conclusion on the real-life example