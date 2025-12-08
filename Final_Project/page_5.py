import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

st.title("Lasso vs Ridge Regression") #title for the page

df =  pd.read_csv("Final_Project/df_final.csv") #loading the merged dataset


st.write("""
        For finding how each feature affects the car price, Lasso and Ridge regression is chosen 
        to compare how both model will be able to predict. To be able to include the fuel types into our model,
        the column was one-hot encoded, which converts the strings into binary matrices for each type. 
        Now, let us see how both model will do!
        """)
#brief explanation of what we are looking for in the both regressions


df_encoded = pd.get_dummies(df,columns=['fuel_type'],drop_first=True)  #to include fuel types into our model we one-hot encode the fuel types 
#for the above code, I used the following pages; https://www.programiz.com/python-programming/pandas/methods/get_dummies
# https://www.geeksforgeeks.org/pandas/python-pandas-get_dummies-method/
 
X = df_encoded.drop(columns=['price','drivetrain','body_type', 'transmission','model','make'])  #in our new dataset we take our numerical columns
y = df_encoded['price'] #for our target column we take the prcie since we are trying to predict the price


model_choice = st.radio("Choose Regression Model:",["Lasso Regression", "Ridge Regression"]) #creating a button feature to choose between the regressions

if model_choice == "Lasso Regression":
    alpha = st.slider("Lasso Alpha (λ)", 0.01, 10.0, 0.1, 0.01) #creating a slider for lasso regression
else:
    alpha = st.slider("Ridge Alpha (λ)", 0.01, 10.0, 1.0, 0.01) #creating a slider for ridge regression

if st.button("Run Regression Model"): #creating a button for running the model

    scaler = StandardScaler() #creating a scaler
    X_scaled = scaler.fit_transform(X) # scaling the X

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #splitting the data for training and testing (80/20). 

    if model_choice == "Lasso Regression":
        model = Lasso(alpha=alpha) #using the specific alpha for creating lasso regression
        model.fit(X_train, y_train) #training the data to fit the Lasso model
        pred = model.predict(X_test) #predicting the model using the test data
    else:
        model = Ridge(alpha=alpha) #using the specific alpha for creating ridge regression
        model.fit(X_train, y_train) #training the data to fit the Ridge model
        pred = model.predict(X_test) #predicting the model using the test data
    #for the regressions I used this website, https://www.datacamp.com/tutorial/tutorial-lasso-ridge-regression
    #I also used an example from the dataset's example projects as well, https://www.kaggle.com/code/mandyreyes/automotive-price-prediction-lr


    mse = mean_squared_error(y_test, pred) #calling a function to calculate MSE
    rmse = np.sqrt(mse) #taking the square root of MSE to find RMSE
    r2 = r2_score(y_test, pred) #calling a function to calculate R-squared
    residuals = y_test - pred #calculating the residuals using the original and model points

    st.subheader(f"Results: {model_choice}") #displaying the results

    col1, col2, col3 = st.columns(3) #dividing into 3 columns
    col1.metric("Mean-Squared Error", f"{mse:,.2f}") #displaying the MSE
    col2.metric("Root Mean-Squared Error", f"{rmse:,.2f}") #displaying the RMSE
    col3.metric("R² Score", f"{r2:.4f}") #displaying the R-squared
    #used metric using the following website, https://docs.streamlit.io/develop/api-reference/data/st.metric
    
    st.write("### Model Coefficients")
    coef_df = pd.DataFrame({"Feature": X.columns,"Coefficient": model.coef_})
    #Chatgpt gave me the idea to use the dataframe function, here is the prompt: "to display a list of names with their corresponding coefficients, what would be a cleaner way in streamlit. give me an example too"
    #OpenAI. (2025). ChatGPT (Dec 1 version) [Large language model]. https://chatgpt.com/share/69362185-99ec-8009-ab59-f896227640f5

    st.dataframe(coef_df) #displaying the coefficients

    st.write("Actual vs Predicted")
    
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, alpha=0.5) #creating a scatter plot for predicted and original prices
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--") #creating a dashed ideal line for the original price
    sns.regplot(x=y_test, y=pred, ax = ax, scatter_kws={'alpha':0.6}, line_kws={'color':'red'}) #creating a regression line for the predicted price
    ax.set_xlabel("Actual Price") #title for the x-axis
    ax.set_ylabel("Predicted Price") #title for the y-axis
    ax.set_title('Actual vs Predicted Prices with Regression Line') #title for the plot
    st.pyplot(fig) #displaying the plot

    
    st.write("Residual Distribution") 

    fig_res, ax_res = plt.subplots()
    ax_res.hist(residuals, bins=30, alpha=0.7) #creating a histogram for the residuals
    ax_res.set_title("Residual Distribution") #title for the histogram
    ax_res.set_xlabel("Residual") #title for the x-axis
    ax_res.set_ylabel("Frequency") #title for the y-axis
    st.pyplot(fig_res) #displaying the histogram

    st.write("""
            - Looking at both Lasso and Ridge Regression models, we can observe that they are almost identical.
            - The extremely high MSE is due to using dollars that are in the range of tens of thousands and we are taking the square of that.
            - The R-squared for both model is very high, meaning that both our model have a very high accuracy in predicting the data.
            - From the scatter plot with regresion lines, we can see that it is not exactly a perfect fit but it is close to be one.
            - This means in the future we might have to add other categorical columns like the Make.
            - Residual distribution for both plot shows that it centers around zero, which means the model does not over or under-predicts.
            - Overall both model is a really good way of predicting the price.""")
    #bullet points of what we observe from the regressions