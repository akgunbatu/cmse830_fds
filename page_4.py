import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(page_title="EDA Visualization", layout="wide")
st.title("Exploratory Data Analysis (EDA)") #title of the page

st.sidebar.title("EDA Visualization") #sidebar title

df_new = pd.read_csv("df_new.csv") #loading the dataset


min_price = int(df_new['price'].min()) #defining the minimum value for price
max_price = 100000 #defining the maximum value for price

price_range = st.slider(
    'Select a Price Range:',
    min_price, max_price,
    value=(min_price, 50000),
    step=1000
) #creating a slider for price for user to pick its own range of price using the minimum and maximu values


df_filtered = df_new[(df_new["price"] >= price_range[0]) & (df_new["price"] <= price_range[1])] #creating a new dataset using the selected price range
#for this line I used ChatGPT to understand how I can intergrate the slider to the dataset 


color_pick = st.selectbox("Select variable for hue:", ["fuel_type", "drivetrain", "Transmission Class"]) #setting up a select box to use for the color part of the plots

numeric_axes = ["price", "engine_hp", "year", "vehicle_age", "mileage"] #creating a new list of columns
x_axis = st.selectbox("Select X-axis:", numeric_axes, index=0) #setting up a x-axis select box from the list of columns for the plot, default is price
y_axis = st.selectbox("Select Y-axis:", numeric_axes, index=1) #setting up a y-axis select box from the list of columns for the plot, default is engine_hp
z_axis = st.selectbox("Select Z-axis:", numeric_axes, index=3) #setting up a z-axis select box from the list of columns for the plot, default is vehicle_age


st.subheader("Interactive Scatter Plot with Regression Line") 


fig_scatter = px.scatter(df_filtered,x=x_axis,y=y_axis,color=color_pick,
    opacity=0.7,trendline="ols",trendline_color_override="red",
    title=f"{y_axis} vs {x_axis}"
) # Creating a scatter plot from the x and y axes and color is the color pick from previous selection with a regression line (built-in function in pyplot)

fig_scatter.update_layout(width=1000,height=600,
    xaxis_title=x_axis,yaxis_title=y_axis
) #updating the width, height, titles and ranges of the plot

st.plotly_chart(fig_scatter, use_container_width=True)


st.subheader(f"Distribution by {color_pick}")
fig_price = px.histogram(df_filtered, x=x_axis, color=color_pick, nbins=40) #setting up a histogram for the x axis and the color picked for distribution visualization
fig_price.update_layout(
    width=1000,height=600,bargap=0.1
) #updating the height and width
st.plotly_chart(fig_price, use_container_width=True)



st.subheader("3D Relationship Between Selected Variables")


fig_3d = px.scatter_3d(df_filtered,x=x_axis,y=y_axis,z=z_axis,color=color_pick,title="3D Scatter Plot",
    opacity=0.7,width=1000,height=700
) #creating a 3d plot with the x, y, z-axis and the color picked (height and width are also defined here)


st.plotly_chart(fig_3d, use_container_width=True)

