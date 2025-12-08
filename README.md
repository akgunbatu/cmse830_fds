# cmse830_fds
Introduction

**This project aims to observe the vehicle features and how it affects the price. It performs imputation to get rid of missing values and initial EDA to explore the features that influences the price. It creates a Lasso and Ridge Regression models and applies it into a real-world estimation using the Streamlit web app. For our web app, we use 5 different dataset. To access the Final Project files, click on the Final_Project folder. From 2 of those is our merged dataset with and without missing values. If you want to see or access the merged dataset and the imputated dataset, you can open the Final Project Jupyter notebook. For loking at which packages I used, you can look at the requirements.txt or again the Jupyter notebook.**

Why you chose your dataset

**I chose this data because I wanted to have a data that I could have related to as a mechanical engineer. I love cars and I always wish to have more money to buy cars. However, I have a budget just like anyone else. I know that there are other car enthusiast that want to get a car but looking for a specific price range. That is why I picked this data, to actaullay enjoy my time while doing this project and also I could feel like I am helping other like me to observe cars with affordable price range. I switcehd one of my dataset for my Final Project because that dataset was messing the imputations and making the correlations decrease a lot. I added a Ford vehicle dataset, to increase the diversity on vehicle brands.**


What you've learned from IDA/EDA

**From the IDA, I saw that the some columns were not helpful for our price observations like he exterior color. So I have dropped some columns to keep the data neat. While looking at duplicates I realized that there were some cars with the same model and same year but with different prices. I did not know if I needed to drop the duplicates or keep them. After looking more carefully, I realized that the price did not change significantly, so I decided to drop it to observe a much smaller sample size with the still the same correlations and everything. From the correlations I learned that some variables that I thought would be a good variable did not affect the price at all like the brand popularity. For the EDA analysis, scatterplot, histogram, boxplot, correlation heatmap and a 3D scatter plot were created. From the visualizations of EDA, I learned that engine horsepower actually increases the price, so the faster the car reaches 0 to 100 mph, the more expensive it gets. Also the younger the car is the more expensive it also gets. The brand popularity has no affect on the price. The mileage has a much lower influence on price than initially predicted.**

Lasso and Ridge Regression

**For our regression models, Lasso and Ridge were chosen because of their simplicity and ability to predict accurately. Additionally, the fuel types were one-hot encoded to be included in the model itself. Other categorical columns like brand were not included because the app ran into an issue couple of times. For the fuel types, the diesel was dropped to avoid multicolliniearity. For both models, the R-squared and MSE was calculated to compare the both model along with the model coefficients. It showed that the both models had a real similar accuracy, which was around 87%. The coefficients slightly changed, but it was not drastical. Scatter plot with regression line was created to compare the orginal line with the model line, and it was not a perfect fit for both models. Also, residual distribution showed that it was a good estimation for both models. For the future, we can add the other categorical columns as well.**

Real-World Example

**We estimated the price of a vehicle with some feature inputs like year, mileage, engine horsepower, etc. Both Lasso and Ridge models predict values and these values were compared later. For this Estimator page in Streamlit, I have used the caching method. This way we don't train and test our data for lasso and ridge model each time we change our inputs. It was somewhat good prediction. Assuming we didn't include some columns like make, this prediction is quite good.**




