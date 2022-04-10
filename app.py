import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#from validators import Max


# Make containers:

header = st.container()
data_sets = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title("Kashti ki app")
    st.text("In this project we will work  on kashti data")
with data_sets:
    st.header("Kashti doob gayi, opzz!!")
    st.text("We will work Titanic dataset")
    #Import dataset
    df = sns.load_dataset('titanic')
    df = df.dropna()
    st.write(df.head(10))

    st.subheader("How many males and females are there?")
    st.bar_chart(df['sex'].value_counts())
    # Other plots
    st.subheader("Class ky hisab sy farq")
    st.bar_chart(df['class'].value_counts())

    # bar plot
    st.bar_chart(df['age'].sample(10)) # also can use as head(10)



with features:
    st.header("These are our app features")
    st.text("Awein buhat sary features add karty hain, asaan hi hy")
    st.markdown('1. **Feature 1:** This will highlight as')
    st.markdown('1. **Feature 2:** This will highlight as')
    st.markdown('1. **Feature 3:** This will highlight as')



with model_training:
    st.header("Kashti walon ka kia bana- Model training")
    st.text("We will increase or decrease our parameters in this section")
    # Making columns
    features, display = st.columns(2)

    # In first column: we will have selection points

    max_depth = st.slider("How many people do you know?", min_value=10, max_value=100, value=20, step=5)

# For using random forest method:
# n-estimators
n_estimators = st.selectbox("How many trees should be there in Random Forest Method? ", 
options=[50, 100, 200, 300, 'No limit'])

# Adding list of features
st.write(df.columns)


# Input feature from user

input_features = st.text_input("Which feature we should use")

# Applying machine learning model

model = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)

# Here we are going to put a condition
# if n_estimators == 'No limit':
#     random_r = RandomForestRegressor(max_depth = max_depth)
# else:
#     random_r = RandomForestRegressor(max_depth = max_depth, n_estimators = n_estimators)


# define X and y
X = df[[input_features]]
y = df[['fare']]
model.fit(X,y)
pred = model.predict(y)

# Display metrices
display.subheader("Mean absolute error of the model is: ")
display.write(mean_absolute_error(y,pred))

display.subheader("Mean squared error of the model is: ")
display.write(mean_squared_error(y,pred))

display.subheader("R2 score of the model is: ")
display.write(r2_score(y,pred))
