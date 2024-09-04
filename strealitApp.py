import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import streamlit as st
model =joblib.load("liveModelV1.pkl")
data= pd.read_csv("mobile_price_range_data.csv")
#splitting the data into features and labels
x= data.iloc[:,:-1]
y=data.iloc[:,-1]
#splitting the data into testing and training set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#make prediction for x_test
y_pred=model.predict(x_test)
#calculate accuracy
accuracy= accuracy_score(y_test,y_pred)
#page title
st.title("Model Accuracy and Real-Time Prediction")
#display accuracy
st.write(f"Model{accuracy}")
#real time prediction on users inputs
st.header("Real-Time Prediction")
input_data=[]
for col in x_test.columns:
    input_value=st.number_input(f'Input for feature{col}',value=0)
    input_data.append(input_value)
#convert input data into dataframe
input_df=pd.DataFrame([input_data],columns=x_test.columns)
#Make Prediction
if st.button("Predict"):
    prediction =model.predict(input_df)
    st.write(f"prediction:{prediction[0]}")

