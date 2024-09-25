import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf
import keras
from tensorflow.keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = dt.datetime.now()

st.title("Stock Price Prediction")

user_input = st.text_input('Enter Stock Ticker' , 'AAPL')
df = yf.download('user_input',start,end )
df.head()

st.subheader('Data From 2010 to 2024')
st.write(df.describe())


st.subheader('Closing Price VS Time Chart with 100MA & 200MA')
ma100 =df.close.rolling(100).mean()
fig = plt.figure(figsize=(12,6)).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma100,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)


data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_train)

#splitting data inyo x_train and y_train


#load my model

model=load_model('keras_model.h5')



past_100_days = data_train.tail(100)

final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

    x_test , y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)
    scaler=scaler.scale_

    scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final graph
st.subheader('predictions vs original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b' , label = "Original Price")
plt.plot(y_predicted, 'r' , label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)