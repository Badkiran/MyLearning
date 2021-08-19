# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 21:51:53 2021

@author: Kiran B
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.arima_model import ARIMA
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU
import tensorflow.keras.backend as K

from sklearn.preprocessing import MinMaxScaler


from sklearn.metrics import r2_score
def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))




st.title('Forecasting USD-INR Timeseries Data')

st.write("Data Exploration and Forecasting the data using ARIMA, LSTM and GRU Models")

image = Image.open('USDvsINR.jpg')
st.image(image, caption='USD vs. INR')


DATA_URL = ('C:/Users/kiran/Desktop/Project1_Spyder/Exchange_rate.csv')

@st.cache
def load_data():
    data = pd.read_csv(DATA_URL)
    data['Date'] = pd.to_datetime(data.Date)
    # Imputing the null values of the Rate column
    data['Rate'] = data['Rate'].fillna(value= 0.0)

    for i in range(0, len(data['Rate'])):
        if data.Rate[i] == 0.0:
            data.Rate[i] = (data.Rate[i-1] + data.Rate[i-2])/2.0
    return data


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')

data = load_data()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

# Display the Processed data
st.subheader('Processed data')
st.write(data)

# Data Visualization 
st.subheader('Visualizing our data')
# Histogram
st.subheader('Histogram')
# fig, ax = plt.subplots()
# ax.hist(data['Rate'], bins=10)
# st.pyplot(fig)
hist_values = np.histogram(data['Rate'], bins=10)[0]
st.bar_chart(hist_values)


# Line Plot
st.subheader('Line Plot')
st.line_chart(data=data['Rate'], width=0, height=0, use_container_width=True)


# Models

def ARIMA_Model(data,p,d,q):
    
    val = data.values
    val = val.astype('float32')
    train, test = val[1:len(val)-30], val[len(val)-30:]
    history = [x for x in train]
    predictions = list()
    
    for i in range(len(test)):
        arima_model = ARIMA(history, order=(p,d,q)).fit()
        output = arima_model.forecast()
        ypred = output[0]
        predictions.append(ypred)
        act = test[i]
        history.append(act)
        print('predicted=%f, expected=%f' % (ypred, act))
       
    # evaluate forecasts
    residuals = test - predictions 
    mse_arima = mean_squared_error(test, predictions) 
    rmse_arima = sqrt(mse_arima)
    st.write('Root Mean Squared Error: %.5f' %rmse_arima)
    mape_arima = np.mean(abs(residuals/test))
    st.write('Mean Absolute Percentage Error: %.5f' %mape_arima)
    st.write('The R2 score on the Test set is:\t{:0.5f}'.format(r2_score(test, predictions)))
    
    #Print summary of the model
    summary_arima = arima_model.summary()
    st.write(summary_arima)
    
    #Plot forecasts against actual outcomes
    forecasted1 = arima_model.forecast(steps=30,alpha=0.05)[0]
    st.write(forecasted1)
    forecasted2 = arima_model.plot_predict()
    st.write(forecasted2)
    forecasted3 = arima_model.plot_predict(12619,12649)
    st.write(forecasted3)
    
    
def LSTM_Model(data):
    
    data = np.array(data).reshape(-1,1)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(data)
    
    #Training and test sets
    train = df[:11919]
    test = df[11919:]
    
    def get_data(data, look_back):
        data_x, data_y = [],[]
        for i in range(len(data)-look_back-1):
            data_x.append(data[i:(i+look_back),0])
            data_y.append(data[i+look_back,0])
        return np.array(data_x) , np.array(data_y)

    look_back = 1

    x_train , y_train = get_data(train, look_back)
    x_test , y_test = get_data(test,look_back)
    
    #Processing train and test sets for LSTM model
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1],1)
    
    #Defining the LSTM model
    K.clear_session()
    lstm_model=Sequential()
    lstm_model.add(LSTM(100,activation='relu',input_shape=(1,x_train.shape[1]), kernel_initializer='lecun_uniform', return_sequences=False))
    lstm_model.add(Dense(1))

    #Model summary
    # summary_lstm = lstm_model.summary()
    # st.write(summary_lstm)
    
    #Compiling
    lstm_model.compile(optimizer='adam', loss = 'mse', metrics=[tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')])

    #Training
    lstm_model.fit(x_train,y_train, epochs = 5, batch_size=1)
    
    #Prediction using the trained model
    scaler.scale_

    y_pred = lstm_model.predict(x_test)
    y_pred = scaler.inverse_transform(y_pred)
    
    #Processing test shape
    y_test = np.array(y_test).reshape(-1,1)
    y_test = scaler.inverse_transform(y_test)
    
    mse_lstm = mean_squared_error(y_test, y_pred)
    rmse_lstm = sqrt(mse_lstm)
    
    st.write('Root Mean Squared Error: %0.5f' %rmse_lstm)
    st.write("The R2 score on the Test set is: {:0.5f}".format(r2_score(y_test, y_pred)))
    r2_test = r2_score(y_test, y_pred)
    st.write("The Adjusted R2 score on the Test set is: {:0.5f}".format(adj_r2_score(r2_test, x_test.shape[0], x_test.shape[1])))
    
    
    return (y_test,y_pred)


def GRU_Model(data):
    
    data = np.array(data).reshape(-1,1)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(data)
    
    #Training and test sets
    train = df[:11919]
    test = df[11919:]
      
    def get_data(data, look_back):
        data_x, data_y = [],[]
        for i in range(len(data)-look_back-1):
            data_x.append(data[i:(i+look_back),0])
            data_y.append(data[i+look_back,0])
        return np.array(data_x) , np.array(data_y)

    look_back = 1

    X_train , y_train = get_data(train, look_back)
    X_test , y_test = get_data(test,look_back)
    
    #Processing train and test sets for LSTM model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)
    
    K.clear_session()
    gru_model = Sequential()
    gru_model.add(GRU(12, input_shape=(1, X_train.shape[1]), activation='linear', kernel_initializer='lecun_uniform', return_sequences=False))
    gru_model.add(Dense(1))
    early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)
    gru_model.compile(loss='mse', optimizer= 'adam', metrics=[tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')])
    gru_model.fit(X_train, y_train, epochs=100, batch_size=20, verbose=1, shuffle=False,callbacks=[early_stop])
    
    #Model Summary
    summary_gru = gru_model.summary()
    st.write(summary_gru)
    
    #Prediction
    y_pred = gru_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = np.array(y_test).reshape(-1,1)
    y_test = scaler.inverse_transform(y_test)
    
    mse_gru = mean_squared_error(y_test, y_pred)
    rmse_gru = sqrt(mse_gru)
    
    st.write("Root Mean Squared Error: %.5f" %rmse_gru)
    st.write("The R2 score on the Test set is: {:0.5f}".format(r2_score(y_test, y_pred)))
    r2_test = r2_score(y_test, y_pred)
    st.write("The Adjusted R2 score on the Test set is: {:0.5f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
    
    return y_test,y_pred


def NeuralNet_Plot(y_test,y_pred):
    
    fig1 = plt.figure(figsize=(12,8))
    fig2 = plt.title('USD-INR Prediction')
    fig3 = plt.plot(y_test , label = 'Actual')
    fig4 = plt.plot(y_pred , label = 'Predicted')
    fig5 = plt.xlabel('Observation')
    fig6 = plt.ylabel('Rate')
    fig7 = plt.legend()
    st.write(fig1,fig2,fig3,fig4,fig5,fig6,fig7)
    

# fig, ax = plt.subplots()
# ax.hist(data, bins=20)
# st.pyplot(fig)

# Model Selection
st.subheader('Model Selection')
# genre = st.radio("Which Model would you like to select, to Forecast 30 days' data",('ARIMA', 'LSTM', 'GRU'))
# if genre == 'ARIMA':
#     st.write('You selected ARIMA Model')
#     ARIMA_Model(data['Rate'],1,1,1)
    
# elif genre == 'LSTM':
#     st.write('You selected LSTM Model(Long Short Term Memory)')
# else:
#     st.write("You selected GNU Model(Gated Neural Network)")

option = st.selectbox("Select a Model you would like to select, to forecast the future 30 days' data",('ARIMA', 'LSTM', 'GRU'))
st.write('You selected:', option)

if option == 'ARIMA':
    image2 = Image.open('ARIMA.jpg')
    st.image(image2, caption='Structure of ARIMA')
    ARIMA_Model(data['Rate'],1,1,0)
    
elif option == 'LSTM':
    image3 = Image.open('LSTM.png')
    st.image(image3, caption='Structure of LSTM')
    y_test2,y_pred2 = LSTM_Model(data['Rate'])
    NeuralNet_Plot(y_test2,y_pred2)
    
else:
    image4 = Image.open('GRU.png')
    st.image(image4, caption='Structure of GRU')
    y_test3,y_pred3 = GRU_Model(data['Rate'])
    NeuralNet_Plot(y_test3,y_pred3)


