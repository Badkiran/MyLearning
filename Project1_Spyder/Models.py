# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:49:06 2021

@author: Kiran B
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from math import sqrt
from time import time
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pmd

from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing

from fbprophet import Prophet 
import plotly.graph_objs as go
import plotly.offline as py
from fbprophet.plot import plot_plotly

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


#Auto Regressive Model
def AR_Model(data):
    series = data
    val = series.values
    train, test = val[1:len(val)-30], val[len(val)-30:]
    
    #Create Model
    start = time()
    ar_model = AutoReg(train, lags=2).fit()
    end = time()
    print("Model fitting time:", end-start)
    print('Coefficients: %s' % ar_model.params)
    
    #Print summary of the model
    print(ar_model.summary())
    
    #Get the predictions
    predictions = ar_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
    residuals = test - predictions
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test[i]))
    rmse = np.sqrt(np.mean(residuals**2))
    print('Root Mean Squared Error: %.3f' %rmse)
    mape = round(np.mean(abs(residuals/test)),4)
    print('Mean Absolute Percentage Error:', mape)
    
    # plot results
    plt.figure(figsize=(10,6))
    plt.plot(test)
    plt.plot(predictions)
    
    plt.legend(('Actual','Prediction'), fontsize=16)
    plt.title('USD-INR Rate Change over time', fontsize=20)
    plt.ylabel('Rate', fontsize=16)
    

def ARMA_Model(data):
       
    val = data.Rate.values
    train_data, test_data =  val[1:len(val)-30], val[len(val)-30:]
    
    #Create Model
    start = time()
    arma_model = ARMA(train_data, order=(3,2)).fit()
    end = time()
    print("Model fitting time:", end-start)
    print('Coefficients: %s' % arma_model.params)
    
    #Print summary of the model
    print(arma_model.summary())
    
   
    #Get the predictions and residuals
    predictions = arma_model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, dynamic=False)
    residuals = test_data - predictions
    for i in range(len(predictions)):
        print('predicted=%f, expected=%f' % (predictions[i], test_data[i]))
    
    rmse = np.sqrt(np.mean(residuals**2))
    print('Root Mean Squared Error: %.3f' %rmse)
    mape = round(np.mean(abs(residuals/test_data)),4)
    print('Mean Absolute Percentage Error:', mape)
    
    #Plot results
    plt.figure(figsize=(10,6))
    plt.plot(test_data)
    plt.plot(predictions)
    
    plt.legend(('Actual','Prediction'), fontsize=16)
    plt.title('USD-INR Rate Change over time', fontsize=20)
    plt.ylabel('Rate', fontsize=16)
  

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
    # rmse = np.sqrt(np.mean(residuals**2))
    mse = mean_squared_error(test, predictions) 
    rmse = sqrt(mse)
    print('Root Mean Squared Error: %.3f' %rmse)
    mape = np.mean(abs(residuals/test))
    print('Mean Absolute Percentage Error: %.3f' %mape)
    
    #Print summary of the model
    print(arima_model.summary())
    
    # plot forecasts against actual outcomes
    arima_model.forecast(steps=30,alpha=0.05)[0]
    arima_model.plot_predict()
    arima_model.plot_predict(12619,12649)
    

def Auto_ARIMA_Model(data):
    
    auto_arima_model = pmd.auto_arima(data.values, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    

def Simple_Exponential(data):
    
    val = data.values
    train, test = val[1:len(val)-30], val[len(val)-30:]
    ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.2)
    pred_ses = ses_model.predict(start = 12620,end = 12649)
    residuals = test - pred_ses 
    
    #Calculating RMSE and MAPE
    rmse = np.sqrt(np.mean(residuals**2))
    print('Root Mean Squared Error: %.3f' %rmse)
    mape = np.mean(abs(residuals/test))
    print('Mean Absolute Percentage Error: %.3f' %mape)                               
    

def Holt_Winters_Linear(data):
    
    val = data.values
    train, test = val[1:len(val)-30], val[len(val)-30:]
    ses_model = Holt(train).fit(smoothing_level=0.8, smoothing_trend=0.2)
    pred_ses = ses_model.predict(start = 12620,end = 12649)
    residuals = test - pred_ses 
    
    
    #Calculating RMSE and MAPE
    rmse = np.sqrt(np.mean(residuals**2))
    print('Root Mean Squared Error: %.3f' %rmse)
    mape = np.mean(abs(residuals/test))
    print('Mean Absolute Percentage Error: %.3f' %mape)                               
    

def FB_Prophet(data):
    
    #instantiate Prophet
    data = data.rename(columns={'Rate': 'y', 'Date': 'ds'})
    data['ds'] =  pd.to_datetime(data['ds'], format='%d/%m/%Y')
    prophet_model = Prophet(daily_seasonality=True) 
    prophet_model.fit(data)
    
    future_data = prophet_model.make_future_dataframe(periods=30, freq = 'D')
    
    
    forecast_data = prophet_model.predict(future_data)
    forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
    
    fig = prophet_model.plot(forecast_data)
    prophet_model.plot_components(forecast_data)
    
    final_df = pd.DataFrame(forecast_data)
    actual_chart = go.Scatter(y=data["y"], name= 'Actual')
    predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
    predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
    predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')
    py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower])
    
    print("Future 30 days' prediction data is:\n",forecast_data.head())
    return prophet_model


def ANN_Model(data):
    data = np.array(data)
    scaler = MinMaxScaler()
    df = scaler.fit_transform(data)
    
    #Training and test sets
    train = df[:11919]
    test = df[11919:]
    
    X_train = train[:-1]
    y_train = train[1:]

    X_test = test[:-1]
    y_test = test[1:]
    
    #Create model
    K.clear_session()
    ann_model = Sequential()
    ann_model.add(Dense(12, input_dim=1, activation='relu'))
    ann_model.add(Dense(1))
    ann_model.summary()
    
    ann_model.compile(loss= 'mse', metrics=[tensorflow.keras.metrics.RootMeanSquaredError(name='rmse')], optimizer='adam')
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history = ann_model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
    
    #Prediction
    y_pred = ann_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    r2_test = r2_score(y_test, y_pred)
    print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
    print("Root Mean Squared Error: %.3f" %rmse)
    
    return y_test,y_pred


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
    lstm_model.summary()
    
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
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    r2_test = r2_score(y_test, y_pred)
    print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, x_test.shape[0], x_test.shape[1])))
    print('Root Mean Squared Error: %.3f' %rmse)
    
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
    gru_model.summary()
    
    #Prediction
    y_pred = gru_model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = np.array(y_test).reshape(-1,1)
    y_test = scaler.inverse_transform(y_test)
    
#     test_mse = tensorflow.keras.metrics.mean_squared_error(y_test, y_pred)
#     rmse_test = np.sqrt(test_mse)
    print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred)))
    r2_test = r2_score(y_test, y_pred)
    print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    print("Root Mean Squared Error: %.3f" %rmse)
    
    return y_test,y_pred