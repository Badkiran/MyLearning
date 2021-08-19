# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:25:23 2021

@author: Kiran B
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from fbprophet.plot import plot_cross_validation_metric
import warnings
warnings.filterwarnings("ignore")

# Read file
data = pd.read_excel("C:/Users/kiran/Desktop/Project 1/DEXINUS.xlsx", skiprows = range(0,10))
data

# Renaming the columns to ease the use
data = data.rename({'observation_date': 'Date', 'DEXINUS': 'Rate'}, axis=1)
data['Date'] = pd.to_datetime(data.Date)
data

data.info()

data.isnull().sum()

# Imputing the null values of the Rate column
data['Rate'] = data['Rate'].fillna(value= 0.0)

for i in range(0, len(data['Rate'])):
    if data.Rate[i] == 0.0:
        data.Rate[i] = (data.Rate[i-1] + data.Rate[i-2])/2.0
data

data.isnull().sum()

data.describe()

df = data.copy()
data_fbmodel = data.copy()
df.set_index('Date',inplace = True)
df

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data

test = data.copy()
test.set_index('Date',inplace = True)
test

from Showgraph import *

#Lineplot
Plot_Series(df)

# Visualizing data at 10-year interval
drange1 = df.loc["1973-01-02": "1983-12-01"]
Sub_Lineplot(drange1)
drange2 = df.loc["1984-01-01": "1994-12-01"]
Sub_Lineplot(drange2)
drange3 = df.loc["1995-01-01": "2005-12-01"]
Sub_Lineplot(drange3)
drange4 = df.loc["2006-01-01": "2016-12-01"]
Sub_Lineplot(drange4)
drange5 = df.loc["2017-01-01": "2021-06-25"]
Sub_Lineplot(drange5)

#Boxplot
Box_Plot1(data) #Entire Data
Box_Plot2(data) #Group by Year

data.set_index('Date',inplace = True)

#Histogram
df.hist()

#Distplot
Dist_Plot(df)

#Lagplot
Lag_Plot(df)

# Checking the correlation between t-1 and t+1 values
values = pd.DataFrame(df.values)
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)

# ACF and PACF plots
ACF_PACF_Plots(df)

#Based on decaying ACF, we are likely dealing with an Auto Regressive process
#Based on PACF, we should start with an Auto Regressive model with 2 lags only

#Time Series Decomposition plot
Decompose(df)

# Testing if Data is Stationary
# Augmented Dickey Fuller Test for Stationarity
from Tests import *
adfuller_test(data['Rate'])

# Normalize
avg, dev = test.mean(), test.std()
test = (test - avg) / dev

Plot_Series(test.Rate)
plt.axhline(0, linestyle='--', color='k', alpha=0.5)

# Removing Trend
# Take First Difference to Remove Trend
first_diffs = data.Rate.values[1:] - data.Rate.values[:-1]
first_diffs = np.concatenate([first_diffs, [0]])

test = test.diff().dropna()
Plot_Series(test.Rate)
plt.axhline(0, linestyle='--', color='k', alpha=0.5)

#Set first difference as variable in dataframe
data['FirstDifference'] = first_diffs
data

plt.figure(figsize=(20,10))
plt.plot(data.FirstDifference)
plt.title('First Difference over Time', fontsize=16)
plt.ylabel('Price Difference', fontsize=16)
for year in range(1973,2022):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
    
# Remove increasing Volatality
annual_volatility = test.Rate.groupby(test.index.year).std()
annual_volatility

annual_vol = test.Rate.index.map(lambda d: annual_volatility.loc[d.year])
annual_vol

test['Rate'] = test['Rate']/annual_vol
test

Plot_Series(test.Rate)
plt.axhline(0, linestyle='--', color='k', alpha=0.5)

# ACF and PACF Plots on First Difference data
ACF_PACF_Plots(data.FirstDifference)

# Testing if First Difference Data is Stationary
adfuller_test(df['Rate'])

# Model Building

#Auto Regressive Model

#1.AR Model (yt = c + ϕ1.yt−1 + ϕ2.yt−2 + εt)
from Models import *
AR_Model(df)

#2.ARMA Model
#Considering data from 2010 onwards
start_date = pd.to_datetime('2010-01-01')
end_date = pd.to_datetime('2021-06-25')
data_ARMA = data[start_date:]

# So the ARMA(2,2) model is
ARMA_Model(data_ARMA)

#3.ARIMA Model
ARIMA_Model(df,1,1,0)

#Simple Exponential Method
Simple_Exponential(df)  

#Holt Winters Linear Trend Method
Holt_Winters_Linear(df)

#FB Prophet Model
model = FB_Prophet(data_fbmodel)

data_cv = fb_crossval(model)

fig = plot_cross_validation_metric(data_cv, metric='rmse')

#Neural Network Models

#1.Artificial Neural Network(ANN)
y_test1,y_pred1 = ANN_Model(df)

NeuralNet_Plot(y_test1,y_pred1)

#2.Long Term Short Memory(LSTM Recurrent Neural Network)
y_test2,y_pred2 = LSTM_Model(df)

NeuralNet_Plot(y_test2,y_pred2)

#3.Gated Recurrent Unit(GRU)
y_test3,y_pred3 = GRU_Model(df)

NeuralNet_Plot(y_test3,y_pred3)