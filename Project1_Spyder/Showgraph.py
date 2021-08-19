# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 18:00:52 2021

@author: Kiran B
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import lag_plot
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.seasonal import seasonal_decompose


def Line_Plot(data):
    
    # set figure size
    plt.figure(figsize = (10, 6))
  
    # plot a simple time series plot using seaborn.lineplot()
    sns.lineplot(x = 'Date', y = 'Rate', data = data, label = 'USD-INR') 
    
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Rate', fontsize=16)
    for year in range(2010, 2023):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)


def Sub_Lineplot(drange):
    
    plt.figure(figsize = (25,2))
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Rate', fontsize=16)
    sns.lineplot(data = drange["Rate"])
    

def Box_Plot1(data):
    
    data.boxplot(column=['Rate'])
    

def Box_Plot2(data):
    
    plt.figure(figsize = (30,10))
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Rate', fontsize=16)
    sns.set(style='whitegrid')
    sns.boxplot(x="Year", y="Rate",data=data)
    
    
def Plot_Series(data):
   
    plt.figure(figsize=(30,10))
    plt.plot(data)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Rate', fontsize=16)
    
    # setting customized ticklabels for x axis
    pos = [ '1973-01-02', '1978-01-01', '1983-01-01', '1988-01-01', '1993-01-01', '1998-01-01', '2003-01-01', '2008-01-01','2013-01-01', '2018-01-01', '2021-01-01']
    lab = [ '1973', '1978', '1983', '1988', '1993', '1998', '2003', '2008', '2013', '2018', '2021']

    plt.xticks(pos, lab)
    
    for year in range(1973, 2023):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
        

def Lag_Plot(data):
    
    plt.figure(figsize=(30,10))
    lag_plot(data, c='blue')
    

def Dist_Plot(data):
   
    sns.distplot(data,bins=8,kde=True,hist_kws={'edgecolor':'black','linewidth':2, 'linestyle':'--'})
    

def ACF_PACF_Plots(data):
   
    tsa_plots.plot_acf(data,lags=50)
    tsa_plots.plot_pacf(data,lags=50)
    plt.show()
    
    
def Decompose(data):
    
    decompose_ts_add = seasonal_decompose(data,period=60)
    decompose_ts_add.plot()
    

def FirstDiff_Plot(data,start_date,end_date):
    data_ARMA = data
    plt.figure(figsize=(10,6))
    plt.plot(data_ARMA)
    plt.title('First Difference of Exchange Rate', fontsize=20)
    plt.ylabel('Rate', fontsize=16)
    for year in range(start_date.year,end_date.year):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
    plt.axhline(data_ARMA.mean(), color='r', alpha=0.2, linestyle='--')
    
    
def NeuralNet_Plot(y_test,y_pred):
    
    plt.figure(figsize=(12,8))
    plt.title('USD-INR Prediction')
    plt.plot(y_test , label = 'Actual')
    plt.plot(y_pred , label = 'Predicted')
    plt.xlabel('Observation')
    plt.ylabel('Rate')
    plt.legend()