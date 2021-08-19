# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 17:58:45 2021

@author: Kiran B
"""

from statsmodels.tsa.stattools import adfuller
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics



def adfuller_test(Rate):
    
    #Ho: Data is non stationary
    #H1: Data is stationary
    result=adfuller(Rate)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print("Reject the null hypothesis(Ho). Data is Stationary")
    else:
        print("Accept the null hypothesis(Ho). Data is non-stationary")
        

def fb_crossval(data_cv):
    
    df_cv = cross_validation(data_cv, initial = '3650 days', period = '365 days', horizon = '30 days')
    df_cv.head()
    df_p = performance_metrics(df_cv)
    df_p.head()
    return df_cv