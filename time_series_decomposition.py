# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 21:58:03 2021

@author: Yasmine
"""

from datetime import datetime
import time
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pylab as plt
from functions import *
from matplotlib.pylab import rcParams
from statsmodels.stats.stattools import durbin_watson
rcParams['figure.figsize'] = 15, 6

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y %W %w')

rand_series=np.random.normal(200, 5, size=(50, 1))
data= pd.read_csv('data.csv',sep=";", parse_dates=['data_type'], index_col='data_type',date_parser=dateparse)


''''''''''''''''''''''''''''' Removing Stationnarity by Differenciation '''''''''''''''
def differenciate(series):
    series_log = np.log(series)
    #plt.plot(series_log)
    test_stationarity(series_log)# Not stationary
    print('Durbin Watson test on the serie: \n')
    print(durbin_watson(series_log))
    series_log = np.log(series)
    moving_avg = pd.Series(series_log).rolling(window=12).mean()
    #plt.plot(series_log)
    #plt.plot(moving_avg, color='red')
    series_log_moving_avg_diff = series_log - moving_avg
    series_log_moving_avg_diff.head(12)
    series_log_moving_avg_diff.dropna(inplace=True)
    print('Stationnarity test of the difference moving average : \n')
    test_stationarity(series_log_moving_avg_diff)#•Stationnary series
    print('Durbin Watson test on the difference moving average: \n')
    print(durbin_watson(series_log_moving_avg_diff))#positive autocorrelation
    
    expwighted_avg = series_log.ewm(span=2).mean()
    series_log_ewma_diff = series_log - expwighted_avg
    print('Stationnarity test of the difference exponential moving average : \n')
    test_stationarity(series_log_ewma_diff)#•Stationnary series
    print('Durbin Watson test on the difference exponential moving average: \n')
    print(durbin_watson(series_log_ewma_diff))#positive autocorrelation
    #Differencing
    series_log_diff = series_log - series_log.shift()
    #plt.plot(series_log_diff)
    series_log_diff.dropna(inplace=True)
    print('Stationnarity test of the differenciation series : \n')
    test_stationarity(series_log_diff)#•Stationnary series
    print('Durbin Watson test on the differenciation series: \n')
    print(durbin_watson(series_log_diff))#positive autocorrelation
    
''''''''''''''''''''''''''''' Removing Stationnarity by Decomposition '''''''''''''''
def decompose(series):
    trend = pd.Series(series).rolling(window=12).mean()
    
    seasonal_and_residual = series-trend
    #seasonal_and_residual.dropna(inplace=True)
    
    freq=52
    period_averages = seasonal_mean(seasonal_and_residual, freq)
    period_averages -= np.mean(period_averages, axis=0)
    nobs=len(series)
    
    seasonal = np.tile(period_averages.T, nobs // freq + 1).T[:nobs]
    seasonal=pd.Series( (v for v in seasonal) )
    seasonal.index=trend.index

    residual=seasonal_and_residual-seasonal

    plt.subplot(411)
    plt.plot(series, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend',c='red')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label='Seasonality',c='green')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual.index,[0]*len(residual.index),c="r")
    matplotlib.pyplot.scatter(residual.index, residual.values)
    plt.show()
    
    
    plt.legend(loc='best')
    plt.tight_layout()
    residual.dropna(inplace=True)
    print('Stationnarity test on residuels: \n')
    test_stationarity(residual)
    print('Durbin Watson test on residuels: \n')
    print(durbin_watson(residual))
    #plt.savefig('Decomposition_img/CN/{product}.png'.format(product=product))
    #plt.close()
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(series,freq=52)
    fig = plt.figure()  
    fig = decomposition.plot()  
    fig.set_size_inches(15, 8)
    residual=decomposition.resid
   
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Original',c='black')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend',c='red')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(decomposition.seasonal,label='Seasonality',c='yellow')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual.index,[0]*len(residual.index),c="r")
    matplotlib.pyplot.scatter(residual.index, residual.values)

differenciate(series)
decompose(series)
    