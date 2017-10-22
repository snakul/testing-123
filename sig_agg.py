import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.linalg import lstsq
import scipy.integrate as integrate
from scipy import optimize as op
from sklearn.decomposition import PCA
from sklearn import preprocessing, svm,grid_search
from utilities import*

'''these functions are intended to be called from the "main" module for each symbol separatlely'''

def simple_av(signal_df, rebal_freq):
    '''simply averages the signals across windows
    signal_panel: panel of dataframes -1 df for each trend method. Panel dimension is (no of trend_methods * time history * no of windows) and we are averaging across windows
    returns a single dataframe containing a column corresponding to each trend_method'''
    agg_signal = signal_df.mean(axis=1) # axis 1= windows
    agg_signal.index = pd.to_datetime(agg_signal.index, unit='s')
    agg_signal = agg_signal.resample(rebal_freq, how='last', axis=0).dropna() # a df of shape (time history, no of trend methods)
    return agg_signal

def running_perf(signal_df, rebal_freq, actual, lookback=30, epsilon =0):
    '''
    returns the one with max average hits over the lookback window, except in epsilon percent of casses where pick a random window
    signal_df: dataframe -1 col for each window. df shape is (time history, no of windows) and we are averaging across windows
    return : a single dataframe containing a single column corresponding to aggregated signal
    '''
    hits= np.sign(signal_df) == np.sign(actual[signal_df.index[0]]) # only compare actual on predicted index dates to avoid issues of unequal lengths
    running_performance = pd.rolling_mean(hits, lookback).dropna()
    best_window = running_performance.idmax(1)
    throws = np.random.uniform(0,1 , len(best_window)) # for epsilon greedy strategies
    best_window[throws < epsilon] = signal_df.columns[np.random.randint(0, signal_df.shape[1],(throws < epsilon).sum())]
    agg_signal = pd.DataFrame(signal_df.lookup(running_performance.index, best_window), index = running_performance.index)
    agg_signal.index = pd.to_datetime( agg_signal.index, unit= 's')
    agg_signal = agg_signal.resample(rebal_freq, how= 'last', axis=0).dropna()
    return agg_signal

def running_regression( signal_df, rebal_freq, actual, lookback=30):
    ''' returns the signal corresponding to regression based estimate of past moves on different windows'''
    common_index = signal_df.index.intersection(actual.index)
    signal_df = signal_df.ix[common_index]
    actual = actual[common_index]

    model = pd.stats.ols.MovingOLS( y= actual, x= signal_df, window_type = 'rolling', window = lookback, intercept = True)
    agg_signal = model.y_predict
    agg_signal.index = pd.to_datetime( agg_signal.index, unit ='s')
    agg_signal = agg_signal.resample(rebal_freq, how='last', axis=0).dropna()
    return agg_signal

def optimal_no_corr(signal_df, rebal_freq, actual, lookback=30):
    ''' '''
    variance_df = pd.rolling_var( signal_df, lookback)
    inv_var_df = 1.0/variance_df
    wts_df = inv_var_df.divide( inv_var_df.sum(axis=1), axis=0 )
    wtd_signals = signal_df.multiply (wts_df, axis=0)
    agg_signal = wtd_signals.sum( axis=1)
    agg_signal.index = pd.to_datetime( agg_signal.index, unit ='s')
    agg_signal = agg_signal.resample( rebal_freq, how='last', axis=0).dropna()
    return  agg_signal

def pca_wt(signal_panel):
    '''attaches more wt to lookback window with which others are most correlated -
    assumes that window that is most influential in determining the variance of all signals in the mix
    is also the most informative, and that its signals are therefore likely to be better predictors of future moves.
    Weighs all signals according to those implied by first eigenvector'''
    '''if you use intraday data, you cant calculate pca at each point in time(memory overload)
    so only take pca at rebal_freq interval but use history = daily intraday data to calculate it'''

    pca = PCA(n_components=1)
    method_list = [i for i in signal_panel]
    agg_signal = dict.fromkeys( method_list )

    for method_no in signal_panel:
        signal_df = signal_panel[method_no]
        g = signal_df.groupby(pd.TimeGrouper('M'))
        d = {k:v for k, v in g}
        spans = d.keys()
        spans.sort()
        l = []
        for span in spans:
            p1 = d[span]
            if p1.shape[0] > 0:
                l.append(pca.fit_transform(p1).T[0])
            else:
                l.append(0)
        agg_signal[method_no] = pd.DataFrame(l, index= spans)
    agg_signal = pd.concat(agg_signal.values(), axis=1)
    agg_signal.columns = method_list
    return agg_signal