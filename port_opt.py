import numpy as np

'''signal_df is this module is a df of signals from different strategies/asset classes/ trend direction methods

signal_df in signal_agg module is a df of signals using different windows but using the same trend detection method'''

def equal_wt(signal_df):
    '''weighs each strategy equally'''
    return (1/signal_df.shape[1])*np.ones(signal_df.shape[1])

def mvo(signal_df):
    '''mean variance opt'''
    NotImplemented

def equal_risk(signal_df):
    '''equal risk (vol) from each strategy- similar to inverse vol but also accounts for correlations'''
    NotImplemented

def max_diversity(signal_df):
    '''estimates wts thatmaximize the distance between wtd av vol of each strategy and overall portfolio'''
    NotImplemented
    