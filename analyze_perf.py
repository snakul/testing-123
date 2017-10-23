import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

"""
This part (other part below) module is intended to have functions related to analyzing performance of individual signal detection methods on various assets and asset classes

All these methods are applicable to signal aggregated across windows and single window except for wts across wndows
    hit ratio and avg return
    relative pl
    flipping ratio
    corr of a method with others averaged across assets
    how the wts of different look back windows change over time and for different assets
"""

"""**********point to note***********
the signals corresponding to a date index correspond to prediction for next day, so careful to algn the prediction with actual date move

Notes:
do the performance analysis for:
selected anems in each asset class 
selected windows- short,med,long and aggreagated (1D, 3D, 7D, agg)"""

def hit_return(predicted, actual):
    '''returns the % times the predicted direction was correct,and average return of a method'''
    common_index= predicted.index.intersection(actual.index)

    if isinstance(predicted, pd.core.series.Series)==False:
        predicted = predicted.ix[:,0]
    if isinstance(actual, pd.core.series.Series)==False:
        actual = actual.ix[:,0]

    predicted = predicted[common_index]
    actual= actual[common_index]
    predicted= predicted.convert_objects(convert_numeric= True)
    '''
    labels=[1,-1,0]
    cm= sklearn.metrics.confusion_matrix(np.sign(actual), np.sign(predicted_,labels=labels)
    cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    cm= pd.DataFrame(cm, index = labels, columns = labels)
    hits= pd.DataFrame(np.sign(actual).values==np.sign(predicted).values, index = actual.index) # only compare actual on predicted index dates to avoid issues of unequal lengths 
    hit_ratio= float(hits.sum())/hits.count()
    '''
    hits_up= predicted[actual>0]>0
    up_ratio = float(hits_up.sum())/(actual>0).sum()
    hits_down=predicted[actual<0]<0
    down_ratio = float(hits_down.sum())/(actual<0).sum()
    hit_ratio = up_ratio+ down_ratio
    return '%.2f' % np.round(hit_ratio,2)

def relative_pl(predicted,actual, levels):
    '''returns the mean move when you are right vs wrong'''
    if isinstance(predicted, pd.core.series.Series)==False:
        predicted = predicted.ix[:,0]
    if isinstance(actual, pd.core.series.Series)==False:
        actual = actual.ix[:,0]
    if isinstance(levels, pd.core.series.Series)==False:
        levels = levels.ix[:,0]

    common_index = predicted.index.intersection(actual.index)
    predicted = predicted[common_index]
    actual = actual[common_index]
    levels = levels[common_index]
    correct_pred = pd.DataFrame(np.sign(actual).values == np.sign(predicted).values, index= actual.index)

    concat= pd.concat([correct_pred, actual, levels], axis=1)
    concat.columns = ['correct', 'actual', 'levels']
    move_correct = abs(concat.loc[concat['correct']==True]['actual'])
    levels_correct = concat.loc[concat['correct'] == True]['levels']
    move_wrong = abs(concat.loc[concat['correct'] != True]['actual'])
    levels_wrong = concat.loc[concat['correct'] != True]['levels']

    mean_move_up= 10000* np.ma.masked_invalid(move_correct/levels_correct).mean()
    mean_move_down = 10000* np.ma.masked_invalid(move_wrong/levels_wrong).mean()

    return '%.2f' % np.round(mean_move_up,2) , '%.2f' % np.round(mean_move_down,2)

def flips (predicted):
    '''returns the number of times prediction changed on consecutive days, the more you flip, the less market impact but more spread and commision costs'''
    df= pd.DataFrame(np.sign(predicted))
    flips = pd.rolling_apply(df, 2,np.product()) # pandas 19 : df.rolling(2).product()==-1)
    return float(flips.sum())/flips.count()

def corr(p1,p2):
    '''returns the corr of predictions for 2 given predicted series'''
    return np.corr(p1,p2)

def window_wts(p1):
    NotImplemented




'''This part of the module intended to have functions related to analyzing performance of different portfolio methods on various assets/asset classes

All these methods are applicable to signal aggregated across windows and single window except for wts across windows
    avg return
    cum return tables and charts
    annual vol
    distribution of returns
    sharpe
    sortino
    no of trades
    hit ratio
    avg + return
    avg - return'''