import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing, svm, grid_search

'''entry exit rules have to be dependant on pnl, and since i have separated the signals from sizing,entry exit 
function cant be called or signal genration now'''
def entryexit(positions, signals, prices, eq):
    '''to be optimized'''
    inv= eq/100 # fixed investmen in single trade # can implement and optimize sizing later
    risk= 1e-2 # % of equity risked on each trade
    potential = 2e-2 # % move up for take profit

    current_pos=0
    for i , signal in enumerate(signals.value):
        if current_pos==0: # new trades only if current position flat
            dir= signal[0] #keeping direction and signal separate as there might be no trades despite a signal
            size=1 # implement sizing algo
            current_pos = current_pos + dir*size
            entry_p = prices.iloc[i]
        else:
            take_profit = get_take_profit(entry_p, inv, eq, potential)
            stop_loss = get_stop_loss(entry_p, inv, eq, risk)
            if prices.iloc[i,0] > take_profit or prices.iloc[i,0]< stop_loss:
                dir=-dir # reverse position completely
                '''can also instead take partial profits- by trading 2 units and selling 1 of them at take profit level'''
                size=1
                current_pos = current_pos + dir*size

        positions.iloc[i,0]= current_pos
    return positions

def get_take_profit(entry_p, inv, eq, potential):
    tp= entry_p*( inv + eq*potential)/ inv
    # this is the mirror image of get_stop_loss with risk replaced by potential
    # eq*potential = target profit. should target profit be based on portfolio equity
    return tp[0]

def get_stop_loss(entry_p, inv, eq, risk):
    # from gibbon burke's risk control and money mgmt article "managing your money"
    # gives a price level where stop loss should be
    # the thing to be optimized in this stop loss method would be "risk"
    sl= entry_p*( inv- eq* risk)/inv # pre determined stop loss or exit price
    # entry_p = entry price on the trade- can use a moving ref price here instead of this that gives just fixed stop
    # inv= investment amount in this/last trade
    # eq = portfolio equity (cash and holdings)- can try to use Seykota's core equity also as mentioned in the article above
    # risk = max risk % per trade
    return sl[0]

def sharpe (pnl, roll_p, benchmark):
    '''pnl is a onl series, roll_p is the lookback window'''
    sharpe = np.zeros(pnl.shape[0]- roll_p +1) # if pnl is 10 periods, and roll is the same, then just 1 output
    for i in range(roll_p, pnl.shape[0] +1):
        num = np.mean(pnl[i- roll_p:i]) - benchmark[i]
        den = np.std(pnl[i- roll_p:i])
        sharpe[i-roll_p]= num/den
    return sharpe