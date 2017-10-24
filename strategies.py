import numpy as np
import pandas as pd
from ABS_classes import Strategy
from risk_mgmt import *
from signal_generation_daily import *

class GTS(Strategy):
    '''stands for generalized trend strategy- applies trend detection to pure price series
    requires:
    symbol - to form a strategy on
    bars- a dataframe of price bars for above symbol
    trend method- how to determine trend'''

    def __init__(self, symbol, bars, signal_method):
        self.symbol = symbol
        self.bars= bars
        self.signal_method = signal_method

    def generate_signals(self, *args):
        '''returns the dataframe of symbols comtaining signals to go long, short or hold (1,-1,0)'''
        signals = self.signal_method(args)
        # take the difference of the signals in order to generate actual trading orders
        return signals

