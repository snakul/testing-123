import numpy as np
import pandas as pd
from ABS_classes import Portfolio
from risk_mgmt import *


class MarketOnOpenPortfolio(Portfolio):
    """Inherits Portfolio to create a system that purchases 100 units of
    a particular symbol upon a long/short signal, assuming the market
    open price of a bar.

    In addition, there are zero transaction costs and cash can be immediately
    borrowed for shorting (no margin posting or interest requirements).

    Requires:
    symbol - A stock symbol which forms the basis of the portfolio.
    bars - A DataFrame of bars for a symbol set.
    signals - A pandas DataFrame of signals (1, 0, -1) for each symbol.
    initial_capital - The amount in cash at the start of the portfolio."""

    def __init__(self, symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        """Creates a 'positions' DataFrame that simply longs or shorts
        100 of the particular symbol based on the forecast signals of
        {1, 0, -1} from the signals DataFrame."""
        positions = pd.DataFrame(index=signals.index).fillna(0.0)
        positions[self.symbol] = 100 * signals['signal']
        return positions

    def backtest_portfolio(self):
        """Constructs a portfolio from the positions DataFrame by
        assuming the ability to trade at the precise market open price
        of each bar (an unrealistic assumption!).

        Calculates the total of cash and the holdings (market price of
        each position per bar), in order to generate an equity curve
        ('total') and a set of bar-based returns ('returns').

        Returns the portfolio object to be used elsewhere."""

        # Construct the portfolio DataFrame to use the same index
        # as 'positions' and with a set of 'trading orders' in the
        # 'pos_diff' object, assuming market open prices.
        ###portfolio = self.positions * self.bars['Open'] pos_diff = self.positions.diff()
        portfolio= pd.DataFrame(index= self.bars.index)

        # Create the 'holdings' and 'cash' series by running through
        # the trades and adding/subtracting the relevant quantity from
        # each column
        portfolio['holdings'] = self.postions.mul(self.bars['Open'], axis ='index') #(self.positions * self.bars['Open']).sum(axis=1)
        portfolio['cash'] = self.initial_capital - portfolio['holdings'].cumsum() #self.initial_capital - (pos_diff * self.bars['Open']).sum(axis=1).cumsum()

        # Finalise the total and bar-based returns based on the 'cash'
        # and 'holdings' figures for the portfolio
        portfolio['total'] = portfolio['cash'] + self.positions[self.symbol].cumsum() #portfolio['holdings']
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio


class EntryExitPortfolio(Portfolio):

    def __init__(self, symbol, bars, signals, initial_capital=100000.0):
        self.symbol = symbol
        self.bars = bars
        self.signals = signals
        self.initial_capital = float(initial_capital)
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(np.zeros(self.signals.shape[0]), index= self.signals.index)
        positions = entryexit (positions, self.signals, self.bars, self.initial_capital)
        positions.columns = self.bars.columns
        return positions

    def backtest_portfolio(self):
        portfolio= pd.DataFrame(index= self.bars.index)
        portfolio['holdings'] = self.postions.mul(self.bars, axis ='index')
        portfolio['cash'] = self.initial_capital - portfolio['holdings'].cumsum()
        portfolio['total'] = portfolio['cash'] + self.positions.cumsum() * self.bars
        portfolio['returns'] = portfolio['total'].pct_change()
        return portfolio