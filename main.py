import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import copy
'''my libs'''
from portfolios import *
from strategies import *
from utilities import *
from signal_generation_daily import *
from analyze_perf import *
from sig_agg import *

import warnings
warnings.filterwarnings("ignore")

display_setting()

'''
Structure:
- This main module depends on modules mentioned under 'my libs' above.
- Strategy and Portfolio are the two abstract classes defined in ABS_classes on which the modules "Strategies" and "Portfolios" depend
- Strategies use one of the trend detection methods defined in signal generation
- The signals generated by strategy are passed to one of the Portfolios
- Mention diff between portfolio types
- Final output?

Hopw much money to invest in each individual sector is a different question not analyzed here. it depends on sector capacity etc,. Just assume something for now if needed.

Still to add
- Histogram of various indicators/predictors (not poredictions)
- overlay of indicator with market moves
- uncertainty in parameter estimation if any
- histogram of returns - if skewed one way - otherwise maybe conditioning required for up move vs down move
- sub periods- to see if there are periods where markets are more mean reverting than trending
'''

'''variables for base data'''
start_date = datetime.datetime(2014, 01, 02, 23, 59)
end_date = datetime.datetime(2016, 02, 19, 23, 59)
base_series = 'px' # can be px or rx for price or return. The selected trend detection method is applied to this base series or its smoothed version
smoothing = 'None' # can be None, MA, EWMA. This is applied on top of the base series.
                    # It might make sense of have different base and smoothing combination for different trend detection methods but is that too much data mining?
'''variables for signal generation'''
method_list = [Heuristic().Lookback_Direction] # specify a trend methof list to loop through
param_list = [30] # addl params e.g. PIN: threshold, SVR/NB?Karat: roll_length, CCRT/MPR: max_lag, var_ratio : q, Hurst: max_blocksetc
method_list_names = [m.__name__ for m in method_list]
signal_method = '' # combined for direction * strength, else only uses direction
windows = ['1D', '2D', '3D']
forecast_horizon = '1D' # what to do if horizon <1D

'''variables for aggregation'''
aggregate_w = 'n' # aggregate across lookback windows for a given method
agg_method_list = [running_perf, optimal_no_corr, running_regression] # specify a method to aggregate signals across diff lookback windows
signal_rebal_freq = 'D' # or H or W. Each day/week/hour you update the signals from different signal genrators, and generate trades keeping the portfolio weights same
# port_opt_method = max diversification etc
# portfolio rebal freq = ?

'''variables for analysis and plots'''
analyze = 'y'
running = 'n'
histogram = 'n'
scatter = 'n'
running_agg = 'n'

'''symbol and data variables'''
bbg_sym_list = ['USDKRW_Curncy'] # ', USDPLN_Curncy, 'USDINR_Curncy'
cols = ['price']
bars = dict.fromkeys(bbg_sym_list)

signals = {x: dict.fromkeys(method_list_names) for x in bbg_sym_list} # empty dict of dicts with keyvals = bbg symbols
agg_signal = copy.deepcopy(signals) # dict instead of df since dates might be different across symbols
actual = dict.fromkeys(bbg_sym_list)
daily = dict.fromkeys(bbg_sym_list)
fig_no = 1

if 1*(running == 'y') + 1*(histogram == 'y') + 1*(scatter == 'y') + 1*(running_agg == 'y') > 0:
    analyze_plt = 'y' # if you want some plots with analysis
else:
    analyze_plt = 'n'

def get_data_for_symbol(sym, cols):
    min_bars = get_data(sym, cols)
    if base_series == 'Rx':
        min_bars = np.log(min_bars).diff(1)
    min_bars = convert_tz(min_bars)
    selected_data = min_bars[(min_bars.index > start_date) & (min_bars.index < end_date)]
    return selected_data

for isym, sym in enumerate(bbg_sym_list):

    '''step 1- get symbol and data'''
    bars[sym] = get_data_for_symbol(sym, cols)

    ''' step 2- signal generation
    using different trend detection methods and using diff lookback window sizes'''

    for im, m in enumerate(method_list):
        df_list = []

        if m.__name__ == "variance_ratio" and base_series == "px":
            bars_sym = np.log(bars[sym]).diff(1)
        else:
            bars_sym = bars[sym]

        if analyze_plt == 'y':
            if running == 'y':
                plt.figure(fig_no)
                plt.title(sym + " " +method_list_names[im] + " running perf")
            if histogram == 'y':
                plt.figure(fig_no + 1 * (running == 'y'))
                plt.title(sym + " " + method_list_names[im] + " hist")
            if scatter == 'y':
                plt.figure(fig_no + 1* (running == 'y') + 1*(histogram == 'y'))
                plt.title(sym + " " + method_list_names[im] + " p_scatter")
            if running_agg == "y":
                f = fig_no + 1*(running == "y") + 1*(histogram == 'y') + 1*(scatter == 'y')
                plt.figure(f)
                plt.title(sym + " " + method_list_names[im] + " Agg running perf")
            plt.figure(1)

        for iw, w in enumerate(windows):
            if (im == 0) & (iw == 0):
                '''calculate actual moves at given forecast horizon and daily prices'''
                '''bars[sym] used here rather than bars_sym because we never need returns here only prices'''
                actual_moves = create_window_df(bars[sym], forecast_horizon, "n")
                actual[sym] = actual_moves.dropna().ix[:, 0] # this actual is only used in analyzing the perf of signal. Only window = '1D' is used as we are only looking 1 day ahead. Only first method is used as method doesnt change actual moves next day
                daily[sym] = pd.DataFrame(bars[sym].resample('1D', how = 'last').dropna())
                daily[sym].index = daily[sym].index.date

            args = (bars_sym, w, param_list[im])
            strat = GTS(sym, bars_sym, m, *args) # instantiate an object of the class GTS using specified trend method
            temp = strat.generate_signals()
            temp.dropna(inplce = True)

            if analyze == 'y': # this will only analyze for a given symbol and window (not aggregate)
                if im == 0 and iw ==0:
                    print '\n', sym
                if iw ==0:
                    print m.__name__, " ", param_list[im]
                print w, " hit ratio: ", hit_return(temp['direction'], actual[sym]), \
                    ", mean move right, wrong: ", relative_pl(temp['direction'], actual [sym], daily[sym])

            if analyze_plt == 'y':
                create_plots(fig_no, temp, actual[sym], iw, w, len(windows), sym, m, [running, histogram, scatter])
                plt.figure(max(fig_no - 1, 1)) # activate this # make this one active for next loop of running performance

            if signal_method == 'combined':
                temp['final'] = temp['direction'] * temp['strength']
                temp.drop(['trend', 'direction', 'strength', 'predictor'], axis = 1, inplace = True)
                temp.rename(columns = {'final' : w}, inplace=True)
            else:
                temp.drop(['trend', 'strength', 'predictor'], axis =1, inplace = True)
                temp.rename(columns = {'direction' : w}, inplace=True)
            df_list.append(temp) # each member of the list is a signal dataframe for a given window

        signals[sym][m.__name__] = pd.concat(df_list, axis =1) # concatenate different windows for a given symbol/method into single df

        '''step 3- signal aggregation across windows'''
        # Input - for each asset in portfoliom, for each signal method , a signal series for each window
        # Output - for each asset in portfolio, for each signal method, a signal aggregated across windows
        if aggregate_w == 'y':
            w_list = [30, 30, 30] # for testing only using various lookback windows for signal aggregation
            e = 0
            for agg_i, agg_method in enumerate(agg_method_list):
                if agg_method == running_perf:
                    agg_signal[sym][m.__name__] = agg_method(signals[sym][m.__name__], signal_rebal_freq, actual[sym], w_list[agg_i], e)
                else:
                    agg_signal[sym][m.__name__] = agg_method(signals[sym][m.__name__], signal_rebal_freq, actual[sym],  w_list[agg_i])

                if analyze == 'y': # this will analyze for agg symbol
                    print agg_method.__name__, " hit ratio: ", hit_return(agg_signal[sym][m.__name__], actual[sym]), \
                        ", mean move right, wrong: ", relative_pl(agg_signal[sym][m.__name__], actual[sym], daily[sym])

                if running_agg == "y":
                    create_plots(f, agg_signal[sym][m.__name__], actual[sym], iw, agg_method.__name__, len(windows), sym, m, ['y', '', ''])
        if running == 'y':
            plt.figure(fig_no)
            plt.legend(loc = 'best')
        if histogram == 'y':
            plt.figure(fig_no + 1*(running == 'y'))
            plt.legend(loc = 'best')
        if running == 'y':
            plt.figure(f)
            plt.legend(loc = 'best')
        fig_no = fig_no + 1*(running == 'y') + 1*(histogram == 'y') + 1*(scatter == 'y') + 1*(running_agg == 'y')

if analyze_plt == 'y':
    plt.show()

'''step 4 portfolio optimization
combine different trend methods together'''
