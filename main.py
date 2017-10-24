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
from plotting_functions import *

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
start_date = datetime.datetime(2014, 1, 2, 23, 59)
end_date = datetime.datetime(2016, 2, 19, 23, 59)
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
agg_method_list = [aggregator.running_perf, aggregator.optimal_no_corr, aggregator.running_regression] # specify a method to aggregate signals across diff lookback windows
signal_rebal_freq = 'D' # or H or W. Each day/week/hour you update the signals from different signal genrators, and generate trades keeping the portfolio weights same
# port_opt_method = max diversification etc
# portfolio rebal freq = ?

'''variables for analysis and plots'''
print_perf = 'y'
plot= plotting(running =0, histogram =0, scatter =0, running_agg =0)

'''symbol and data variables'''
bbg_sym_list = ['USDKRW_Curncy'] # ', USDPLN_Curncy, 'USDINR_Curncy'
cols = ['price']
prices = dict.fromkeys(bbg_sym_list)

signals = {x: dict.fromkeys(method_list_names) for x in bbg_sym_list} # empty dict of dicts with keyvals = bbg symbols
agg_signal = copy.deepcopy(signals) # dict instead of df since dates might be different across symbols
rlzd = dict.fromkeys(bbg_sym_list)
daily = dict.fromkeys(bbg_sym_list)
fig_no = 1

for isym, sym in enumerate(bbg_sym_list):

    '''step 1- get symbol and data'''
    prices[sym] = get_data_for_symbol(sym, cols)

    ''' step 2- signal generation
    using different trend detection methods and using diff lookback window sizes'''

    '''calculate actual moves at given forecast horizon and daily prices'''
    rlzd_moves = create_window_df(prices[sym], forecast_horizon, "n")
    rlzd[sym] = rlzd_moves.dropna().ix[:, 0] # this actual is only used in analyzing the perf of signal. Only window = '1D' is used as we are only looking 1 day ahead. Only first method is used as method doesnt change actual moves next day
     
    for im, m in enumerate(method_list):
        df_list = []
        plot.start_plot()

        for iwindow, window in enumerate(windows):
            
            args = (prices[sym], window, param_list[im])
            strat = GTS(sym, prices[sym], m) # instantiate an object of the class GTS using specified trend method
            temp = strat.generate_signals(*args)
            temp.dropna(inplace = True)

            print_perf()

            if plot.analyze_plt == 'y':
                plot.create_signal_plots( temp, rlzd[sym], iwindow, window, len(windows), sym, m)
                plt.figure(max(fig_no - 1, 1)) # activate this # make this one active for next loop of running performance

            if signal_method == 'combined':
                temp['final'] = temp['direction'] * temp['strength']
                temp.drop(['trend', 'direction', 'strength', 'predictor'], axis = 1, inplace = True)
                temp.rename(columns = {'final' : window}, inplace=True)
            else:
                temp.drop(['trend', 'strength', 'predictor'], axis =1, inplace = True)
                temp.rename(columns = {'direction' : window}, inplace=True)
            df_list.append(temp) # each member of the list is a signal dataframe for a given window

        signals[sym][m.__name__] = pd.concat(df_list, axis =1) # concatenate different windows for a given symbol/method into single df

        '''step 3- signal aggregation across windows'''
        # Input - for each asset in portfolio, for each signal method, a signal series for each window
        # Output - for each asset in portfolio, for each signal method, a signal aggregated across windows
        if aggregate_w == 'y':
            lookback_list = [30] # for testing only using various lookback windows for signal aggregation
            e = 0
            for i_agg, agg_method in enumerate(agg_method_list):
                agg_object = aggregator(signal_df = signals[sym][m.__name__],
                                        rebal_freq = signal_rebal_freq,
                                        realized = rlzd[sym],
                                        lookback = lookback_list[i_agg],
                                        eps = e)
                agg_signal[sym][m.__name__] = agg_method(agg_object)
                print_perf_2()

                if plot.running_agg:
                    plot.create_agg_plots( agg_signal[sym][m.__name__], rlzd[sym],  agg_method.__name__, len(windows))
        plot.activate_next()

if plot.analyze_plt == 'y':
    plt.show()

'''step 4 portfolio optimization
combine different trend methods together'''
