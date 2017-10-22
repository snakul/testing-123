import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing, svm, grid_search
import datetime
import pytz

TZ_dict = {'USDCZK_Curncy':'Europe/Prague','USDNOK_Curncy':'Europe/Oslo','USDHUF_Curncy':'Europe/Budapest',
           'USDPLN_Curncy':'Europe/Warsaw', 'USDRUB_Curncy':'Europe/Moscow', 'EURUSD_Curncy':'Europe/London',
           'GBPUSD_Curncy':'Europe/London', 'USDMXN_Curncy':'America/Mexico_City', 'USDCOP_Curncy':'America/Bogota',
           'USDCLP_Curncy':'Chile/Continental','USDBRL_Curncy':'Brazil/East', 'USDZAR_Curncy':'Africa/Johannesburg',
           'USDILS_Curncy':'Israel', 'USDCNH_Curncy':'Asia/Shanghai','USDHKD_Curncy':'Hongkong', 'USDIDR_Curncy':'Asia/Jakarta',
           'USDINR_Curncy':'Asia/Calcutta','USDKRW_Curncy':'Asia/Seoul','USDMYR_Curncy':'Asia/Manila', 'USDSGD_Curncy':'Asia/Singapore',
           'USDTRY_Curncy':'Asia/Istanbul','USDPHP_Curncy':'Asia/Manila','USDTWD_Curncy':'Asia/Taipei', 'USDJPY_Curncy':'Asia/Tokyo',
           'AUDUSD_Curncy':'Australia/Sydney'}

def display_setting():
    pd.set_option('display.max_rows', 500, 'display.max_colums', 500, 'display.width', 1000, 'display.precision', 3)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    np.set_printoptions(precision =2, linewidth=1000, threshold=500, suppress=True)

def map_bin(x, bins, labels):
    '''mapper that bins data
    np.digitize is doing our heavy lifting, by bracketing our x vlaue in the bin.
    the kwargs are necessary in order to address the edge case where x and the max bin are the same, and this to
    include it we need to set right =True in np.digitize
    '''
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    bin = bins[np.digitize([x], bins, **kwargs)[0]]
    return labels[bins.index(bin)]

def get_data(sym, cols):
    NotImplemented

def convert_tz( df_singlename, drop_original = 'T'):
    '''
    :param df_singlename: time series or dataframe of proces/returns etc for a single currency where the column name = BBG tcker and index = timezone aware datetimes
    :param drop_original: drop the original index
    :return: dataframe/series with index changed to new timezone
    '''
    target_zone = TZ_dict[df_singlename.columns[0]]
    if drop_original == 'T':
        return df_singlename.tz_convert(pytz.timezone(target_zone))

def smooth (df, method="None", arg=5):
    '''
    :param df: input series or dataframe with ideally only one column
    :param method: method intended for smoothing, currently None(default), MA ot EWMA
    :param arg: for MA arg equals window, for EWMA arg equals span
    :return: smoothed df
    '''
    if method == "None":
        smoothed_df = df
    elif method == "MA":
        smoothed_df = pd.rolling_mean(df, arg, min_periods=1)
    elif method == "EWMA":
        smoothed_df = pd.ewma(df, span=arg, ignore_na=1)
    return smoothed_df

def pca_comps(n_comp, df_list):
    '''
    Not suitable for vectorization. Returns "n_comp" number of top PCs for each df in df_list
    :param n_comp:  number of components required
    :param df_list: list of dataframes for whcih PCs are reqd
    :return: n_comp number of top PCs for each df in df_list
    '''
    pca = PCA(n_components = n_comp)
    comp = np.zeros(df_list[0])
    comp_list = [comp for i,_ in enumerate(df_list)]
    for i, df in enumerate(df_list):
        df_s = preprocessing.scale(df)
        fit_pca = pca.fit_transform(df_s)
        comp_list[i] = fit_pca
    return comp_list

def last_valid_value(x):
    '''Utility function intended to return the last valid value in each row of a dataframe which has some values followed by nan in each row
    Use it like this: y= df.apply(last_valid_value, axis=1)'''
    if x.last_valid_index() is None:
        return np.nan
    else:
        return x[x.last_valid_index()]

def resample_keyval(window):
    '''Utility function currently only in for ML class. intended to return the resample freq for a given window.
    Objective is to divide the window into equal parts irrespective of window length, so the resampling freq decreases
    in line with window increase. In most of the methods in ML class, the moves at selected freq (given by this function)
    are used as k-dimensional input vector'''
    resample_dict = {'1D': '35T', '2D': '70T', '3D': '2H', '4D': '3H', '5D': '175T'}
    return resample_dict[window]

def create_window_df(ts, window):
    '''
    For a single timeseries, creates a dataframe where on each day, observation = timeseries over a given window.
    So if you pass a timeseries of 1 observation each day for a year(250 days), and a window = "10D" (10 days), then
    it returns a dataframe where on each day, you have as entry a timeseries of length 10 days ending on that day.
    :param ts: timeseries vector
    :param window: convert window to days or hours. for example, 1week = "7D" or "5D".
    :return: a dataframe with each entry = last "window"
    '''
    int_window = int(window[0:-1]) # just the numeric part
    if window[-1] == "H": # only for forecast horizon < 1D
        ts = ts.resample(window, how='first').dropna()
        ts_moves = ts.diff(1)
        ts_moves.index = [ts_moves.index.date, ts_moves.inedx.time]
        ts_moves = ts_moves.unstack()
        change = np.where( np.isnan (ts_moves.ix[:,0]), ts_moves.ix[:, 1], ts_moves.ix[:, 0])
        change = pd.DataFrame(change, index = ts_moves.index)
        return pd.DataFrame(change.shift(-1))

    if isinstance(ts,pd.core.series.Series):
        x = ts
    else:
        x = ts.ix[:, 0] # convert dataframe to series

    unique_days = np.unique(x.index.date)
    d = {}
    for day in unique_days[int_window]:
        dat_end = str(day)
        iloop = 0
        # make sure that day at the end of the window is available in data. If not, keep looping till a limit (10 days here)
        # and then loop forward to 5 days (try to remove 10 and 5 hardcoding here)
        while day -pd.Timedelta( str(int_window + iloop) + 'D') not in unique_days:
            iloop += 1
            if iloop > 10:
                iloop = -5
        dat_strt = str((day - pd.Timedelta( str (int_window + iloop) + 'D')))
        d[day] = pd.Series(x.loc[dat_strt:dat_end].values)

    x_df = pd.DataFrame.from_dict(d, orient='index')

    return x_df

def change_over_window(df, window):
    '''
    takes a window_df created in "create_window_df" function and just calculates changes over each index (Day)
    :param df: window_df created in "create_window_df" function
    :param window: window used to calculate df
    :return: (last value - first value) for each entry since each entry is a series over the given window
    '''
    if window[-1] == "D":
        x_last = df.apply(last_valid_value, axis=1)
    elif window[-1] == "H":
        x_last = df.resample(window, axis=1, how="first")

    x_first = df.ix[:, 0]
    change = pd.DataFrame((x_last - x_first), index=df.index)

    #if signals == "y":
    #    signals = pd.DataFrame(index = x_df.index, columns =['predictor', 'trend', 'direction', 'strength'])
    #    return x_df, change, signals
    #else:
    #    return change.shift(-1)
    return change

def assign_signals_generic(predictor, trend_indic = False, direction_indic = False):
    '''
    Commonly occuring situation in various signal genration methods is that there is one main series on which strength,
    direction,and/or trending nature of the signals is dependent.
    For example, in lookback direction method, we just look at change over a given window and use that to predict move
    over the next day. So, the move over window acts as our 'predictor' and the direction and strength of signals is
    dependent on predictor. In some other cases, predictor may be related to trending or non trending indicator rather
    than direction.
    :param predictor: primary determinator of signals
    :param trend_indic: whether to use predictor to set the "trend" signal
    :param direction_indic: whether to use the predictor to set the "direction" signal
    :return: signals
    '''
    signals_df = pd.DataFrame(index = predictor.index)
    signals_df['predictor'] = predictor
    signals_df['strength'] = np.abs(predictor)
    if trend_indic:
        signals_df['trend'] = np.sign(predictor)
    elif direction_indic:
        signals_df['direction'] = np.sign(predictor)
    return signals_df

def not_used_assign_signals(input_dict):
    '''Combines all args into a single dataframe intended to be "signals". All inputs are numeric. Input is a dict to
    reduce chance of mixup of labels. To keep it robust to different data structures, each of the arguments can be
    dataframe, series, ndarray or numeric scalar.
    '''
    signals_df = pd.DataFrame()
    for key, value in input_dict.items():
        assert isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, float, int)), key + " datatype may cause issues."
        signals_df[key] = value
    return signals_df
