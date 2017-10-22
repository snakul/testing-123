import numpy as np
import pandas as pd
from utilities import *
from sklearn import preprocessing, svm, grid_search, naive_bayes
import math
from scipy import stats
from scipy.linalg import lstsq
import scipy.integrate as integrate
from itertools import product
from scipy import optimize as op
import copy
import datetime
import inspect
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt

'''there are 6 different categories of signal generation methods here:
1. heuristic: Lookback, MA cross, CCRT, MPR
2. variance scaling: var ratio, hurst
3. Ranked correlation: Mann Kendall
4. Parametric: Karatzas
5. Microstructure: PIN
6. ML: Naive Baues, SVR

Each category has a set of signal generation functions, all of which take as arguments a 1D time series and a window size.
Some methods also require other arguments. Almost all methods return a Dataframe named signals, generally with 3 cols: trend, direction, strength.
Trend classifies series as "trending/ non trending" at the time. If trending, mostly the change in direction over window is taken to be direction prediction. 
In most cases, the strength is the abs value of change over the current window

Note- signal on a given data makes use of data up to and including the day, hence it is the signal for next day's move
'''

class Heuristic:
    '''no assumptions regarding distribution of returns'''
    def lookback_direction(self, time_series, window ='1D'):
        '''dumb method- predicts same change for the next day as for the window
        so if window =2D, it predicts that the move direction tomorrow = net move over last 2 days'''
        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        signals = assign_signals_generic(change, direction_indic=True)
        signals['trend'] = 1 # always trending

        return signals

    def ma_cross(self, time_series, short_window, long_window): # needs more work
        '''returns the df of symbols containing the signals to go long, short or hold(1, -1, 0)'''

        '''PENDING'''
        # create the set of short and long simple moving averages over the respective periods
        short_mavg = pd.rolling_mean(time_series, short_window, min_periods=1)
        long_mavg = pd.rolling_mean(time_series, long_window, min_periods=1)

        # create a signal (invested or not invested) when the short MA crosses the long MA, but only for the period greater than the shortest MA window
        signals = assign_signals_generic(short_mavg - long_mavg, direction_indic = True)
        signals['trend'] = 1  # always trending
        signals['signal'] = np.where(signals['direction'][short_window:] > 0, 1.0, 0.0)
        return signals

    def ccrt(self, time_series, window='1D', max_lag=25, use_probability ='T'):
        '''
        Counting continuation and reversal times. Calculates empirical freq/probability of movement in same vs opposite
        direction in successive steps. Uses different step sizes and averages over time.
        Similarity to others - compares data to a random walk.
        Difference from others - no strict assumptions on the time series, thus more heuristic.
        I simply compare empirical probabilites without reference to what would be implied by a random walk
        This method is also closely related to Kendall's Tau ( a ranked correlation measure- read on wiki). This might be
        better for the purpose of trend/reversal but need to investigate more. A test case is where a small series (say 30 observations)
        :param use_probability: for each lag, use probability (relevant counts/total counts) rather than using freq (relevant counts)
        :return: if P(continue) > P(reversal) then trend else reversal
        '''
        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        n = x_df.shape[0]
        pc = np.zeros((n, max_lag))
        pr = np.zeors((n, max_lag))
        lags = np.arange(2, max_lag + 1)
        for lag in lags:
            dx = np.asarray(x_df.diff(lag, axis=1))
            ddx = np.roll(dx, -lag)
            track = np.sign(dx) * np.sign(ddx)
            # to check whether each succesive step is continuation or reversal, you can just multiply signs of successive changes and see if its positive or negative
            diff = np.nansum(track, axis=1) # for each window - gives diff of number of continuation vs reversal # ignores nan
            summ = (1 - np.isnan(track)).sum(1) # for each window - gives sum of number of continuation and reversal # ignores nan
            if use_probability == 'T':
                diff = diff/summ # p(c) - p(r) = (# continuations - # reversals)/(# continuations + # reversals)
                summ = 1 # p(c) + p(r) = 1
            pc[:, lag-1] = (summ + diff)/2.0
            pr[:, lag-1] = (summ - diff)/2.0
        mc = np.mean(pc,1)
        mr = np.mean(pr,1)
        signals = assign_signals_generic(mc - mr, trend_indic =True)  # if mc > mr: "trending" else : "mean reverting"
        signals['direction'] = pd.concat([signals['trend'], np.sign(change)], axis=1).product(axis=1)
        return signals

    def mpr(self, time_series, window='1D', max_lag=25):
        '''modified push response- similar to ccrt- no assumption on time series, but includes a regression on modified data thus making it parametric.
        Unlike ccrt,the strength of the trend (continuation or reversal) is also taken into account'''
        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        n = x_df.shape[0] # special handling needed for n<10 not implemented here, see http://vsp.psnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm
        # subjective, if I am looking at n day history daily, I think it makes sense to keep the max lag in that vicinity
        lags= np.arange(1, max_lag+1)
        resp_low = np.zeros((n, max_lag))
        resp_med = np.zeros((n, max_lag))
        resp_high = np.zeros((n, max_lag))
        push_low = np.zeros((n, max_lag))
        push_med = np.zeros((n, max_lag))
        push_high = np.zeros((n, max_lag))
        '''In DB paper, they regress responses across %ile buckets against push to estimate beta, which doesnt make sense.
        In original paper, they clearly show that the response for different push values (across lags) differs
        So i dont regress, I take 3 buckets (low, med, high %ile) and take mean response across different lags
        Also, in original paper, the push and response both are measured symmetrically, so if the push is change in px over
        x mins, response is the change in following x mins. Since I am only looking for next day's prediction, I am keeping the'''
        for lag in lags: #lag starts from 1
            dx = np.asarray(x_df.diff(lag, axis=1))
            dxm, dxs = np.nanmean(dx, axis=1), np.nanstd(dx, axis=1)
            zx = (dx - dxm[:,None]) / dxs[:,None] #since we are demeaning here, sum of all =0

            low = np.ceil(np.nanpercentile(zx, 25, axis=1) * 100)/100
            med = np.ceil(np.nanpercentile(zx, 75, axis=1) * 100)/100
            zxq = np.roll(zx , -lag, axis=1)

            push = copy.deepcopy(zx)
            resp = copy.deepcopy(zxq)
            push[(push < low[:, None]) | (push > med[:, None])] = np.nan
            resp[(push < low[:, None]) | (push > med[:, None])] = np.nan
            push_med[:, lag-1] = np.nanmean(push, axis=1)
            resp_med[:, lag-1] = np.nanmean(resp, axis=1)

        av_push_low = push_low.mean(axis=1)
        av_push_med = push_med.mean(axis=1)
        av_push_high = push_high.mean(axis=1)
        av_resp_low = resp_low.mean(axis=1) # mean response across lag buckets when push = low, step 3
        av_resp_med = resp_med.mean(axis=1)
        av_resp_high = resp_high.mean(axis=1)
        ''' can plot scatterplots of av_push_low vs av_resp_low etc to see generic relation'''
        # step 4
        av_pushresponse = ((av_resp_low / av_push_low) + (av_resp_med / av_push_med) + (av_resp_high / av_push_high))/3.0 # av of av response in each bucket
        signals = assign_signals_generic(av_pushresponse, trend_indic = True)
        signals['direction'] = pd.concat([signals['trend'], np.sign(change)], axis=1).product(axis=1)
        return signals

class Var_scale:
    '''variance scaling methods
    non-parametric methods focusing on how fractal the variability of the asset is and then compare price action to rand walk'''
    def variance_ratio(self, time_series, window='1D', q=15.0):
        '''if px is random walk, then variance of returns increases over time linearly. The var of its q-differences is q times var of its first difference.
        ref paper- Lo, Mackinlay 1988. Stock market prices do not follow random walks: Evidence from a specification test'''
        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        w = x_df.apply(lambda x: x.last_valid_index(), axis=1) # w is different for each row
        n = (w -1.0) / q # on page 46 of ref paper, it says number of observations = nq+1 => n = (#obs in window - 1)/q
        mu = (x_df.diff(1, axis=1)).mean(axis=1) # eqn 8a in ref paper, takes care of different n for different rows
        var1 = ((x_df.diff(1, axis=1).subtract(mu, axis=0)) ** 2).mean(axis=1) # eqn 8b of paper
        varq = ((x_df.diff(q, axis=1).subtract(q*mu, axis=0)) ** 2).mean(axis=1) # eqn 10 of paper
        vr_series = varq / (var1 * q) # if VR<1, likely to mean revert, if VR>1 likely to trend, if VR ==1 random walk
        zq = np.sqrt(n*q) * (vr_series -1) # ~N(0, 2(2q-1)(q-1)/3q # eqn 14b of paper, this is a time series of zq
        dist_var = 2.0 * (2*q -1) * (q-1)/ (3*q)
        alpha = 5.0 /100 # prob of type 1 error in z test
        k = stats.norm.ppf(1 - alpha, loc=0, scale=np.sqrt(dist_var)) # cdf(k) = 1- alpha =>> stats.norm.cdf(k, scale=dist_var) = 1- alpha

        signals = assign_signals_generic(zq, trend_indic=True)
        signals['trend'] = np.where(zq>k, 1.0, np.where(zq < -k, -1, 0)) # if zq > -k: trending else not
        signals['direction'] = pd.concat([signals['trend'], np.sign(change)], axis=1).product(axis=1)
        return signals

    def hurst_exponent(self, time_series, window='1D', max_blocks=10):
        """measure of long term memory of time series. Relates to the autocorr of series, and the rate at which these decrease as the lag
        between pairs of values increases. The Joseph effect tells whether movements in a time series are part of a long term trend and
        refers to the Old testament where Egypt would experience seven years of rich harvest followed by 7 years of famine. The Noah effect
        is the tendency pf a time series to have abrupt changes and the name is derived from the biblical story of the great flood. Both
        these effects in a time series can be inferref from the Hurst exponent.

        Hurst exponent is not so much calculated as it is estimated. A variety of techniques exist for estimating it. Accuracy of various
        estimation techniques can vary - see http://www.r-bloggers.com/exploring-the-market-with-hurst/

        Hurst exponent applies to data sets that are statistically self similar, which means that the statistical properties for the entire
        data set are the same for sub-sections of the data set. for example, the halves of the data set have the same statistical properties as the entire data set.

        more info : http://www.bearcave.com/mis1/mis1_tech/wavelets/hurst/index.html#Why

        This function implements the H_d, p method of estimation with double bias correction as suggested in "Estimation of Hurst exponent revisited by Mielniczuka"""
        def rs(x): # using the method in paper
            y = np.cumsum(x)
            s = np.std(x)
            n = len(x)
            index = np.arange(n)
            deviations = y - index * (y[-1]/n)
            rs_stat = (1.0 / s) * (max(deviations) - min(deviations))
            return rs_stat

        def bias_correction (h):
            return -0.618 * h + 0.5597 # page 8/16 of the paper

        time_series1 = np.log2(time_series)
        time_series2 = time_series1.diff(1)
        x_df, change, signals = create_window_df(time_series2, window)

        n_series = x_df.apply(lambda x: x.last_valid_index(), axis=1) # each day will have variable length of data points
        x_df =  x_df[n_series > max_blocks]
        change = change[n_series > max_blocks]
        # since n for each row is different, I have kept the max value of (n/k) = max blocks same for all rows
        # so size of blocks (ki) for each row varies from k to n (where k and n are different for all rows)
        # since there is no averaging, for each ki, we'll have n/ki values of R/S. Max value of n/ki is max_blacks param
        no_of_rs= max_blocks * (max_blocks + 1) / 2.0 # sum(n) = n(n+1)/2 gives the number of rs values to calculate  for each row/series
        k_values = np.zeros((len(n_series), no_of_rs)) # block size = k(min 8 here) , no of blocks of size k = n/k
        rs_values = np.zeros((len(n_series), no_of_rs)) # for each t, there will be a different RS

        for irow , n in enumerate(n_series): # n = length of available datapoints on a given day
            position = 0
            for ib, blocks in enumerate(np.arange(1, max_blocks+1)): # gives the no of blocks to divide the series (lookback each day) into
                block_size = math.floor(n / blocks) # will be different for different rows as n is different for all rows
                x_2d = x_df.iloc[irow, 0:(blocks * block_size)].reshape(blocks, block_size)
                rs_1d = np.apply_along_axis(rs, 1, x_2d) # returns RS statistic for each block
                rs_values[irow, position : position + blocks] = rs_1d
                k_values[irow, position: position + blocks] = block_size
                position = position + blocks

        rs = np.log2(rs_values ) # math.log doesnt work on arrays
        k = np.log2(k_values)

        hurst = np.einsum('ij,ij -> i', rs, k) / np.einsum('ij,ij -> i', k, k)
        hurst_single_bias = np.apply_along_axis(bias_correction, 0, hurst)
        hurst_double_bias = np.apply_along_axis(bias_correction, 0, hurst_single_bias)

        print (time_series.shape, len(hurst), '%.3f' % max(hurst), '%.3f' % min(hurst))

        signals = assign_signals_generic(hurst_double_bias - 0.5)
        signals['trend'] = np.where(hurst_double_bias > 0.55, 1, np.where (hurst_double_bias < 0.45, -1, 0))
        signals['direction'] = pd.concat([signals['trend'], np.sign(change)], axis=1).product(axis=1)
        return signals

class Ranked_corr:
    '''Ranked correlation method- focus on sign but not magnitude of sequential moves'''
    def mann_kendall(self, time_series,window ='1D', alpha =0.05):
        '''Mann Kendall is a non parametric trend test and as such, it is not dependant upon:
        * The magnitude of data,
        * Assumptions of distributions
        * Missing data or
        * irregularly spaced monitoring periods
        A monotonic trend means that the variable consistently increases (decreases) through time, but the trend may or may not be linear.
        Thhe MK test can be used in place of a parametric linear regression analysis, which can be used to test if the slope of estimated linear reg line <>0.
        The regression analysis requires that residuals from fitted regression line be normally distributed; an assumption not reqd by the MK test

        Assumptions:
            When no trend is present, the measurements over time ar iid. So not serially correlated.
            The observations are representative of the true conditions at sampling times.
        Input:
            x: a vector of data
            alpha: significance level (0.05 default)
        Output:
            trend: tells the trend(increasing, decreasing or no trend)
            h: True(trend present) or False
            p: p value of significance test
            z: normalized test statistic
        '''
        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        n,w = x_df.shape # special handling meeded for n<10 not implemented here, see http://vsp.pnnl.gov/help/Vsample/Design_Trend_Mann_Kendall.htm
        x = x_df.as_matrix()
        # calculate s- sum of signals of all nc2 pairs of differences xj-xk (j>k). Gives you number of positive differences minus number of negative differences.
        # If s>0, observations obtained later in time tend to be larger than observations made earlier and vice versa
        s = np.zeros((n,w))
        for lag in np.arange(w):
            temp = (x - np.roll(x,lag+1,axis=1))[:,lag+1:]
            s[:,lag] = np.sum(np.sign(temp), axis=1) # no of increases - decreases for given lag
        s_av = np.mean(s, axis=1)
        s_std = math.sqrt(w*(w-1)*(2*w+5)/18)
        # calculate MK test stat z
        z = np.zeros_like(s_av)
        z[s_av > 0] = (s_av[s_av > 0] - 1) / s_std
        z[s_av < 0] = (s_av[s_av < 0] + 1) / s_std
        # if s >0: z = (s-1)/std_s elif s==0: z=0 elif s<0: z = (s+1)/ std_s
        # calculate the p value = 2*(1- stats.norm.cdf(abs(z))) # two tail test
        h = abs(z) >  stats.norm.ppf(1 - alpha / 2)
        # if z<0 and h=1: trend decreasing elif z>0 and h=1: trend increasing , else fail to reject null of no trend
        signals = assign_signals_generic(z, direction_indic= True)
        signals['trend'] = np.where(h, 1.0, 0)
        return signals

class Parametric:
    ''' parametric changepoint- detect changes to the params that govern a time series. modelling of switch (stopping time) is a part of exercise'''
    def karatzas(self, time_series, window='1D', roll_length =30):
        '''Based on Karatzas changepoint method as outlined in this paper : http:/www.math.columbia.edu/~ik/detect.pdf
        I assume that price follows a GBM, and thus its log transformation follows ABM and the method in the paper becomes applicable
        Assume const vol
        Values of lambda and theta in paper are assumed given. I am optimizing for them in backtest
        Return: an estimate of the changepoint t(assumed to be a trend signal. If no changepoint then series has no trend)'''
        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        x_orig = x_df.as_matrix()
        x_orig = x_orig - x_orig[:,0][:,None] # we subtract the first element in each row as the hypothesis is that drift starts at 0 and maybe jumps to theta at some point
        x_std = np.nanstd(x_orig, axis=1)
        x_norm = x_orig/(x_std[:, None])
        w = np.linspace(0, 10, x_norm.shape[1])
        actual = create_window_df(time_series, '1D', 'n')

        def core_function(x_array, x, chg="", act="", call_reason=""):
            '''theta is the new drift of the BM after changepoint (assuming zero before that)
            lambda is the parameter of the exponential dist followed by changepoint, multiplying it by x_std to keep it of the same order of theta
            theta, lambda, delta all have to be of similar order. If you just make theta dependent on stdev without doing anything abt lambda,
            and if stdev is tiny, then delta is going to be huge and (1-z)**(2_delta) ~ inf. So as sugested in paper, standardize x so that stdev is 1 in each series'''
            theta, lamb, p = x_array[0], x_array[1], x_array[2]
            delt = 2 * lamb / (theta ** 2)
            def f(z):
                nmrtr = (1-2*z) * np.exp(-delt/z)
                dnmntr =  ((1-z)**(2+delt)) * (z**(2-delt))
                return nmrtr/dnmntr

            lhs = integrate.quad(f, 0, 0.5)[0]
            def integral(p):
                return integrate.quad(f, 0.5, p)[0] + lhs

            p_star = op.fsolve(integral, 0.75)[0] # root or fsolve?
            main = np.exp(-theta * x - ((lamb - 0.5 * (theta **2)) * w))
            rhs = main * p_star / (1 - p_star) # eqn 5.9 of paper
            cumsum_t = np.cumsum(main, axis=1)
            lhs = p / (1 - p) + (lamb * cumsum_t)
            t_solution = np.argmax( lhs > rhs, 1) + np.any( lhs > rhs, 1) # using argmax only returns zero in 2 cases : 1) the first value in each row > default 2) no value in row > default

            if call_reason == "optimize":
                trend = t_solution > 0
                direction = trend * np.sign( np.asarray(chg).ravel())
                miss_rate = np.sum(direction != np.sign( np.asarray(act).ravel() )) / float(len(direction))
                return miss_rate
            else:
                return t_solution

        detect_times = []
        for iday,_ in enumerate(x_df.index[roll_length +1 :], start = roll_length +1):
            # variables to be optimized:
            t, l, p = 1, 0.5, 0.5
            # thresh = 5 # every week/day/hour you look back and determine the time where change probably occured. If this time is within a given threshold of current time, you treat it as a trading signal
            x = x_norm[iday - (roll_length + 1) : iday - 1]
            chg = change.ix[iday - (roll_length + 1) : iday - 1]
            act = actual.ix[iday - (roll_length + 1) : iday - 1]
            res = op.minimize( core_function, np.asarray(([t, 1, p])), args=(x, chg, act, "optimize") )
            x_next = x_norm[iday].reshape(1, len(x_norm[iday]))
            detect_times.append( core_function(res.x, x_next)[0] )
            '''in case optimization is not working, we can do our own
            # s= np.nanstd(time_series)
            t = np.linspace(-1, 1, 10)
            l = np.linspace(0.1, 1, 10) # mean time of change = 1/lambda
            p = np.linspace(0, 1, 10)
            arr = list(product(t,1,p))
            miss = []
            for i, arr_i in enumerate(arr):
                miss.append( core_function(arr_i, x, chg, act, "optimize"))
            min_index = np.argmin(np.array(miss))
            arr_final = arr[min_index]
            placeholder = core_function(arr_final)
            '''
        signals = assign_signals_generic(detect_times, direction_indic= True)
        signals['trend'] = np.where( detect_times > 0, 1.0, 0)
        return signals

class Microstructure:
    def pin(self, time_series, window='1D', threshold=0.5):
        '''probability of informed trading as described in Ohara paper "factoring information into returns"
        it uses tick data but i am treating minutely data as tick data
        it requires classifying every trade as buyer or seller initiated and i am not using lee-ready algo but only marked move oer 1 min
        need to see how PIN varies over time for different currencies and what pin values can influence movements (maybe pin = 30% is good enough and not the intuitive 50%)
        classify pin > threshold to be trending and otherwise as mean reverting'''
        def log_likelihood (arg_tuple, s,b,m):
            delta, mu, eb, es, alpha = arg_tuple
            xs = es / (mu + es)
            xb = eb / (mu + eb)
            f1 = alpha * (1 - delta) * np.exp(-1 * mu) * (xs ** (s - m)) * (xb ** (-m))
            f2 = alpha * delta * np.exp(-1 * mu) * (xb ** (b - m)) * (xb ** (-m))
            f3 = (1 - alpha) * (xs ** (s - m)) * (xb ** (b - m))
            l = (-eb - es) * len(s) + (np.log(xb) + np.log(xs)) * m + np.log(mu + xb) * b + np.log(mu + xs) * s + np.log(f1 + f2 + f3)
            return -l.sum()

        x_df = create_window_df(time_series, window)
        change = change_over_window(x_df, window)
        moves = time_series.shift(-1) - time_series
        moves[moves > 0] = 'b'
        moves[moves < 0] = 's'
        moves.index = moves.index.date
        bt = moves[moves == "b"].groupby(moves[moves == 'b'].index).count()
        st = moves[moves == "s"].groupby(moves[moves == 's'].index).count()
        common_index = x_df.index.intersection(st.index)
        bt = bt.ix[common_index]
        st = st.ix[common_index]
        mt = pd.concat([bt, st], axis=1).max(axis=1) # using definition of mt as in paper leads to computational issues. Since
        # Mt is a frivolous variable used only for computational conveninece in the paper, I am using a different version here
        delta = 0.5 # prob bad news
        mu = 0.5 # daily arrival rate of orderts from informed traders
        eb = 0.5 # daily arrival rate of buy orders from uninformed traders
        es = 0.5 # daily arrival rate of sell orders from uninformed traders
        alpha = 0.5 # probability that an information event occurs
        w = int(window[0 : -1])
        pin = []

        for iday in np.arange(len(st)):
            sts = st.iloc[iday : iday + w].ix[: , 0]
            bts = bt.iloc[iday : iday + w].ix[: , 0]
            mts = mt.iloc[iday : iday + w]
            res = op.minimize(log_likelihood, (delta, mu, eb, es, alpha) , args = (st,bt, mt) , bounds = ((0,1), (0, None), (0, None), (0, None), (0,1) )).x
            d, m, b, s, a = res[0], res[1], res[2], res[3], res[4]
            pin.append( a * m / ( a * m + b + s ))

        pin = pd.DataFrame(pin, index = st.index)
        signals['predictor'] = pin
        signals['trend'] = np.where( pin > threshold, 1.0, -1.0)
        signals['direction'] = pd.concat([signals['trend'], np.sign(change)], axis=1).product(axis=1)
        signals['strength'] = np.abs(change) # not too sure what the strength should be here
        return signals

class ML:
    '''common machine learning based, these are also parametric but dont focus on detecting changepoints'''
    def nbayes(self, time_series, window='1D', type='gaussian', roll_length= 30):
        '''uses the k dimension input vector of moves at given freq (see resample function) to classify series as "move up next day" or "move down" '''
        copy_ts = time_series.copy()
        copy_ts.index = [copy_ts.index.date, copy_ts.index.time]
        copy_ts = copy_ts.unstack() # results in a dataframe with rows = days and columns = times
        new_cols = [datetime.timedelta(hours = x[1].hour, minutes = x[1].minute, seconds = x[1].second, microseconds = x[1].microsecond) for x in copy_ts.columns]
        copy_ts.columns = new_cols
        copy_ts.interpolate( method = 'linear', axis =1, inplace = True)
        minute_moves = copy_ts.diff(1, axis =1)
        cum_moves = minute_moves.cumsum( axis=1, skipna=True)
        sampled_df = cum_moves.T.resample(resample_keyval(window), how='first').T
        features = sampled_df.diff(1, axis=1).ix[:,1:]
        d = {}

        '''note if window =5 days, day starts at 6th day in the series, d[day] has data from last 5 days EXCLUDING  the 6th day'''
        int_window = int(window[0 : -1])

        for iday, day in enumerate( features.index[int_window : ], start = int_window):
            data = features.iloc[iday - int_window : iday]
            items = data.shape[0] * data.shape[1]
            d[day] = pd.Series( data.values.reshape(items, 1).ravel() )

        x_df = pd.DataFrame.from_dict(d, orient='index') # for a given day, it has data from last window period exlcuding current day
        y_df = minute_moves.sum(axis=1)
        x_df.fillna( method='bfill', axis=1, inplace=True )
        x_df.fillna( method='ffill', axis=1, inplace=True )
        x_df.dropna() # svr doesnt work with na
        y_df = y_df[x_df.index]

        if type == "gaussian":
            clf = naive_bayes.GaussianNB()
        elif type == "bernoulli": # instead of using moves each frequency as input vector, use the direction of moves only
            clf = naive_bayes.BernoulliNB()
            x_df = np.sign(x_df)

        prediction = []
        for iday,_ in enumerate( x_df.index[roll_length + 1 : ], start = roll_length + 1): # start with day 30 and run rolling 30 day svr
            clf.fit( x_df.ix[iday - (roll_length + 1) : iday - 1], np.sign(y_df[iday - (roll_length + 1) : iday - 1]) )
            prediction.append( clf.predict( x_df.ix[iday] )[0] )
        prediction = np.asarray(prediction)
        signals = pd.DataFrame( index = x_df.index, columns = ['trend', 'direction', 'strength', 'predictor'])
        signals['trend'] = 1 # always trending
        signals['direction'].ix[roll_length + 1 :] = prediction
        signals['strength'].ix[roll_length + 1:] = prediction
        signals['predictor'].ix[roll_length + 1:] = prediction

        return signals

    def svr(self, time_series, window='1D', roll_length=30):
        '''support vector regression
        For predicting next day return or claffifying current day as trending or mean reverting.
        For each window selected, I need to create feature vectors using time series over that window.
        The feature vectors are returns over different frequencies.
        The number of feature vectors is same for any chosen window. So lets say number of feature vectors =10, and window = '5D' ,
        then i need to use the time series over last 5 days to calcualte returns at 10 approx equidistant points.
        For this I can use 2 approaches:
        1) Take the first, middle and last value on each day and compute 2 returns (first half and second half of the day).
            This will give me a total of 10 returns over 5 days as required. I have an open question on stackoverflow on how to do it without using loops:
            http://stackoverflow.com/questions/42638734/pandas-selecting-sampling-at-different-interval-frequencies
        2) Resample the time series at a given freq so lets say '150T', then use the values over last 5 days.
            This is troublesome as
            a) resampling approach might not lead to same number of data points for each day(try it)
            b) if you try to counter the unequal number of sampled points by just taking the last 10  returns, then you might over or undershoot the window selected.
        '''

        copy_ts = time_series.copy()
        copy_ts.index = [copy_ts.index.date, copy_ts.index.time]
        copy_ts = copy_ts.unstack() # results in a dataframe with rows = days and columns = times
        new_cols = [datetime.timedelta(hours = x[1].hour, minutes = x[1].minute, seconds = x[1].second, microseconds = x[1].microsecond) for x in copy_ts.columns]
        copy_ts.columns = new_cols
        copy_ts.interpolate(method = 'linear', axis = 1, inplace = True)
        minute_moves = copy_ts.diff(1, axis = 1)
        cum_moves = minute_moves.cumsum(axis =1, skipna = True)
        sampled_df = cum_moves.T.resample(resample_keyval(window), how ='first').T
        features = sampled_df.diff(1, axis=1).ix[:, 1:]
        d = {}
        '''NOTE : if window = 5d, day starts at 6th day in the series, d[day] has data from last 5 days EXCLUDING 6th day'''
        int_window = int(window[0 : -1])
        for iday, day in enumerate(features.index[int_window : ], start = int_window):
            data = features.iloc[iday - int_window : iday]
            items = np.prod(data.shape)
            d[day] = pd.Series(data.values.reshape(items, 1).ravel())

        x_df = pd.DataFrame.from_dict(d, orient = 'index') # for a given day, it has data from lat window period excluding current day
        y_df = cum_moves.apply(last_valid_value, axis = 1) # this has the move on the current day
        x_df.fillna(method = 'bfill', axis = 1, inplace = True)
        x_df.fillna(method = 'ffill', axis = 1, inplace = True)
        x_df = x_df.dropna() # svr doesnt work with na
        y_df = y_df.dropna()
        common_index = x_df.index.intersection(y_df.inex)
        y_df = y_df[common_index]
        x_df = x_df.loc[common_index]

        parameters = {'C': np.logspace(5e-4, 0, 100), 'gamma' : np.logspace(5e-3, 0.5, 100)}
        svr = svm.SVR()
        clf = grid_search.GridSearchCV(svr, parameters) # default kernel is rbf, default gamma (kernel coeff) is 1/no of features
        # following points from http://scikit-learn.org/stable/modules/svm.html#svm-regression
            #recommend to standardize the data as SVM algo are not scale invariant
            # proper choice of C and gamma is critical to the SVM's performance. One is advised to use sklearn.model_selection.GridSearchCV with C and gamma spaced exponentially far apart to choose good values
        prediction = []
        for iday,_ in enumerate(x_df.index[roll_length : ], start = roll_length): # start with day 30 and run rolling 30 day svr
            clf.fit(x_df.ix[iday - (roll_length) : iday], y_df[iday - (roll_length) : iday])
            prediction.append(clf.predict(x_df.ix[iday])[0]) # latest predicton
        prediction = np.asarray(prediction)
        signals = pd.DataFrame(index = x_df.index[roll_length : ], columns = ['trend', 'direction', 'strength', 'predictor'])
        signals = assign_signals_generic(prediction, direction_indic=True)
        signals['trend'] = 1 # always trending

        return signals

