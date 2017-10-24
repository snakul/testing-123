import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class plotting(object):

    def __init__(self, run, hist, scat, run_agg, fig_no):
        self.running = run
        self.histogram = hist
        self.scatter = scat
        self.running_agg = run_agg
        self.fig_no = fig_no
        if run + hist + scat + run_agg:
            self.analyze_plt = 'y'  # if you want some plots with analysis
        else:
            self.analyze_plt = 'n'

    def start_plot(self):

        if self.analyze_plt == 'y':
            if self.running:
                plt.figure(self.fig_no)
                plt.title(sym + " " + method_list_names[im] + " running perf")
            if self.histogram:
                plt.figure(self.fig_no + 1 * (self.running == 'y'))
                plt.title(sym + " " + method_list_names[im] + " hist")
            if self.scatter:
                plt.figure(self.fig_no + 1 * (self.running == 'y') + 1 * (self.histogram == 'y'))
                plt.title(sym + " " + method_list_names[im] + " p_scatter")
            if self.running_agg:
                f = self.fig_no + 1 * (self.running == "y") + 1 * (self.histogram == 'y') + 1 * (self.scatter == 'y')
                plt.figure(f)
                plt.title(sym + " " + method_list_names[im] + " Agg running perf")
            plt.figure(1)

    def create_signal_plots(self):#, predict, rlzd, iw, w, symbol, method):

        predict = pd.DataFrame(predict)
        if "direction" in predict.columns:
            predicted = predict['direction']
        else:
            predicted = predict.ix[:, 0]

        if isinstance(rlzd, pd.core.series.Series) == False:
            rlzd = rlzd.ix[:, 0]

        if self.running:
            common_index = predicted.index.intersection(rlzd.index)
            predicted = predicted[common_index]
            actual = rlzd[common_index]
            hits = (np.sign(predicted) == np.sign(actual))
            running_perf = pd.rolling_mean(hits, 30).dropna()
            plt.figure(self.fig_no)
            plt.plot(running_perf.index.to_datetime(), running_perf, label=w)

        if self.histogram:
            plt.figure(self.fig_no + 1 * (self.running == 'y'))  # activate this fig
            plt.hist(temp['predictor'], histtype='step', label='w')
            plt.axvline(x=0)

        if self.scatter:
            plt.figure(self.fig_no + self.running + self.histogram)
            plt.subplot(1, 1, iw)
            plt.scatter(temp['predictor'][rlzd.index], rlzd)
            plt.axvline(x=0)
            plt.axhline(y=0)
            plt.xlabel(method.__name__ + ' predictor ' + w)
            plt.ylabel('actual ' + symbol)

    def create_agg_plots(self):  # , predict, rlzd, w):

        predict = pd.DataFrame(predict)
        if "direction" in predict.columns:
            predicted = predict['direction']
        else:
            predicted = predict.ix[:, 0]

        if isinstance(rlzd, pd.core.series.Series) == False:
            actual = rlzd.ix[:, 0]

        common_index = predicted.index.intersection(rlzd.index)
        predicted = predicted[common_index]
        actual = rlzd[common_index]
        hits = (np.sign(predicted) == np.sign(actual))
        running_perf = pd.rolling_mean(hits, 30).dropna()
        plt.figure(self.fig_no)
        plt.plot(running_perf.index.to_datetime(), running_perf, label=w)


    def activate_next(self):
        if self.running == 'y':
            plt.figure(self.fig_no)
            plt.legend(loc = 'best')
        if self.histogram == 'y':
            plt.figure(self.fig_no + self.running)
            plt.legend(loc = 'best')
        if self.running_agg == 'y':
            plt.figure(f)
            plt.legend(loc = 'best')
        self.fig_no = self.fig_no + self.running + self.histogram + self.scatter + self.running_agg


