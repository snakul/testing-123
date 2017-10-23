import matplotlib.pyplot as plt
import pandas as pd

class plotting(object):

    def __init__(self, analyze, run, hist, scat, run_agg):
        self.analyze_plt = analyze
        self.running = run
        self.histogram = hist
        self.scatter = scat
        self.running_agg = run_agg
        if (run == 'y') or (hist == 'y') or (scat == 'y') or (run_agg == 'y'):
            self.analyze_plt = 'y'  # if you want some plots with analysis

    def start_plot(self):

        if self.analyze_plt == 'y':
            if self.running == 'y':
                plt.figure(fig_no)
                plt.title(sym + " " + method_list_names[im] + " running perf")
            if self.histogram == 'y':
                plt.figure(fig_no + 1 * (running == 'y'))
                plt.title(sym + " " + method_list_names[im] + " hist")
            if self.scatter == 'y':
                plt.figure(fig_no + 1 * (running == 'y') + 1 * (histogram == 'y'))
                plt.title(sym + " " + method_list_names[im] + " p_scatter")
            if self.running_agg == "y":
                f = fig_no + 1 * (running == "y") + 1 * (histogram == 'y') + 1 * (scatter == 'y')
                plt.figure(f)
                plt.title(sym + " " + method_list_names[im] + " Agg running perf")
            plt.figure(1)

    def create_plots(self, fig_no, temp, actual, iw, w, l, symbol, method, chartarr):

        temp = pd.DataFrame(temp)
        if "direction" in temp.columns:
            predicted = temp['direction']
        else:
            predicted = temp.ix[:, 0]

        if isinstance(actual, pd.core.series.Series) == False:
            actual = actual.ix[:, 0]

        r, h, s = chartarr

        if r == 'y':
            common_index = predicted.index.intersection(actual.index)
            predicted = predicted[common_index]
            actual = actual[common_index]
            hits = (np.sign(predicted) == np.sign(actual))
            running_perf = pd.rolling_mean(hits, 30).dropna()
            plt.figure(fig_no)
            plt.plot(running_perf.index.to_datetime(), running_perf, label=w)

        if h == 'y':
            plt.figure(fig_no + 1 * (r == 'y'))  # activate this fig
            plt.hist(temp['predictor'], histtype='step', label='w')
            plt.axvline(x=0)

        if s == 'y':
            plt.figure(fig_no + 1 * (r == 'y') + 1 * (h == 'y'))
            plt.subplot(1, 1, iw)
            plt.scatter(temp['predictor'][actual.index], actual)
            plt.axvline(x=0)
            plt.axhline(y=0)
            plt.xlabel(method.__name__ + ' predictor ' + w)
            plt.ylabel('actual ' + symbol)


