import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import indicators as indi
import marketsimcode as msim
from matplotlib.ticker import LinearLocator
from matplotlib.gridspec import GridSpec
from util import get_data


def author():
    return 'cfleisher3'


class ManualStrategy:
    def __init__(self):
        pass

    def author(self):
        return 'cfleisher3'

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=1e5):
        # get adj close data for underlying
        dates = pd.date_range(sd, ed)
        df = indi.ml4t_load_data(['JPM'], dates)
        df_pxs = pd.DataFrame(df['AdjClose'])
        df_metrics = df_pxs.copy()

        # get indicators for adj close data
        ws = [20, 5, 3]
        metrics = [indi.pct_sma, indi.rsi, indi.bollinger]
        standards = [True, False, False]
        for i, metric in enumerate(metrics):
            df_tmp = metric(df_pxs, window_sizes=[ws[i]],
                            standard=standards[i])
            df_metrics = df_metrics.join(df_tmp, how='inner')

        # rules engine
        df_jpm = df_metrics.loc[symbol]
        df_trades = pd.DataFrame(index=df_jpm.index)
        df_feats = df_trades.copy()

        sma_key = f'AdjClose_pct_sma_{ws[0]}'
        df_feats[sma_key] = 0.0
        sma_bcount = 3
        sma_bounds = [-1.5, 1.0]

        sma_lmask = df_jpm[sma_key] < sma_bounds[0]
        sma_umask = df_jpm[sma_key] > sma_bounds[1]
        tmp_sma_lmask = sma_lmask.copy()
        tmp_sma_umask = sma_umask.copy()
        # check for consecutive indicator hits
        for i in range(sma_bcount):
            sma_lmask = sma_lmask & tmp_sma_lmask.shift(i)
            sma_umask = sma_umask & tmp_sma_umask.shift(i)
        df_feats[sma_key][sma_lmask] = -1.0
        df_feats[sma_key][sma_umask] = 1.0

        rsi_key = f'AdjClose_rsi_{ws[1]}'
        df_feats[rsi_key] = 0.0
        rsi_bcount = 3
        rsi_bounds = [0.3, 0.7]

        # adjust rsi scale
        df_jpm[rsi_key] = df_jpm[rsi_key]/100
        rsi_lmask = df_jpm[rsi_key] < rsi_bounds[0]
        rsi_umask = df_jpm[rsi_key] > rsi_bounds[1]
        # check for consecutive indicator hits
        tmp_rsi_lmask = rsi_lmask.copy()
        tmp_rsi_umask = rsi_umask.copy()
        for i in range(rsi_bcount):
            rsi_lmask = rsi_lmask & tmp_rsi_lmask.shift(i)
            rsi_umask = rsi_umask & tmp_rsi_umask.shift(i)
        df_feats[rsi_key][rsi_lmask] = 1.0
        df_feats[rsi_key][rsi_umask] = -1.0

        bolli_key = f'AdjClose_bollinger_{ws[2]}'
        df_feats[bolli_key] = 0.0
        bolli_bcount = 3
        bolli_bounds = [-1.0, 1.5]

        bolli_lmask = df_jpm[bolli_key] < bolli_bounds[0]
        bolli_umask = df_jpm[bolli_key] > bolli_bounds[1]
        tmp_bolli_lmask = bolli_lmask.copy()
        tmp_bolli_umask = bolli_umask.copy()
        for i in range(bolli_bcount):
            bolli_lmask = bolli_lmask & tmp_bolli_lmask.shift(i)
            bolli_umask = bolli_umask & tmp_bolli_umask.shift(i)
        df_feats[bolli_key][bolli_lmask] = -1.0
        df_feats[bolli_key][bolli_umask] = 1.0

        # determine trades based on chg in position
        pos = df_feats.sum(axis=1).rolling(10).sum().clip(-2, 2)
        pos_chg = (pos-pos.shift(1)).fillna(pos.iloc[0])
        df_trades[symbol] = pos_chg*1e3
        return df_trades

    def benchmark_policy(self, symbol, sd=dt.datetime(2008, 1, 1),
                         ed=dt.datetime(2009, 12, 31), sv=1e5, quantity=1e3,
                         commission=0.0, impact=0.0):
        """
            Generates portfolio value over given timeframe. The given quantity
            of the given symbol is acquired at the adjusted close price of the
            given start date assuming the given commissions and market impact.
            The portfolio value consists of the given symbol's market value and
            and residual cash.

            params:
            - symbol: underlying
            - sd: start date
            - ed: end date
            - sv: starting portfolio value
            - quantity: starting quantity of symbol
            - commission: trade commission
            - impact: market impact of trade

            return:
            - pd series with given symbol as name
        """
        # get trading days
        dates = get_data(['SPY'], pd.date_range(sd, ed)).index.values

        # set initial portfolio
        amnts = np.zeros(dates.shape)
        amnts[0] = quantity

        # build trades
        trades = pd.DataFrame(amnts, index=dates, columns=[symbol])

        # compute portfolio value and give column relevant name
        vals = msim.compute_portvals(trades, start_val=sv,
                                     commission=commission, impact=impact)
        vals = vals.rename(symbol)
        return vals

    def get_regions(self, df, bounds, n=3):
        """find regions meeting bounds criteria"""
        umask = bounds
        for i in range(n):
            umask = umask & bounds.shift(-i)
        return umask

    def cmp_benchmark(self, symbol, sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=1e5,
                      bench_quantity=1e3, commission=0.0,
                      impact=0.0, should_plot=False, labels=None):
        df_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        sp = msim.compute_portvals(df_trades, start_val=sv,
                                   commission=commission, impact=impact)
        bp = self.benchmark_policy(symbol, sd=sd, ed=ed, sv=sv,
                                   quantity=bench_quantity,
                                   commission=commission, impact=impact)

        if labels is None:
            labels = ['benchmark', 'strategy']

        # combine policies into single dataframe
        df_cmp = pd.concat([bp, sp], axis=1)
        df_cmp.columns = labels

        # optionally plot comparison
        if should_plot:
            self.plot_strats(df_cmp, labels=labels, trades=df_trades)

        return df_cmp

    def plot_strats(self, df_strat, trades=None, labels=None, idxd=True,
                    ylabel=None, show_legend=True, colors=None, line_alpha=0.7,
                    line_width=1.5, line_style='dotted', tc='0.25',
                    bgc='0.90', fc='0.6'):
        """
            params:
            - df_strat: df with dates and strategy values for each col
            - labels: list of strategy names; if None defaults to col vals
            - idxd: boolean toggle for index values based on start
            - ylabel: y-axis label
            - show_legend: whether to show data legend
            - colors: list of line colors
            - line_alpha: line alpha
            - line_width: line width
            - line_style: grid line style
            - tc: axes tick, axis label, legend color (grayscale)
            - bgc: background color (grayscale)
            - fc: frame color (grayscale)
            - trades: pd trades; draw vertical lines for position chgs if given
        """
        df = df_strat.copy()
        df_pos = trades.cumsum()

        # df strat labels defaults to column values
        if labels is None:
            labels = df.columns.values

        # set default colors if none provided
        if colors is None:
            colors = ['g', 'r', 'b', 'c', 'm', 'k', 'y']

        # optionally index strategy values
        if idxd:
            df = df/df.iloc[0]

        # plot figure with given attributes
        fig = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        gs.update(hspace=0.025)
        ax1 = fig.add_subplot(gs[0])
        ax1.grid(linestyle=line_style)

        for i, strat in enumerate(df.columns.values):
            ax1.plot(df.index, df[strat], color=colors[i],
                     alpha=line_alpha, linewidth=line_width,
                     label=labels[i])

        ax2 = fig.add_subplot(gs[1])
        ax2.grid(linestyle=line_style)
        ax2.plot(df_pos.index, df_pos.values, color='m',
                 alpha=line_alpha, linewidth=line_width)

        # restrict gridlines for x-axis
        ax1.xaxis.set_major_locator(LinearLocator(7))
        ax2.xaxis.set_major_locator(LinearLocator(7))
        ax1.set_xlim(ax2.get_xlim())  # align x-axis subplots

        # format axes
        # x-axis date format
        date_fmt = mdates.DateFormatter('%Y-%m')
        ax1.tick_params(axis='x', which='both', bottom=False,
                        top=False, labelbottom=False)
        ax1.tick_params(axis='y', colors=tc)
        ax1.xaxis.set_major_formatter(date_fmt)
        ax2.tick_params(colors=tc)

        # optional y-axis label
        if ylabel:
            ax1.set_ylabel(ylabel, color=tc)

        # format background and frame colors
        ax1.set_facecolor(bgc)
        ax2.set_facecolor(bgc)
        plt.setp(ax1.spines.values(), color=fc)
        plt.setp(ax2.spines.values(), color=fc)

        # legend with given attributes
        if show_legend:
            leg = ax1.legend()
            for txt in leg.get_texts():
                txt.set_color(tc)

            for lh in leg.legendHandles:
                lh.set_alpha(line_alpha)

        # draw vertical lines for position changes
        if trades is not None:
            buys = (trades > 0.0).values
            idxs = trades.index.values
            for i, v in enumerate(buys):
                if v:
                    ax1.axvline(x=idxs[i], linewidth=1.5,
                                color='b', alpha=0.5)

            sells = (trades < 0.0).values
            for i, v in enumerate(sells):
                if v:
                    ax1.axvline(x=idxs[i], linewidth=1.5,
                                color='k', alpha=0.5)

        plt.show()
        plt.clf()


if __name__ == '__main__':
    ms = ManualStrategy()
    ms.cmp_benchmark('JPM', should_plot=True, commission=9.95,
                     impact=0.001)
