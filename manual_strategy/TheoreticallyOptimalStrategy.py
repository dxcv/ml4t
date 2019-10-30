import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import marketsimcode as ms
from matplotlib.ticker import LinearLocator
from util import get_data


class TheoreticallyOptimalStrategy:
    def __init__(self):
        pass

    def cmp_benchmark(self, symbol, sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=1e5,
                      bench_quantity=1e3, commission=0.0,
                      impact=0.0, should_plot=False, labels=None):
        tp_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        tp = ms.compute_portvals(tp_trades, start_val=sv,
                                 commission=commission, impact=impact)
        bp = self.benchmark_policy(symbol, sd=sd, ed=ed, sv=sv,
                                   quantity=bench_quantity,
                                   commission=commission, impact=impact)

        if labels is None:
            labels = ['benchmark', 'optimal']

        # combine policies into single dataframe
        df_cmp = pd.concat([bp, tp], axis=1)
        df_cmp.columns = labels

        # optionally plot comparison
        if should_plot:
            self.plot_strats(df_cmp, labels=labels)

        return (tp, bp)

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=1e6):
        """
            Legal daily trade values are as follows:
                * +1000 (BUY 1000 shares)
                * -1000 (SELL 1000 shares)
                * +2000 (BUY 2000 shares if short 1000)
                * -2000 (SELL 2000 shares if long 1000)
            params:
            - symbol: stock symbol to act on
            - sd: datetime start date
            - ed: datetime end date
            - sv: starting portfolio value
            return:
            - df_trades: data frame whose values represent daily trades
        """
        # get prices for underlying
        pxs = get_data([symbol], pd.date_range(sd, ed))
        pxs = pxs[symbol]
        pxchgs = pxs/pxs.shift(1)-1

        # prior day ups and downs
        pups = pxchgs > 0.0
        pdwns = ~pups

        # forward day ups and downs
        fups = pups.shift(-1).fillna(False)
        fdwns = pdwns.shift(-1).fillna(False)

        # set buys to max shares
        buys = fups & pdwns
        buys[0] = fups[1]
        buys[buys] = 2000

        # set sells to max shares
        sells = ~fups & pups
        sells[0] = fdwns[1]
        sells[sells] = -2000

        # consolidate trades and cvt first to 1000 lot
        trades = buys+sells
        tidx = trades.nonzero()[0][0]
        trades[tidx] = trades[tidx]/2

        return pd.DataFrame(trades)

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
        vals = ms.compute_portvals(trades, start_val=sv, commission=commission,
                                   impact=impact)
        vals = vals.rename(symbol)
        return vals

    def plot_strats(self, df_strat, labels=None, idxd=True, ylabel=None,
                    show_legend=True, colors=None, line_alpha=0.7,
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
        """
        df = df_strat.copy()

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
        ax = fig.add_subplot(111)
        for i, strat in enumerate(df.columns.values):
            ax.plot(df.index, df[strat], color=colors[i],
                    alpha=line_alpha, linewidth=line_width,
                    label=labels[i])

        # format axes
        # x-axis date format
        date_fmt = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_fmt)
        # axes tick colors
        ax.tick_params(colors=tc)

        # restrict gridlines for x-axis
        ax.grid(linestyle=line_style)
        ax.xaxis.set_major_locator(LinearLocator(7))

        # optional y-axis label
        if ylabel:
            ax.set_ylabel(ylabel, color=tc)

        # format background and frame colors
        ax.set_facecolor(bgc)
        plt.setp(ax.spines.values(), color=fc)

        # legend with given attributes
        if show_legend:
            leg = ax.legend()
            for txt in leg.get_texts():
                txt.set_color(tc)

            for lh in leg.legendHandles:
                lh.set_alpha(line_alpha)

        plt.show()
        plt.clf()


if __name__ == '__main__':
    opt_strat = TheoreticallyOptimalStrategy()
    opt_strat.cmp_benchmark('JPM', should_plot=True)
