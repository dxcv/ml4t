import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import marketsimcode as ms
from util import get_data


class TheoreticallyOptimalStrategy:
    def __init__(self):
        pass

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

    def benchmark_policy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1),
                         ed=dt.datetime(2009, 12, 31), sv=1e5, quantity=1e3,
                         commission=0.0, impact=0.0):
        """
            Generates portfolio value over given timeframe. The given quantity
            of the given symbol is acquired at the adjusted close price of the
            given start date assuming the given commissions and market impact.
            The portfolio value consists of the given symbol's market value and
            and residual cash.
        """
        dates = pd.date_range(sd, ed)
        pxs = get_data([symbol], dates)[symbol]
        pxs = pd.DataFrame(pxs)
        vals = ms.compute_portvals(pxs, start_val=sv, commission=commission,
                                   impact=impact)
        return vals

    def plot_strategy(self, df_strat, label=None, idxd=True, ylabel=None,
                      show_legend=True, color='g', line_alpha=0.7,
                      line_width=1.5, line_style='dotted', tc='0.25',
                      bgc='0.90', fc='0.6'):
        """
            params:
            - df_strat: pd df with dates and strategy portfolio values
            - label: values name; if None defaults to df Name
            - idxd: boolean toggle for index values based on start
            - ylabel: y-axis label
            - show_legend: whether to show data legend
            - color: line color
            - line_alpha: line alpha
            - line_width: line width
            - line_style: grid line style
            - tc: axes tick, axis label, legend color (grayscale)
            - bgc: background color (grayscale)
            - fc: frame color (grayscale)
        """
        df = df_strat.copy()

        # df values label defaults to df name
        if label is None:
            label = df.name

        # optionally index strategy values
        if idxd:
            df = df/df.iloc[0]

        # plot figure with given attributes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(linestyle=line_style)
        ax.plot(df.index.values, df.values, color=color, alpha=line_alpha,
                linewidth=line_width, label=label)

        # format axes
        # x-axis date format
        date_fmt = mdates.DateFormatter('%Y-%m')
        ax.xaxis.set_major_formatter(date_fmt)
        # axes tick colors
        ax.tick_params(colors=tc)

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
    bm = opt_strat.benchmark_policy()
    opt_strat.plot_strategy(bm, label='benchmark')

    if False:
        df_trades = opt_strat.testPolicy()
        portvals = ms.compute_portvals(df_trades, start_val=1e6,
                                       commission=0.0, impact=0.0)
        print(f'Portfolio Values:\n{portvals.head()}')
