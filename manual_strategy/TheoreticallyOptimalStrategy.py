import datetime as dt
import pandas as pd
import numpy as np
import marketsimcode as ms
from util import get_data
from Plotter import Plotter


class TheoreticallyOptimalStrategy:
    def __init__(self):
        pass

    def author(self):
        return 'cfleisher3'

    def cmp_benchmark(self, symbol, sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=1e5,
                      bench_quantity=1e3, commission=0.0,
                      impact=0.0, should_plot=False, save_path=None,
                      labels=None):
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
        if should_plot or save_path is not None:
            plotter = Plotter()
            yax_labels = [f'indexed portfolio value', 'alpha']
            title = f'{symbol} Theoretically Optimal Strategy'
            colors = [[(0, 1, 0), (1, 0, 0)], [(0.35, 0.35, 0.35)]]
            df_alpha = pd.DataFrame({'alpha': (tp-bp)/sv})
            df_cmp['benchmark'] = df_cmp['benchmark']/bp.iloc[0]
            df_cmp['optimal'] = df_cmp['optimal']/tp.iloc[0]
            plotter.stacked_plot(df_cmp, df_alpha, yax_labels=yax_labels,
                                 title=title, colors=colors,
                                 should_show=should_plot, save_path=save_path)

        return df_cmp

    def performance(self, df_strats, show_table=False, save_path=None):
        pxchg = df_strats/df_strats.shift(1)-1
        data = {
            'cr': df_strats.iloc[-1]/df_strats.iloc[0]-1,
            'std': pxchg.std(),
            'adr': pxchg.mean(),
        }
        metrics = pd.DataFrame(data)
        metrics = metrics.round(5)

        if show_table or save_path is not None:
            plotter = Plotter()
            plotter.table(metrics, show_table=show_table, save_path=save_path)
        return metrics

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


def author():
    return 'cfleisher3'


if __name__ == '__main__':
    opt_strat = TheoreticallyOptimalStrategy()
    cmps = opt_strat.cmp_benchmark('JPM', should_plot=False,
                                   save_path='opt_port_chart.png')
    opt_strat.performance(cmps, show_table=False, save_path='opt_table.png')
