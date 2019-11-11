import datetime as dt
import pandas as pd
import numpy as np
import indicators as indi
import marketsimcode as msim
from util import get_data
from Plotter import Plotter


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
        # get volume and px data for vwpc
        vals = df.loc[symbol, 'AdjClose']
        volvals = df.loc[symbol, 'Volume']
        df_vwpc = pd.concat([vals, volvals], axis=1,
                            keys=['AdjClose', 'Volume'])
        df_vwpc['Symbol'] = symbol
        df_vwpc = df_vwpc.reset_index().set_index(['Symbol', 'Date'])

        df_metrics = df_pxs.copy()

        # get indicators for adj close data
        ws = [20, 5, 30]
        data_inputs = [df_pxs, df_pxs, df_vwpc]
        metrics = [indi.pct_sma, indi.rsi, indi.vwpc]
        standards = [True, False, True]
        for i, metric in enumerate(metrics):
            df_tmp = metric(data_inputs[i], window_sizes=[ws[i]],
                            standard=standards[i])
            df_metrics = df_metrics.join(df_tmp, how='inner')

        # rules engine
        df_jpm = df_metrics.loc[symbol]
        df_trades = pd.DataFrame(index=df_jpm.index)
        df_feats = df_trades.copy()

        sma_key = f'AdjClose_pct_sma_{ws[0]}'
        df_feats[sma_key] = 0.0
        sma_bcount = 3
        sma_bounds = [-1.0, 1.0]

        sma_lmask = df_jpm[sma_key] < sma_bounds[0]
        sma_umask = df_jpm[sma_key] > sma_bounds[1]
        tmp_sma_lmask = sma_lmask.copy()
        tmp_sma_umask = sma_umask.copy()
        # check for consecutive indicator hits
        for i in range(sma_bcount):
            sma_lmask = sma_lmask & tmp_sma_lmask.shift(i)
            sma_umask = sma_umask & tmp_sma_umask.shift(i)
        df_feats[sma_key][sma_lmask] = 1.0
        df_feats[sma_key][sma_umask] = -1.0

        rsi_key = f'AdjClose_rsi_{ws[1]}'
        df_feats[rsi_key] = 0.0
        rsi_bcount = 1
        rsi_bounds = [0.2, 0.8]

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

        vwpc_key = f'AdjClose_vwpc_{ws[2]}'
        df_feats[vwpc_key] = 0.0
        vwpc_bcount = 1
        vwpc_bounds = [-1.5, 1.5]

        vwpc_lmask = df_jpm[vwpc_key] < vwpc_bounds[0]
        vwpc_umask = df_jpm[vwpc_key] > vwpc_bounds[1]
        tmp_vwpc_lmask = vwpc_lmask.copy()
        tmp_vwpc_umask = vwpc_umask.copy()
        for i in range(vwpc_bcount):
            vwpc_lmask = vwpc_lmask & tmp_vwpc_lmask.shift(i)
            vwpc_umask = vwpc_umask & tmp_vwpc_umask.shift(i)
        df_feats[vwpc_key][vwpc_lmask] = 1.0
        df_feats[vwpc_key][vwpc_umask] = -1.0

        # determine trades based on chg in position
        pos = df_feats.sum(axis=1).clip(-1, 1)
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

    def cmp_benchmark(self, symbol, sd=dt.datetime(2008, 1, 1),
                      ed=dt.datetime(2009, 12, 31), sv=1e5,
                      bench_quantity=1e3, commission=9.95,
                      impact=0.001, should_plot=False, labels=None,
                      title=None, save_path=None):
        df_trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        sp = msim.compute_portvals(df_trades, start_val=sv,
                                   commission=commission, impact=impact)
        bp = self.benchmark_policy(symbol, sd=sd, ed=ed, sv=sv,
                                   quantity=bench_quantity,
                                   commission=commission, impact=impact)
        if labels is None:
            labels = ['benchmark', 'manual']

        # combine policies into single dataframe
        df_cmp = pd.concat([bp, sp], axis=1)
        df_cmp.columns = labels

        # optionally plot comparison
        if should_plot:
            plotter = Plotter()
            yax_labels = [f'indexed portfolio value', 'shares']
            colors = [[(0, 1, 0), (1, 0, 0)], [(0.35, 0.35, 0.35)]]
            df_pos = df_trades.cumsum()
            df_cmp['benchmark'] = df_cmp['benchmark']/bp.iloc[0]
            df_cmp['manual'] = df_cmp['manual']/sp.iloc[0]
            ycs = [[3, ['>', 0, 1]], [3, ['<', 0, 1]]]
            hcolors = [(0, 0, 1), (0, 0, 0)]
            plotter.stacked_plot(df_cmp, df_pos, yax_labels=yax_labels,
                                 title=title, ycs=ycs, colors=colors,
                                 hcolors=hcolors, yc_data=df_trades,
                                 save_path=save_path)

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


if __name__ == '__main__':
    ms = ManualStrategy()
    is_title = f'JPM Manual Strategy (In Sample)'
    df_in_sample = ms.cmp_benchmark('JPM', should_plot=False, commission=9.95,
                                    impact=0.001, title=is_title,
                                    save_path='is_chart.png')
    ms.performance(df_in_sample, show_table=False,
                   save_path='in_sample_table.png')
    os_title = f'JPM Manual Strategy (Out Sample)'
    df_out_sample = ms.cmp_benchmark('JPM', sd=dt.datetime(2010, 1, 1),
                                     ed=dt.datetime(2011, 12, 31),
                                     should_plot=False, commission=9.95,
                                     impact=0.001, title=os_title,
                                     save_path='os_chart.png')
    ms.performance(df_out_sample, show_table=False,
                   save_path='out_sample_table.png')
