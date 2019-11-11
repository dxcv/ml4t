import os
import datetime as dt
import pandas as pd
from util import get_data
from Plotter import Plotter


def load_data(symbols, dates, addSPY=True):
    """Read stock data for given symbols from CSV files."""
    df = pd.DataFrame()

    # conditionally add SPY for reference
    if addSPY and 'SPY' not in symbols:
        symbols = ['SPY'] + symbols

    # add data for each symbol provided
    for symbol in symbols:
        df_tmp = pd.read_csv(sym_to_path(symbol), index_col='Date',
                             parse_dates=True, na_values=['nan'])
        mi = pd.MultiIndex.from_product([[symbol], df_tmp.index.values],
                                        names=['Symbol', 'Date'])
        df_tmp = pd.DataFrame(df_tmp.values, index=mi, columns=df_tmp.columns)
        df = pd.concat([df, df_tmp])

    # conditionally filter SPY trading days
    if addSPY or 'SPY' in symbols:
        tdays = df.loc['SPY'].index.values
        df = df.loc[(slice(None), tdays), :]

    # pull available trading days from SPY filtered and fillna
    df = df.loc[(slice(None), dates), :]
    df = df.groupby('Symbol').fillna(method='ffill')
    df = df.groupby('Symbol').fillna(method='bfill')

    # remove whitespace from column names
    df.columns = [c.replace(' ', '') for c in df.columns.values]

    # return data sorted in ascending order
    return df.sort_index()


def ml4t_load_data(symbols, dates, addSPY=True,
                   cols='all'):
    # check if cols supplied user list or default to all ml4t feats
    if cols == 'all':
        cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    df = pd.DataFrame()

    # pull data by col using required util function
    for col in cols:
        df_tmp = get_data(symbols, dates, addSPY, col)
        df_tmp = pd.concat([df_tmp], keys=[col.replace(' ', '')])
        df = pd.concat([df, df_tmp])

    df = df.unstack().T
    df.index = df.index.rename(['Symbol', 'Date'])

    if addSPY or 'SPY' in symbols:
        tdays = df.loc['SPY'].index.values
        df = df.loc[(slice(None), tdays), :]

    # pull available trading days from SPY filtered and fillna
    df = df.loc[(slice(None), dates), :]
    df = df.groupby('Symbol').fillna(method='ffill')
    df = df.groupby('Symbol').fillna(method='bfill')

    return df.sort_index()


def sym_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol"""
    if base_dir is None:
        base_dir = os.environ.get('MARKET_DATA_DIR', '../data/')
    return os.path.join(base_dir, f'{symbol}.csv')


def pct_sma(df, window_sizes=[5, 10], standard=True):
    tmp = df.copy()
    df_pct_sma = pd.DataFrame(index=tmp.index)
    col_names = tmp.columns.values
    for n in window_sizes:
        pct = tmp/sma(tmp, n)
        pct.columns = [f'{c}_pct_sma_{n}' for c in col_names]
        df_pct_sma = df_pct_sma.join(pct)

    # standardize data across all symbols by feature
    if standard:
        df_pct_sma = (df_pct_sma-df_pct_sma.mean())/df_pct_sma.std()

    return df_pct_sma


def sma(df, n):
    """Simple Moving Average with window size n"""
    return df.reset_index('Symbol').groupby('Symbol').rolling(n).mean()


def vwpc(df, window_sizes=[5, 10], standard=True):
    df_vwpc = pd.DataFrame(index=df.index)
    tmp = pd.DataFrame(df.iloc[:, 0]*df.iloc[:, 1],
                       columns=['weighted'])
    col_names = [df.columns.values[0]]
    for n in window_sizes:
        chg = tmp/tmp.shift(n)-1
        chg.columns = [f'{c}_vwpc_{n}' for c in col_names]
        df_vwpc = df_vwpc.join(chg)

    if standard:
        df_vwpc = (df_vwpc-df_vwpc.mean())/df_vwpc.std()

    return df_vwpc


def rsi(df, window_sizes=[5, 10], standard=False):
    """
        RSI = 100 - 100/1+RS
        RS1 = total_gain/total_loss
        RS2 = [((n-1)total_gain+gain_n]/[(n-1)total_loss+loss_n]
        Note: standard default False given absolute value relevance
    """
    chg = df.copy()
    chg = (chg/chg.shift(1)-1)
    gain = chg[chg >= 0].fillna(0)
    loss = chg[chg < 0].abs().fillna(0)
    gain_grp = gain.reset_index('Symbol').groupby('Symbol')
    loss_grp = loss.reset_index('Symbol').groupby('Symbol')
    df_rsi = pd.DataFrame(index=chg.index)
    col_names = chg.columns.values
    for n in window_sizes:
        tgain = gain_grp.rolling(n).sum()
        tloss = loss_grp.rolling(n).sum()
        rs2 = ((n-1)*tgain+gain)/((n-1)*tloss+loss)
        rsi = 100-100/(1+rs2.fillna(tgain/tloss))
        rsi.columns = [f'{c}_rsi_{n}' for c in col_names]
        df_rsi = df_rsi.join(rsi)

    if standard:
        df_rsi = (df_rsi-df_rsi.mean())/df_rsi.std()

    return df_rsi


def author():
    """Required by ml4t rubric: returns gt username"""
    return 'cfleisher3'


def display_sma(df, symbol, metric, ws=10, ycs=None, save_path=None):
    vals = df.loc[symbol, metric]
    df_vals = pd.DataFrame(pd.concat([vals], keys=[symbol],
                                     names=['Symbol']))
    df_sma = sma(df_vals, ws)
    smas = df_sma.loc[symbol, metric]

    df_pct_sma = pct_sma(df_vals, window_sizes=[ws],
                         standard=True)
    psmas = df_pct_sma.loc[symbol, f'{metric}_pct_sma_{ws}']
    vals = vals/vals[vals.first_valid_index()]
    smas = smas/smas[smas.first_valid_index()]
    x1 = pd.DataFrame({metric: vals, 'sma': smas})
    x2 = pd.DataFrame({'standard pct sma': psmas})

    plotter = Plotter()
    yax_labels = [f'indexed {metric}, sma', 'standard pct sma']
    colors = [[(0.6, 0.13, 0.79), (0.79, 0.5, 0.13)], [(0.35, 0.35, 0.35)]]
    hcolors = [(0, 0, 1), (0, 0, 0)]
    plotter.stacked_plot(x1, x2, yax_labels=yax_labels,
                         title=f'{symbol} {metric} SMA', ycs=ycs,
                         colors=colors, hcolors=hcolors,
                         save_path=save_path)


def display_rsi(df, symbol, metric, ws=10, ycs=None, save_path=None):
    vals = df.loc[symbol, metric]
    df_vals = pd.DataFrame(pd.concat([vals], keys=[symbol],
                                     names=['Symbol']))
    df_rsi = rsi(df_vals, window_sizes=[ws])
    rsivals = df_rsi.loc['JPM', f'AdjClose_rsi_{ws}']
    vals = vals/vals[vals.first_valid_index()]
    rsivals = rsivals/100
    x1 = pd.DataFrame({metric: vals})
    x2 = pd.DataFrame({'rsi': rsivals})
    plotter = Plotter()
    colors = [[(0.6, 0.13, 0.79)], [(0.35, 0.35, 0.35)]]
    hcolors = [(0, 0, 1), (0, 0, 0)]
    plotter.stacked_plot(x1, x2, yax_labels=[f'indexed {metric}', 'rsi'],
                         title=f'{symbol} {metric} RSI', ycs=ycs,
                         colors=colors, hcolors=hcolors, save_path=save_path)


def display_vwpc(df, symbol, metric, ws=10, ycs=None, save_path=None):
    vals = df.loc[symbol, metric]
    volvals = df.loc[symbol, 'Volume']
    df_vals = pd.concat([vals, volvals], axis=1,
                        keys=[metric, 'Volume'])
    df_vals['Symbol'] = symbol
    df_vals = df_vals.reset_index().set_index(['Symbol', 'Date'])
    df_vwpc = vwpc(df_vals, window_sizes=[ws], standard=False)
    vwpcvals = df_vwpc.loc['JPM', f'{metric}_vwpc_{ws}']
    vals = vals/vals[vals.first_valid_index()]
    weighted = vals*volvals
    weighted = (weighted-weighted.min())/(weighted.max()-weighted.min())
    x1 = pd.DataFrame({metric: vals, 'norm vwp': weighted})
    x2 = pd.DataFrame({'vwpc': vwpcvals})
    plotter = Plotter()
    colors = [[(0.6, 0.13, 0.79), (0.79, 0.5, 0.13)], [(0.35, 0.35, 0.35)]]
    hcolors = [(0, 0, 1), (0, 0, 0)]
    yax_labels = [f'indexed {metric}, norm vwp', 'vwpc']
    plotter.stacked_plot(x1, x2, yax_labels=yax_labels,
                         title=f'{symbol} {metric} VWPC',
                         colors=colors, hcolors=hcolors, ycs=ycs,
                         save_path=save_path)


if __name__ == '__main__':
    universe = ['JPM', 'GS']
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    dates = pd.date_range(start_date, end_date)

    data = ml4t_load_data(universe, dates)
    # sma
    sma_ycs = [[2, ['<', -1.0, 3]], [2, ['>', 1.0, 3]]]
    display_sma(data, 'JPM', 'AdjClose', ws=20, ycs=sma_ycs,
                save_path='sma.png')

    # rsi
    rsi_ycs = [[2, ['<', 0.2, 1]], [2, ['>', 0.8, 1]]]
    display_rsi(data, 'JPM', 'AdjClose', ws=5, ycs=rsi_ycs,
                save_path='rsi.png')

    # vwpc
    vwpc_ycs = [[2, ['<', -1.5, 1]], [2, ['>', 1.5, 1]]]
    display_vwpc(data, 'JPM', 'AdjClose', ws=30, ycs=vwpc_ycs,
                 save_path='vwpc.png')
