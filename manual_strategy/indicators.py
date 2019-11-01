import os
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import LinearLocator
from matplotlib.gridspec import GridSpec
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


def pct_vwap(df, window_sizes=[5, 10], standard=True):
    df_pct_vwap = pd.DataFrame(index=df.index)
    tmp = pd.DataFrame(df.iloc[:, 0]*df.iloc[:, 1],
                       columns=['weighted'])
    col_names = [df.columns.values[0]]
    for n in window_sizes:
        pct = tmp/sma(tmp, n)
        pct.columns = [f'{c}_pct_vwap_{n}' for c in col_names]
        df_pct_vwap = df_pct_vwap.join(pct)

    if standard:
        df_pct_vwap = (df_pct_vwap-df_pct_vwap.mean())/df_pct_vwap.std()

    return df_pct_vwap


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


def bollinger_bands(df, ma, n, k):
    """
        Bollinger Bands for n window size and k stdevs
        returns lower, upper bands
    """
    groups = df.reset_index('Symbol').groupby('Symbol')
    std = groups.rolling(n).std()
    return ma-std*k, ma+std*k


def bollinger(df, window_sizes=[5, 10], k=2, mafn=sma, standard=True):
    """
        Bollinger Value for n window size and k stdevs
    """
    groups = df.reset_index('Symbol').groupby('Symbol')
    df_bvals = pd.DataFrame(index=df.index)
    col_names = df.columns.values
    for n in window_sizes:
        ma = mafn(df, n)
        std = groups.rolling(n).std()
        bval = (df-ma)/(k*std)
        bval.columns = [f'{c}_bollinger_{n}' for c in col_names]
        df_bvals = df_bvals.join(bval, how='inner')

    if standard:
        df_bvals = (df_bvals-df_bvals.mean())/df_bvals.std()

    return df_bvals


def author():
    """Required by ml4t rubric: returns gt username"""
    return 'cfleisher3'


def plot_standard(X, labels, ylabel, lbound=None, ubound=None,
                  lbc='g', ubc='r'):
    fig = plt.figure()

    colors = ['b', 'm', 'k', 'y', 'o']
    line_alpha = 0.7

    ax1 = fig.add_subplot(111)
    ax1.grid(linestyle='dotted')
    for i, x in enumerate(X):
        ax1.plot(x.index.values, x.values,
                 color=colors[i], alpha=line_alpha,
                 linewidth=1.5, label=labels[i])

    # set the number of gridlines for both subplots
    ax1.xaxis.set_major_locator(LinearLocator(7))

    # format x-axis dates and hide labels for top plot
    date_fmt = mdates.DateFormatter('%Y-%m')
    tc = '0.25'
    ax1.tick_params(axis='y', colors=tc)
    ax1.xaxis.set_major_formatter(date_fmt)

    # background colors
    bgc = '0.90'
    ax1.set_facecolor(bgc)

    # frame colors
    fc = '0.6'
    plt.setp(ax1.spines.values(), color=fc)

    # axis labels
    if ylabel:
        ax1.set_ylabel(ylabel, color=tc)

    # legend
    leg1 = ax1.legend()
    for txt in leg1.get_texts():
        txt.set_color(tc)

    for lh in leg1.legendHandles:
        lh.set_alpha(line_alpha)

    # shade boundary regions
    if ubound is not None:
        above = get_regions(data[-1], data[-1] > ubound)
        for r in above:
            ax1.axvspan(*r, color=ubc, alpha=0.5)

    if lbound is not None:
        below = get_regions(data[-1], data[-1] < lbound)
        for r in below:
            ax1.axvspan(*r, color=lbc, alpha=0.5)

    plt.show()
    plt.clf()


def plot_vertical(X, bdata, labels, ylabel=None, bylabel=None,
                  lbound=None, ubound=None, bcnt=3, lbc='g', ubc='r'):
    """
        data: list of series to plot
        bdata: series for bottom plot
        labels: data legend labels
        ylabel: y-axis label
        bylabel: bottom y-axis label
    """
    fig = plt.figure()
    gs = GridSpec(2, 1, height_ratios=[2, 1])  # smaller bottom subplot
    gs.update(hspace=0.025)  # spacing between subplots

    colors = ['b', 'm', 'k', 'y', 'o']
    line_alpha = 0.7

    ax1 = fig.add_subplot(gs[0])
    ax1.grid(linestyle='dotted')
    for i, x in enumerate(X):
        ax1.plot(x.index.values, x.values,
                 color=colors[i], alpha=line_alpha,
                 linewidth=1.5, label=labels[i])

    ax2 = fig.add_subplot(gs[1])
    ax2.grid(linestyle='dotted')
    ax2.plot(bdata.index.values, bdata.values,
             color=colors[len(X)], alpha=line_alpha,
             linewidth=1.25)

    # set the number of gridlines for both subplots
    ax1.xaxis.set_major_locator(LinearLocator(7))
    ax2.xaxis.set_major_locator(LinearLocator(7))
    ax1.set_xlim(ax2.get_xlim())  # align x-axis of subplots

    # format x-axis dates and hide labels for top plot
    date_fmt = mdates.DateFormatter('%Y-%m')
    tc = '0.25'
    ax1.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)
    ax1.tick_params(axis='y', colors=tc)
    ax2.xaxis.set_major_formatter(date_fmt)
    ax2.tick_params(colors=tc)

    # background colors
    bgc = '0.90'
    ax1.set_facecolor(bgc)
    ax2.set_facecolor(bgc)

    # frame colors
    fc = '0.6'
    plt.setp(ax1.spines.values(), color=fc)
    plt.setp(ax2.spines.values(), color=fc)

    # axis labels
    if ylabel:
        ax1.set_ylabel(ylabel, color=tc)
    if bylabel:
        ax2.set_ylabel(bylabel, color=tc)

    # legend
    leg1 = ax1.legend()
    for txt in leg1.get_texts():
        txt.set_color(tc)

    for lh in leg1.legendHandles:
        lh.set_alpha(line_alpha)

    # shade boundary regions
    if ubound is not None:
        print((bdata > ubound).head(50))
        print(bdata.head(50))
        above = get_regions(bdata, bdata > ubound, n=bcnt)
        for r in above:
            ax1.axvspan(*r, color=ubc, alpha=0.5)
            ax2.axvspan(*r, color=ubc, alpha=0.5)

    if lbound is not None:
        below = get_regions(bdata, bdata < lbound, n=bcnt)
        for r in below:
            ax1.axvspan(*r, color=lbc, alpha=0.5)
            ax2.axvspan(*r, color=lbc, alpha=0.5)

    plt.show()
    plt.clf()


def get_regions(data, bounds, n=3):
    """find regions meeting bounds criteria"""
    umask = bounds
    for i in range(n):
        umask = umask & bounds.shift(i)

    uidxs = umask.index.values
    uvls = umask.values
    um = len(uidxs)-1
    return [[uidxs[i], uidxs[min(um, i+n-1)]] for i, v in enumerate(uvls) if v]


def display_sma(df, symbol, metric, ws=10, lbound=-1.5, ubound=1.5):
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
    x2 = pd.DataFrame({'norm pct sma': psmas})

    plotter = Plotter()
    ycs = [[2, ['<', -1.5, 20]], [2, ['>', 1.5, 20]]]
    plotter.stacked_plot(x1, x2, yax_labels=['indexed', 'norm pct sma'],
                         title=f'{symbol} {metric} SMA', ycs=ycs)


def display_bollinger(df, symbol, metric, ws=10, k=2, lbound=-1.0, ubound=0.9):
    vals = df.loc[symbol, metric]
    df_vals = pd.DataFrame(pd.concat([vals], keys=[symbol],
                           names=['Symbol']))
    df_sma = sma(df_vals, ws)
    df_lband, df_uband = bollinger_bands(df_vals, df_sma, ws, 2)
    uband = df_uband.loc['JPM', 'AdjClose']
    lband = df_lband.loc['JPM', 'AdjClose']
    df_bval = bollinger(df_vals, window_sizes=[ws], standard=False)
    bvals = df_bval.loc['JPM', f'AdjClose_bollinger_{ws}']
    band_labels = [metric, 'upper band', 'lower band']
    bbds = [s/vals[vals.first_valid_index()] for s in [vals, uband, lband]]
    plot_vertical(bbds, bvals, band_labels,
                  ylabel=f'indexed {metric}', bylabel='bollinger',
                  lbound=-1.0, ubound=0.9, bcnt=2, lbc='r', ubc='g')


def display_rsi(df, symbol, metric, ws=10, lbound=None, ubound=None):
    vals = df.loc[symbol, metric]
    df_vals = pd.DataFrame(pd.concat([vals], keys=[symbol],
                                     names=['Symbol']))
    df_rsi = rsi(df_vals, window_sizes=[ws])
    rsivals = df_rsi.loc['JPM', f'AdjClose_rsi_{ws}']
    labels = [metric, 'rsi']
    X = [vals/vals[vals.first_valid_index()]]
    bvals = rsivals/100
    plot_vertical(X, bvals, labels,
                  ylabel=f'indexed {metric} and rsi', bylabel='rsi',
                  lbound=0.3, ubound=0.7, bcnt=4, lbc='g', ubc='r')


def display_vwap(df, symbol, metric, ws=10, lbound=-1.0, ubound=1.0):
    vals = df.loc[symbol, metric]
    volvals = df.loc[symbol, 'Volume']
    df_vals = pd.concat([vals, volvals], axis=1,
                        keys=[metric, 'Volume'])
    df_vals['Symbol'] = symbol
    df_vals = df_vals.reset_index().set_index(['Symbol', 'Date'])
    df_vwap = pct_vwap(df_vals, window_sizes=[ws], standard=True)
    vwapvals = df_vwap.loc['JPM', f'{metric}_pct_vwap_{ws}']
    labels = [metric, 'vwap']
    X = [vals/vals[vals.first_valid_index()]]
    plot_vertical(X, vwapvals, labels, ylabel=f'indexed {metric} and vwap',
                  bylabel='vwap', lbound=lbound, ubound=ubound, bcnt=3,
                  lbc='r', ubc='g')


if __name__ == '__main__':
    universe = ['JPM', 'GS']
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    dates = pd.date_range(start_date, end_date)

    data = ml4t_load_data(universe, dates)
    # display_vwap(data, 'JPM', 'AdjClose')
    display_sma(data, 'JPM', 'AdjClose')
    # display_rsi(data, 'JPM', 'AdjClose')
