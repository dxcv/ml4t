import os
import datetime as dt
import pandas as pd


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
    if 'SPY' in symbols:
        tdays = df.loc['SPY'].index.values
        df = df.loc[(slice(None), tdays), :]

    # remove whitespace from column names
    df.columns = [c.replace(' ', '') for c in df.columns.values]
    return df


def sym_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol"""
    if base_dir is None:
        base_dir = os.environ.get('MARKET_DATA_DIR', '../data/')
    return os.path.join(base_dir, f'{symbol}.csv')


def pct_sma(df, window_sizes=[5, 10]):
    tmp = df.sort_index().copy()
    groups = tmp.reset_index('Symbol').groupby('Symbol')
    df_pct_sma = pd.DataFrame(index=tmp.index)
    col_names = tmp.columns.values
    for ws in window_sizes:
        sma = tmp / groups.rolling(ws).mean()
        sma.columns = [f'{c}_pct_sma_{ws}' for c in col_names]
        df_pct_sma = df_pct_sma.join(sma)

    return df_pct_sma


if __name__ == '__main__':
    universe = ['JPM', 'GS']
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    dates = pd.date_range(start_date, end_date)
    data = load_data(universe, dates)
    print(data.info())
    print(f'\npct_sma...')
