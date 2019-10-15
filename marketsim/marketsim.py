"""MC2-P1: Market simulator.
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved
Template code for CS 4646/7646
Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.
We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.
-----do not edit anything above this line---
Student Name: Christopher Fleisher
GT User ID: cfleisher3
GT ID: 903421975
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def order_sign(row):
    if row['Order'] == 'SELL':
        row['Shares'] = -row['Shares']
        return row
    else:
        return row


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000,
                     commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # orders_file may be a string, or it may be a file object
    orders = pd.read_csv(orders_file, index_col=['Date'], parse_dates=True,
                         na_values=['nan']).sort_index()

    # add trade count column for aggregation purposes
    orders['Trades'] = 1

    # order summary
    # trade_count = orders.shape[0]
    # print(orders)
    # print()

    # get date range and symbols for indices
    symbols = list(orders['Symbol'].unique())
    start_date = pd.to_datetime(orders.index.values[0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(orders.index.values[-1]).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, end_date)

    # cvt sells to negatives and drop Order col
    orders = orders.apply(order_sign, axis=1).drop(['Order'], axis=1)
    # consolidate same day orders and resort
    orders = orders.groupby(['Symbol', 'Date']).sum().reset_index('Symbol')
    orders = orders.sort_index()

    # update date range and fill na with 0
    orders = orders.groupby('Symbol') \
        .apply(lambda df: df.resample('D').first()) \
        .drop(['Symbol'], axis=1).fillna(0).reset_index('Symbol')

    orders = orders.sort_index()

    # cumulative shares
    orders['Shares'] = orders.loc[:, ['Symbol', 'Shares']].groupby('Symbol').cumsum()

    # pivot df to match get_data
    # orders = orders.pivot(index='Date', columns='Symbol', values='Shares')
    # orders = orders.pivot(index='Date', columns='Symbol').swaplevel(i=0, j=1, axis=1)
    orders = orders.pivot(columns='Symbol').swaplevel(i=0, j=1, axis=1)
    orders.loc[:, (slice(None), 'Shares')] = orders.loc[:, (slice(None), 'Shares')].fillna(method='ffill')
    orders = orders.fillna(0)
    # ffill and bfill 0 to handle dates out of order range for ticker
    # orders = orders.fillna(method='ffill').fillna(0)

    # get cumulative shares
    # orders = orders.groupby('Symbol').cumsum().reset_index()
    # orders.loc[:, (slice(None), 'Shares')] = orders.loc[:, (slice(None), 'Shares')].cumsum()
    # for sym in symbols:
    #     print(sym, orders.loc[orders.Symbol == sym].iloc[-1])
    # print(f'cumulative orders:\n{orders.head()}\n')

    # update column names for multiindex Shares
    # orders.columns = pd.MultiIndex.from_product([orders.columns, ['Shares']])

    # get price data and remove SPY
    pxs = get_data(symbols, dates).drop(['SPY'], axis=1)
    pxs.columns = pd.MultiIndex.from_product([pxs.columns, ['Price']])
    # concatenate prices with shares
    portvals = pd.concat([orders, pxs], axis=1, join='outer')
    portvals.loc[:, (slice(None), 'Shares')] = portvals.loc[:, (slice(None), 'Shares')].fillna(method='ffill')
    portvals.loc[:, (slice(None), 'Trades')] = portvals.loc[:, (slice(None), 'Trades')].fillna(0)
    portvals = portvals.dropna()
    for sym in symbols:
        portvals[sym, 'MV'] = portvals[sym, 'Shares']*portvals[sym, 'Price']
        portvals[sym, 'ShareChg'] = portvals[sym, 'Shares'].diff()
        portvals[sym, 'Commis'] = 0.0

    # update na in first row to be shares
    tmp = portvals.iloc[0][:, 'Shares'].sort_index()
    for idx, shares in enumerate(tmp.values):
        portvals.loc[start_date, (tmp.index[idx], 'ShareChg')] = shares

    for idx, row in portvals.iterrows():
        for sym in symbols:
            row[sym, 'Commis'] = -row[sym, 'Trades']*commission
            # if row[sym, 'ShareChg'] > 0 or row[sym, 'ShareChg'] < 0:
            #     row[sym, 'Commis'] = -commission

    # get cash basis
    for sym in symbols:
        portvals[sym, 'Basis'] = -portvals[sym, 'ShareChg']*portvals[sym, 'Price']*(1+impact)

    # cash balance
    portvals.loc[:, ('Cash', 'MV')] = 0.0
    portvals.iloc[0, -1] = start_val
    
    # update rolling basis
    basis = portvals.loc[:, (slice(None), 'Basis')].sum(axis=1)
    commis = portvals.loc[:, (slice(None), 'Commis')].sum(axis=1)
    for idx, val in enumerate(basis):
        if idx == 0:
            portvals.iloc[0]['Cash', 'MV'] = 1e6 + val + commis.iloc[idx]
        else:
            portvals.iloc[idx]['Cash', 'MV'] = portvals.iloc[idx-1]['Cash', 'MV']+val+commis.iloc[idx]

    # commission summary
    # gross_commis = portvals.loc[:, (slice(None), 'Commis')]
    # trade_commis = gross_commis[gross_commis != 0].dropna(how='all').fillna(0)
    # gross_commis_count = gross_commis[gross_commis != 0].count().sum()
    # gross_commis_sum = gross_commis.sum().sum()
    # trades = portvals.loc[:, (slice(None), 'Trades')]
    # tcount = trades[trades != 0]
    # print(f'Trade Count: {trade_count}')
    # print(f'Updated Trade Count: {tcount.sum().sum()}')
    # print(f'Commision Count: {gross_commis_count}')
    # print(f'Expected Commissions: {-trade_count*commission}')
    # print(f'Actual Commissions: {gross_commis_sum}')
    # print(trade_commis)

    # generate mv
    mv = portvals.loc[:, (slice(None), 'MV')].sum(axis=1)

    # In the template, instead of computing the value of the portfolio, we just
    # read in the value of IBM over 6 months
    # start_date = dt.datetime(2008, 1, 1)
    # end_date = dt.datetime(2008, 6, 1)
    # portvals = get_data(['IBM'], pd.date_range(start_date, end_date))
    # portvals = portvals[['IBM']]  # remove SPY
    # rv = pd.DataFrame(index=portvals.index, data=portvals.values)
    return mv


def author():
    return 'cfleisher3'


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    of = "./orders/orders-11.csv"
    sv = 1000000
    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Get portfolio stats
    # Here we fake data. use code from previous assignments
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cr, adr, stdr, sr = [0.2, 0.01, 0.02, 1.5]
    cr_SPY, adr_SPY, stdr_SPY, sr_SPY = [0.2, 0.01, 0.02, 1.5]
    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sr}")
    print(f"Sharpe Ratio of SPY : {sr_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cr}")
    print(f"Cumulative Return of SPY : {cr_SPY}")
    print()
    print(f"Standard Deviation of Fund: {stdr}")
    print(f"Standard Deviation of SPY : {stdr_SPY}")
    print()
    print(f"Average Daily Return of Fund: {adr}")
    print(f"Average Daily Return of SPY : {adr_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
