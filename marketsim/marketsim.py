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


def update_start(groupdf):
    groupdf.iloc[0]['ShareChg'] = groupdf.iloc[0]['Shares']
    return groupdf


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000,
                     commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # orders_file may be a string, or it may be a file object
    orders = pd.read_csv(orders_file, index_col=['Date'], parse_dates=True,
                         na_values=['nan']).sort_index()

    # add trade count column for aggregation purposes
    orders['Trades'] = 1
    print(f'{orders}\n')

    # get date range and symbols for indices
    symbols = list(orders['Symbol'].unique())
    start_date = pd.to_datetime(orders.index.values[0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(orders.index.values[-1]).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, end_date)

    # cvt sells to negatives and drop Order col
    orders = orders.apply(order_sign, axis=1).drop(['Order'], axis=1)

    # get price data and remove SPY
    pxs = get_data(symbols, dates).drop(['SPY'], axis=1)
    pxs = pxs.rename_axis('Symbol', axis=1)
    pxs.index = pxs.index.rename('Date')
    pxs = pxs.unstack().rename('Price')
    pxs = pd.DataFrame(pxs)

    # merge orders and prices
    orders = orders.reset_index().set_index(['Symbol', 'Date']).sort_index()
    portvals = pxs.join(orders)
    portvals['Price'] = portvals['Price'].fillna(method='ffill')
    portvals['Shares'] = portvals.groupby('Symbol').apply(lambda gdf: gdf.Shares.cumsum().fillna(method='ffill').fillna(0).reset_index('Symbol')).drop(['Symbol'], axis=1)
    # portvals['Shares'] = portvals.groupby('Symbol').apply(lambda gdf: gdf.Shares.fillna(method='ffill').fillna(0).reset_index('Symbol')).drop(['Symbol'], axis=1)
    portvals['Trades'] = portvals['Trades'].fillna(0)
    portvals = portvals.sort_index()

    # consolidate same day orders and sort on date index
    # orders = orders.groupby(['Symbol', 'Date']).sum().reset_index('Symbol')
    # orders = orders.sort_index()

    # cumulative shares
    # orders['Shares'] = orders.loc[:, ['Symbol', 'Shares']].groupby('Symbol').cumsum()

    # set MV, ShareChg cols; init Commis col
    portvals['MV'] = portvals.Shares*portvals.Price
    portvals['ShareChg'] = portvals.groupby('Symbol').apply(lambda gdf: gdf.Shares.diff().reset_index('Symbol')).drop(['Symbol'], axis=1)
    portvals['Commis'] = portvals.groupby('Symbol').apply(lambda gdf: (-gdf.Trades*commission).reset_index('Symbol')).drop(['Symbol'], axis=1)

    # update na in ShareChg col of first row to be Shares
    portvals = portvals.groupby('Symbol').apply(lambda gdf: update_start(gdf)).sort_index()

    portvals['Impact'] = (portvals.ShareChg/portvals.ShareChg.abs()).fillna(0)*impact+1
    portvals['Basis'] = -portvals.ShareChg*portvals.Price*portvals.Impact
    portvals[['Trades', 'Basis', 'ShareChg', 'Commis']] = portvals.groupby(['Symbol', 'Date'])[['Trades', 'Basis', 'ShareChg', 'Commis']].transform('sum')
    portvals = portvals.groupby(['Symbol', 'Date']).apply(lambda gdf: gdf.drop_duplicates(keep='last').reset_index(['Symbol','Date'])).drop(['Symbol','Date'], axis=1)
    portvals = portvals.groupby(['Symbol']).apply(lambda gdf: gdf.reset_index().drop_duplicates(subset=['Date'], keep='last').set_index('Date')).drop(['Symbol'], axis=1)

    # add cash balance
    cashdf = portvals.loc[portvals.index.values[0][0]].copy()
    cashdf = pd.concat([cashdf], keys=['CASH'], names=['Symbol'])
    cashdf.Price = 1.0
    cashdf.Shares = start_val
    cashdf.Trades = 0
    cashdf.ShareChg = 0
    cashdf.MV = cashdf.Price*cashdf.Shares
    cashdf.Commis = 0.0
    portvals = pd.concat([portvals, cashdf])
    # update na in ShareChg col of first row to be Shares
    portvals = portvals.groupby('Symbol').apply(lambda gdf: update_start(gdf)).sort_index()

    # calculate the trade basis
    portvals.loc['CASH'].Impact = 1.0
    portvals.loc['CASH'].MV = portvals.query('Symbol != "CASH"')[['Basis', 'Commis']].groupby('Date').sum(axis=1).sum(axis=1)
    portvals.loc['CASH', 'MV'].iloc[0] += start_val
    # print(f'cashdf portvals:\n{portvals.loc["CASH"].head(20)}')
    portvals.loc['CASH'].MV = portvals.loc['CASH'].MV.cumsum()

    # generate mv
    # print(portvals)
    # print('IBM')
    # print(portvals.loc['IBM'])
    # print(portvals.loc['CASH'])
    # for sym in symbols:
    #     print(sym)
    #     print(portvals.loc[sym].head(20))
    # print('CASH...')
    # print(portvals.loc['CASH'].head(20))
    mv = portvals.MV.groupby('Date').sum(axis=1)
    return mv


def author():
    return 'cfleisher3'


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    of = "./orders/orders-01.csv"
    sv = 1000000
    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv, commission=0.0, impact=0.000)
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
    # print('result...')
    # print(portvals.head(20))
    # for idx, row in enumerate(portvals):
    #     print(idx, row)
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
