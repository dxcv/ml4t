"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
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
import datetime as dt
import pandas as pd
import util as ut
import indicators as indi
# import random


class StrategyLearner(object):
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact

    def author(self):
        return 'cfleisher'

    def addEvidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 12, 31), sv=1e5):
        """
        Creates and trains a QLearner for trading
        """
        # load data
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        df = indi.ml4t_load_data(syms, dates)
        df_pxs = pd.DataFrame(df.AdjClose)
        df_met = df_pxs.copy()

        # generate indicators
        metrics = [indi.pct_sma, indi.rsi, indi.vwpc]
        data_inputs = [df_pxs, df_pxs, df.loc[:, ['AdjClose', 'Volume']]]
        standards = [True, False, True]
        ws = [[20], [5], [30]]
        for m, d, s, w in zip(metrics, data_inputs, standards, ws):
            df_met = df_met.join(m(d, window_sizes=w, standard=s), how='inner')

        # discretize indicators
        bincnt = 4
        df_bins = df_met.copy()
        for c in df_met.columns.values:
            df_bins[c] = pd.cut(df_met[c], bincnt, labels=False)

        if self.verbose:
            print(f'metrics:\n{df_met.head(35)}')
            print(f'bins:\n{df_bins.head(35)}')

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=1e5):
        """
        Tests existing policy against new data
        """
        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data([symbol], dates)  # automatically adds SPY
        trades = prices_all[[symbol, ]]  # only portfolio symbols
        # trades_SPY = prices_all['SPY']  # only SPY, for comparison later
        trades.values[:, :] = 0  # set them all to nothing
        trades.values[0, :] = 1000  # add a BUY at the start
        trades.values[40, :] = -1000  # add a SELL
        trades.values[41, :] = 1000  # add a BUY
        trades.values[60, :] = -2000  # go short from long
        trades.values[61, :] = 2000  # go long from short
        trades.values[-1, :] = -1000  # exit on the last day
        if self.verbose:
            print(type(trades))  # it better be a DataFrame!
        if self.verbose:
            print(trades)
        if self.verbose:
            print(prices_all)
        return trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")
