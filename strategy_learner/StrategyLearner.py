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
import numpy as np
import util as ut
import indicators as indi
from QLearner import QLearner
# import random


class StrategyLearner(QLearner):
    def __init__(self, epochs=500, impact=0.001, positions=[-1, 0, 1],
                 bincnt=4, indicators='all', alpha=0.2, gamma=0.9,
                 rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.epochs = epochs
        self.impact = impact
        self.positions = np.array(positions)
        pos_rng = self.positions.max()-self.positions.min()
        self.actions = np.arange(-pos_rng, pos_rng+1)
        self.bincnt = bincnt
        if indicators == 'all':
            self.indicators = [indi.pct_sma, indi.rsi, indi.vwpc]
        elif isinstance(indicators, list):
            self.indicators = indicators[:]
        else:
            raise ValueError(f'indicators param must be list or \'all\'')

        # state space: bins*indicators*positions
        num_states = len(self.indicators)*self.bincnt*self.positions.shape[0]
        super().__init__(num_states=num_states,
                         num_actions=self.actions.shape[0],
                         alpha=alpha, gamma=gamma, rar=rar, radr=radr,
                         dyna=dyna, verbose=verbose)

    def author(self):
        return 'cfleisher'

    def addEvidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 12, 31), sv=1e5):
        """
        Trains QLearner
        """
        # load data
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        df = indi.ml4t_load_data(syms, dates)
        df_pxs = pd.DataFrame(df.AdjClose)
        df_met = df_pxs.copy()

        # generate indicators
        data_inputs = [df_pxs, df_pxs, df.loc[:, ['AdjClose', 'Volume']]]
        standards = [True, False, True]
        ws = [[20], [5], [30]]
        for i, d, s, w in zip(self.indicators, data_inputs, standards, ws):
            df_met = df_met.join(i(d, window_sizes=w, standard=s), how='inner')

        # discretize indicators
        df_met = df_met.loc[symbol].drop(['AdjClose'], axis=1).dropna()
        df_bins = df_met.copy()
        for c in df_met.columns.values:
            df_bins[c] = pd.cut(df_met[c], self.bincnt, labels=False)

        if self.verbose:
            print(f'metrics:\n{df_met.head()}')
            print(f'bins:\n{df_bins.head()}')

        # train qlearner
        pos_col = np.zeros((df_bins.shape[0], 1))
        scores = np.zeros((self.epochs, 1))
        pxchgs = df.loc[symbol, 'AdjClose']
        pxchgs = (pxchgs.shift(1)/pxchgs-1).dropna().values
        goal_reward = 0.05
        max_iters = 1e3
        sdi = np.arange(0, (len(self.indicators)+1)*self.bincnt, self.bincnt)
        for epoch in range(1, self.epochs+1):
            total_reward = 0
            # state_data = np.concatenate((df_bins.values, pos_col), axis=1)
            sd = np.concatenate((df_bins.values, pos_col), axis=1)
            sd += sdi
            count = 0
            while True:
                if total_reward > goal_reward or count > max_iters:
                    break
                s = sd[0]
                a = self.actions[self.querysetstate(s.sum())]
                for i, sp in enumerate(sd[1:]):
                    # update state position for action
                    sp[-1] = s[-1]+a*sdi[-1]
                    # get reward for prior state
                    r = pxchgs[i]*s[-1]
                    # get next a, roll state fwd; increment r
                    a = self.query(s.sum(), r)
                    s = sp
                    total_reward += r
                count += 1

            scores[epoch-1] = total_reward
            if self.verbose:
                print(f'epoch: {epoch} reward: {total_reward} count: {count}')
        return np.median(scores)

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
    slearner = StrategyLearner(verbose=True)
    result = slearner.addEvidence()
    print(f'median score: {result}')
