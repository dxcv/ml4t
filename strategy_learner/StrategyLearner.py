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
import indicators as indi
from QLearner import QLearner
# import random


class StrategyLearner(QLearner):
    def __init__(self, epochs=500, impact=0.001, positions=[-1, 0, 1],
                 bincnt=4, indicators='all', alpha=0.2, gamma=0.9,
                 rar=0.5, radr=0.999, dyna=200, verbose=False):
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

        # train qlearner
        pos_col = np.ones((df_bins.shape[0], 1))
        scores = np.zeros((self.epochs, 1))
        pxchgs = df.loc[symbol, 'AdjClose']
        pxchgs = (pxchgs/pxchgs.shift(1)-1).dropna().values
        cash_reward = 0.0
        sdi = np.arange(0, (len(self.indicators)+1)*self.bincnt, self.bincnt)
        pfloor = self.positions.min()
        pceil = self.positions.max()
        rar = self.rar
        converge = 0.025
        for epoch in range(1, self.epochs+1):
            self.rar = rar
            sd = np.concatenate((df_bins.values, pos_col), axis=1).astype(int)
            sd += sdi
            s = sd[0]
            a = self.actions[self.querysetstate(s.sum())]
            rewards = np.zeros((pxchgs.shape[0],))
            for i, sp in enumerate(sd[1:]):
                # update state position for action
                prior_pos = int(s[-1]-sdi[-1]-1)
                pos = np.clip(prior_pos+a, pfloor, pceil)
                sp[-1] = pos+sdi[-1]+1
                # get reward for prior state
                if prior_pos == 0:
                    r = cash_reward
                else:
                    r = pxchgs[i]*prior_pos
                # get next a, roll state fwd; increment r
                rewards[i] = r
                a = self.actions[self.query(s.sum(), r)]
                s = sp

            scores[epoch-1] = ((pxchgs-rewards)**2).mean()**0.5
            print(f'epoch: {epoch} score: {scores[epoch-1]}')
            if epoch >= 5:
                break
                bench = scores[epoch-10:epoch-2].mean()
                rmseschg = np.abs(scores[epoch-1]/bench-1)
                if rmseschg < converge:
                    break

        return scores

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1),
                   ed=dt.datetime(2011, 12, 31), sv=1e5):
        """
        Tests existing policy against new data
        """
        # get data
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        df = indi.ml4t_load_data(syms, dates)
        df_pxs = pd.DataFrame(df.AdjClose)
        df_met = df_pxs.copy()

        # generate indicators
        data_inputs = [df_pxs, df_pxs, df.loc[:, ['AdjClose', 'Volume']]]
        standards = [True, False, False]
        ws = [[20], [5], [30]]
        for i, d, s, w in zip(self.indicators, data_inputs, standards, ws):
            df_met = df_met.join(i(d, window_sizes=w, standard=s), how='inner')

        # discretize indicators
        df_met = df_met.loc[symbol].drop(['AdjClose'], axis=1).dropna()
        df_bins = df_met.copy()
        for c in df_met.columns.values:
            df_bins[c] = pd.cut(df_met[c], self.bincnt, labels=False)

        # pass data thru policy
        df_trades = pd.DataFrame(index=df_pxs.loc[symbol].index)
        df_trades['Shares'] = 0
        pos_col = np.ones((df_bins.shape[0], 1))
        sd = np.concatenate((df_bins.values, pos_col), axis=1).astype(int)
        sdi = np.arange(0, (len(self.indicators)+1)*self.bincnt, self.bincnt)
        sd += sdi
        actions = np.zeros((sd.shape[0],))
        p = 0
        for i, s in enumerate(sd):
            a = self.actions[self.Q[s.sum(), :].argmax()]
            actions[i] = np.clip(p+a, -1, 1)-p
            p += actions[i]

        df_pos = pd.DataFrame(actions.cumsum(), index=df_bins.index,
                              columns=['Shares'])
        df_trades.update(df_pos)
        df_trades *= 1e3
        return df_trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")
    slearner = StrategyLearner(verbose=False)
    slearner.addEvidence()
    trades = slearner.testPolicy()
    lng = (trades.Shares > 0).sum()
    shrt = (trades.Shares < 0).sum()
    cash = trades.shape[0]-lng-shrt
    tot = lng + shrt + cash
    print(f'trades summary:')
    print(f'longs: {lng/tot:.2f} shorts: {shrt/tot:.2f} cash: {cash/tot:.2f}')
