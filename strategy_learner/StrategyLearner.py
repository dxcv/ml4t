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
import marketsimcode as msim
from QLearner import QLearner
import random


class StrategyLearner(QLearner):
    def __init__(self, epochs=500, impact=0.00, positions=[-1, 0, 1],
                 actions=[0, -2, -1, 1, 2], bincnt=4, indicators='all',
                 alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=200,
                 verbose=False, commission=0.0):
        self.epochs = epochs
        self.impact = impact
        self.commission = commission
        self.positions = np.array(positions)
        # make sure okay with the first action as default for no experience
        self.actions = np.array(actions)
        self.bincnt = bincnt
        self.base_rar = rar
        if indicators == 'all':
            self.indicators = [indi.pct_sma, indi.rsi, indi.vwpc]
        elif isinstance(indicators, list):
            self.indicators = indicators[:]
        else:
            raise ValueError(f'indicators param must be list or \'all\'')

        # state space: bins*indicators
        num_states = self.bincnt**len(self.indicators)
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
        # clear learner
        self.Q = np.zeros((self.num_states, self.num_actions))
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
        self.scores = np.zeros((self.epochs, 1))
        pxchgs = df.loc[symbol, 'AdjClose']
        pxchgs = (pxchgs/pxchgs.shift(1)-1).dropna()
        pxchgs = pxchgs[pxchgs.index.isin(df_bins.index.values)].values
        # multiplier mapping from indicators to state [1, 10, 100]
        sdi = np.full((len(self.indicators),), self.bincnt)
        sdi = sdi**np.arange(0, len(self.indicators))
        sdt = df_bins.astype(int).values*sdi
        states = sdt.sum(axis=1)
        pfloor = self.positions.min()
        pceil = self.positions.max()
        for epoch in range(1, self.epochs+1):
            a = self.actions[self.querysetstate(states[0])]
            rewards = np.zeros((pxchgs.shape[0],))
            pos = 0
            prior_pos = 0
            evidence = zip(states[:-1], states[1:], pxchgs)
            for i, (s, sp, pchg) in enumerate(evidence):
                next_pos = np.clip(pos+a, pfloor, pceil)
                r = pchg*pos
                if self.impact != 0:
                    r -= abs(prior_pos-pos)/1000*self.impact
                rewards[i] = r
                a = self.actions[self.query(sp, r)]
                prior_pos = pos
                pos = next_pos

            self.scores[epoch-1] = ((pxchgs-rewards)**2).mean()**0.5
            print(f'epoch: {epoch} score: {self.scores[epoch-1]}')

        return self.scores[:]

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
        sdt = np.concatenate((df_bins.values, pos_col), axis=1).astype(int)
        sdi = np.arange(0, (len(self.indicators)+1)*self.bincnt, self.bincnt)
        sdt += sdi
        actions = np.zeros((sdt.shape[0],))
        p = 0
        for i, s in enumerate(sdt):
            a = self.actions[self.Q[s.sum(), :].argmax()]
            actions[i] = np.clip(p+a, -1, 1)-p
            p += actions[i]

        df_actions = pd.DataFrame(actions, index=df_bins.index,
                                  columns=['Shares'])
        df_trades.update(df_actions)
        df_trades *= 1e3
        df_trades = df_trades.rename(columns={'Shares': symbol})
        return df_trades

    def cmp_policy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1),
                   ed=dt.datetime(2009, 12, 31), sv=1e5):
        trades = self.testPolicy(symbol=symbol, sd=sd, ed=ed)
        sp = msim.compute_portvals(trades, start_val=sv,
                                   commission=self.commission,
                                   impact=self.impact)

        buys = (trades[symbol] > 0).sum()
        sells = (trades[symbol] < 0).sum()
        holds = trades.shape[0]-buys-sells
        total = buys+sells+holds
        cr = sp[-1]/sp[0]-1

        qfull = self._qfull()
        qfullpct = qfull/self._qtotal()

        summary = (
            f'- buys: {buys}/{total} ({buys/total:.2f}) '
            f'\n- sells: {sells}/{total} ({sells/total:.2f}) '
            f'\n- holds: {holds}/{total} ({holds/total:.2f}) '
            f'\n- aum: {sp[0]} --> {sp[-1]} ({cr:.2f}) '
            f'\nQ metrics: '
            f'\n- pct full: {qfull} ({qfullpct:.2f}) '
        )
        print(f'\n{summary}')

    def _qempty(self):
        return (self.Q == 0).sum()

    def _qfull(self):
        return self._qtotal()-self._qempty()

    def _qtotal(self):
        return np.prod(self.Q.shape)


if __name__ == "__main__":
    print("One does not simply think up a strategy")
    random.seed(10)
    np.random.seed(10)
    slearner = StrategyLearner(impact=0.0, verbose=False)
    slearner.addEvidence(symbol='AAPL', sd=dt.datetime(2008, 1, 1),
                         ed=dt.datetime(2009, 12, 31))
    slearner.cmp_policy(symbol='AAPL', sd=dt.datetime(2010, 1, 1),
                        ed=dt.datetime(2011, 12, 31))
