import itertools
import datetime as dt
from StrategyLearner import StrategyLearner


class CVStrat():
    def __init__(self, k=5, verbose=True):
        self.k = k
        self.verbose = verbose

    def fit(self, symbols, **kwargs):
        self.symbols = symbols[:]
        self.hypers = self._get_hypers(**kwargs)
        self.learners = []
        for _ in symbols:
            self.learners += [[StrategyLearner(**p) for p in self.hypers]]

        if self.verbose:
            print('\nCV Settings')
            print(f'symbols ({len(symbols)}): {symbols}')
            for k, v in kwargs.items():
                print(f'{k} ({len(v)}): {v}')
            print(f'learners ready: {self._lrner_cnt()}\n')

    def train(self, train_periods):
        trials = zip(self.symbols, train_periods, self.learners)
        cnt = 1
        for sym, [sd, ed], lrnrs in trials:
            for lrn in lrnrs:
                print(f'\nTrial {cnt}: {sym} {sd} --> {ed}')
                params = self.hypers[(cnt-1) % len(self.hypers)]
                for k, v in params.items():
                    print(f'{k}: {v}')
                print()
                lrn.addEvidence(symbol=sym, sd=sd, ed=ed)
            cnt += 1

    def test(self, test_periods):
        tests = zip(self.symbols, test_periods, self.learners)
        # cnt = 1
        results = []
        for sym, [sd, ed], lrnrs in tests:
            for lrn in lrnrs:
                # print(f'\nTest {cnt}: {sym} {sd} --> {ed}')
                results.append(lrn.cmp_policy(sym, sd, ed))
        print(f'\nTest Results')
        for i, d in enumerate(results):
            sym, [sd, ed] = self.symbols[i], test_periods[i]
            cr, buys, sells, = d['cr'], d['buys'], d['sells']
            holds, tot = d['holds'], d['total']
            explored, pctexplored = d['explored'], d['pctexplored']
            msg = (
                f'{i+1}. {sym} {sd} --> {ed} '
                f'\n- cr: {cr:.2f} '
                f'\n- explored: {explored} ({pctexplored:.2f}) '
                f'\n- buys: {buys}/{tot} ({buys/tot: .2f}) '
                f'\n- sells: {sells}/{tot} ({sells/tot:.2f}) '
                f'\n- holds: {holds}/{tot} ({holds/tot:.2f})\n '
            )
            print(msg)

    def _lrner_cnt(self):
        return len([l for lrners in self.learners for l in lrners])

    def _get_hypers(self, **kwargs):
        k, p = list(zip(*kwargs.items()))
        combos = list(itertools.product(*p))
        return [dict(zip(k, v)) for v in combos]


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt
    import indicators as indi

    if False:
        target = 'AAPL'
        df_ml4t = indi.ml4t_load_data([target],
                                      pd.date_range(dt.datetime(2008, 1, 1),
                                                    dt.datetime(2009, 12, 31)))
        plt.plot(df_ml4t.loc[target, 'AdjClose'])
        plt.show()

    cvs = CVStrat()
    symbols = ['ML4T-220', 'AAPL', 'SINE_FAST_NOISE', 'UNH']
    train_periods = [
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            ]
    test_periods = [
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            ]
    settings = {'epochs': [20], 'dyna': [100], 'bincnt': [4], 'alpha': [0.2],
                'gamma': [0.9], 'rar': [0.6], 'radr': [0.9999]}
    cvs.fit(symbols, **settings)
    cvs.train(train_periods)
    cvs.test(train_periods)
