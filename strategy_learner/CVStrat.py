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
        total = self._lrner_cnt()
        cnt = 1
        for sym, [sd, ed], lrnrs in trials:
            for lrn in lrnrs:
                print(f'\nTrial {cnt}: {sym} {sd} --> {ed}')
                params = self.hypers[(cnt-1)%len(self.symbols)]
                for k, v in params.items():
                    print(f'{k}: {v}')
                print()
                lrn.addEvidence(symbol=sym, sd=sd, ed=ed)
                cnt += 1

    def test(self, test_periods):
        tests = zip(self.symbols, test_periods, self.learners)
        total = self._lrner_cnt()
        cnt = 1
        for sym, [sd, ed], lrnrs in tests:
            for lrn in lrnrs:
                print(f'\nTest {cnt}: {sym} {sd} --> {ed}')
                lrn.cmp_policy(sym, sd, ed)

    def _lrner_cnt(self):
        return len([l for lrners in self.learners for l in lrners])

    def _get_hypers(self, **kwargs):
        k, p = list(zip(*kwargs.items()))
        combos = list(itertools.product(*p))
        return [dict(zip(k, v)) for v in combos]

if __name__ == '__main__':
    cvs = CVStrat()
    symbols = ['JPM']
    train_periods = [
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            ]
    test_periods = [
            (dt.datetime(2010, 1, 1), dt.datetime(2011, 12, 31)),
            ]
    settings = {'epochs': [10], 'bincnt': [4]}
    cvs.fit(symbols, **settings)
    cvs.train(train_periods)
    cvs.test(test_periods)
