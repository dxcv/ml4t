import itertools
import datetime as dt
from StrategyLearner import StrategyLearner


class CVStrat():
    def __init__(self, k=5, verbose=True):
        self.k = k
        self.verbose = verbose

    def fit(self, symbols, periods, **kwargs):
        self.hypers = self._get_hypers(**kwargs)
        self.learners = []
        for _ in symbols:
            self.learners += [StrategyLearner(**p) for p in self.hypers]

        if self.verbose:
            print('\nCV Settings')
            print(f'symbols ({len(symbols)}): {symbols}')
            for k, v in kwargs.items():
                print(f'{k} ({len(v)}): {v}')
            print(f'learners ready: {len(self.learners)}\n')

    def train(self):
        pass

    def _get_hypers(self, **kwargs):
        k, p = list(zip(*kwargs.items()))
        combos = list(itertools.product(*p))
        return [dict(zip(k, v)) for v in combos]

if __name__ == '__main__':
    cvs = CVStrat()
    symbols = ['JPM', 'AAPL']
    periods = [
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31)),
            (dt.datetime(2008, 1, 1), dt.datetime(2009, 12, 31))
            ]
    settings = {'epochs': [100, 150], 'bincnt': [4, 10]}
    cvs.fit(symbols, periods, **settings)
