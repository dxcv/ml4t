import numpy as np
from LinRegLearner import LinRegLearner
from BagLearner import BagLearner


class InsaneLearner(object):
    def __init__(self, verbose=False):
        opts = {'learner': LinRegLearner, 'verbose': verbose, 'bags': 20}
        self.lrns = [BagLearner(**opts) for _ in range(20)]

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        for learner in self.lrns:
            learner.addEvidence(dataX, dataY)

    def query(self, X):
        return np.mean(np.array([lrn.query(X) for lrn in self.lrns]), axis=0)
