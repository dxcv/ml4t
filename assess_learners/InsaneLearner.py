import numpy as np
from LinRegLearner import LinRegLearner
from BagLearner import BagLearner


class InsaneLearner(object):
    def __init__(self, verbose=False):
        self.learners = [BagLearner(learner=LinRegLearner, verbose=verbose, bags=20) for _ in range(20)]

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        for learner in self.learners:
            learner.addEvidence(dataX, dataY)

    def query(self, points):
        return np.mean(np.array([learner.query(points) for learner in self.learners]), axis=0)
