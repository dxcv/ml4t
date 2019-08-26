import numpy as np


class BagLearner(object):
    def __init__(self, learner=None, kwargs={}, bags=10, boost=False, verbose=False):
        opts = {**kwargs, **{'verbose': verbose}}
        self.learners = [learner(**opts) for _ in range(bags)]

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        '''
        Distributes data across bags, with replacement.
        '''
        for learner in self.learners:
            idxs = np.random.choice(dataX.shape[0], dataX.shape[0])
            trainX, trainY = dataX[idxs], dataY[idxs]
            learner.addEvidence(trainX, trainY)

    def query(self, points):
        '''
        Average prediction of ensemble.
        '''
        outs = np.array([learner.query(points) for learner in self.learners])
        return np.mean(outs, axis=0)
