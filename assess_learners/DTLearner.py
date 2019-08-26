import numpy as np


class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor for decision tree learner.

        params:
        leaf_size - max number of samples aggregated at a leaf
        verbose - debugging flag
        """
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        """
        Training step for decision tree learner
        """
        data = np.concatenate((dataX, dataY[:, None]), axis=1)
        self._root = self._build_tree(data)

    def _build_tree(self, data):
        """
        Recursively constructs decision tree.
        Returns:
        leaf node - [None, mean y, None, None, example cnt]
        tree - [feat idx, split val, ltree, rtree, example cnt]
        """
        # leaf if fewer than leaf_size examples remaining
        if data.shape[0] <= self.leaf_size:
            return [None, np.mean(data[:, -1]), None, None, data.shape[0]]

        # leaf if all targets same value
        if np.all(np.where(data[:, -1] == data[0][-1], True, False)):
            return [None, data[0][-1], None, None, data.shape[0]]

        # get best feature to split on
        idx = self._best_feat(data)
        med = np.median(data[:, idx])

        ldata = data[data[:, idx] <= med]
        rdata = data[data[:, idx] > med]

        if ldata.shape[0] == 0:
            return [None, np.mean(rdata[:, -1]), None, None, rdata.shape[0]]

        if rdata.shape[0] == 0:
            return [None, np.mean(ldata[:, -1]), None, None, ldata.shape[0]]

        ltree = self._build_tree(ldata)
        rtree = self._build_tree(rdata)

        return [idx, med, ltree, rtree, data.shape[0]]

    def _best_feat(self, data):
        """
        Returns idx of feat with highest abs val corr to Y.
        """
        corr_coef = np.corrcoef(data, rowvar=False)
        return np.argmax(np.abs(corr_coef[-1, :-1]))

    def query(self, points):
        """
        Estimate set of test points with previously build decision tree.
        Returns 1d array of estimated values.
        """
        Y = np.zeros((points.shape[0],))
        for i in range(points.shape[0]):
            pt = points[i]
            node = self._root
            while True:
                # break if leaf
                if node[0] is None:
                    break

                # update node based on split val
                idx = node[0]
                split_val = node[1]
                if pt[idx] <= split_val:
                    node = node[2]
                else:
                    node = node[3]

            Y[i] = node[1]
        return Y
