import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = False

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        data = np.concatenate((dataX, dataY[:, None]), axis=1)
        self._root = self._build_tree(data)

    def _build_tree(self, data):
        # leaf if fewer than leaf_size examples remaining
        if data.shape[0] <= self.leaf_size:
            return [None, np.mean(data[:, -1]), None, None, data.shape[0]]

        # leaf if all targets same value
        if np.all(np.where(data[:, -1] == data[0][-1], True, False)):
            return [None, data[0][-1], None, None, data.shape[0]]

        # get best feature to split on
        idx = self._get_feat(data)
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

    def _get_feat(self, data):
        '''
        Returns random feature
        '''
        return np.random.randint(0, data.shape[1]-1)

    def query(self, points):
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
