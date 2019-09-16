import numpy as np


class RTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = False
        self.max_depth = 20
        self.tree = np.zeros((2**(self.max_depth+1)-1, 5))
        self.tree[:] = np.nan

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        data = np.concatenate((dataX, dataY[:, None]), axis=1)
        self._builder(0, 0, data)

    def _builder(self, idx, depth, data):
        """
        node: [idx, split_feat, split_val, left_idx, right_idx]
        """
        # stop if examples <= leaf_size
        if data.shape[0] <= self.leaf_size:
            self.tree[idx] = np.array([idx,
                                       np.nan,
                                       data[:, -1].mean(),
                                       np.nan,
                                       np.nan])
            return

        # stop if depth >= max_depth
        if depth >= self.max_depth:
            self.tree[idx] = np.array([idx,
                                       np.nan,
                                       data[:, -1].mean(),
                                       np.nan,
                                       np.nan])
            return

        # stop if y vals all the same
        if np.where(data[:, -1] == data[0][-1], True, False).all():
            self.tree[idx] = np.array([idx,
                                       np.nan,
                                       data[0, -1],
                                       np.nan,
                                       np.nan])
            return

        # stop if x vals all the same
        if np.where(data[:, :-1] == data[0, :-1], True, False).all():
            self.tree[idx] = np.array([idx,
                                       np.nan,
                                       data[:, -1].mean(),
                                       np.nan,
                                       np.nan])
            return

        # update tree with split_feat and split_val
        split_feat = None
        while True:
            split_feat = np.random.randint(0, data.shape[1]-1)
            if not np.where(data[:, split_feat] == data[0, split_feat],
                            True, False).all():
                break

        # pick two points at random
        pts = data[np.random.choice(range(data.shape[0]),
                                    size=2, replace=False)]

        count = 0
        ldata = None
        rdata = None
        while True:
            if count > 10:
                break
            alpha = np.random.rand()
            split_val = alpha*pts[0][split_feat] + (1-alpha)*pts[1][split_feat]

            ldata = data[data[:, split_feat] <= split_val]
            rdata = data[data[:, split_feat] > split_val]

            if ldata.shape[0] > 0 and rdata.shape[0] > 0:
                break

            count += 1

        # stop if no split feat
        if ldata.shape[0] == 0 or rdata.shape[0] == 0:
            self.tree[idx] = np.array([idx,
                                       np.nan,
                                       data[:, -1].mean(),
                                       np.nan,
                                       np.nan])
            return

        ridx = 2**(self.max_depth-depth)
        self.tree[idx] = np.array([idx,
                                   split_feat,
                                   split_val,
                                   1,
                                   ridx])

        # split data and continue recursion
        self._builder(idx+1, depth+1, ldata)
        self._builder(idx+ridx, depth+1, rdata)
        return

    def query(self, points):
        Y = np.zeros((points.shape[0],))
        for i in range(points.shape[0]):
            pt = points[i]
            node = self.tree[0]
            while True:
                idx, split_feat, split_val, left_idx, right_idx = node

                # break if leaf
                if np.isnan(split_feat):
                    break

                # update node based on split_feat
                if pt[int(split_feat)] <= split_val:
                    node = self.tree[int(idx)+int(left_idx)]
                else:
                    node = self.tree[int(idx)+int(right_idx)]

            Y[i] = node[2]
        return Y
