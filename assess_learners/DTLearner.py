import numpy as np


class DTLearner(object):
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor for decision tree learner.
        Tree np array with cols having following structure:
        [node idx, split feat, split val, left idx, right idx]

        params:
        leaf_size - max number of samples aggregated at a leaf
        verbose - debugging flag
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.max_depth = 10
        self.tree = np.zeros((2**(self.max_depth+1)-1, 5))
        self.tree[:] = np.nan

    def author(self):
        return 'cfleisher3'

    def addEvidence(self, dataX, dataY):
        """
        Training step for decision tree learner
        """
        data = np.concatenate((dataX, dataY[:, None]), axis=1)
        # self._root = self._build_tree(data)
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
        split_feat = self._get_feat(data)

        # stop if no split feat
        if split_feat is None:
            self.tree[idx] = np.array([idx,
                                       np.nan,
                                       data[:, -1].mean(),
                                       np.nan,
                                       np.nan])

        split_val = np.median(data[:, split_feat])
        ridx = 2**(self.max_depth-depth)
        self.tree[idx] = np.array([idx,
                                   split_feat,
                                   split_val,
                                   1,
                                   ridx])

        # split data and continue recursion
        ldata = data[data[:, split_feat] <= split_val]
        rdata = data[data[:, split_feat] > split_val]
        self._builder(idx+1, depth+1, ldata)
        self._builder(idx+ridx, depth+1, rdata)
        return

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
        """
        Returns idx of feat with highest abs val corr to Y.
        """
        # split_feat = np.argmax(np.abs(corr_coef[-1, :-1]))
        # return np.argmax(np.abs(corr_coef[-1, :-1]))
        # make sure data splits o/w return None
        corr_coef = np.corrcoef(data, rowvar=False)
        coef_idxs = list(range(corr_coef[-1, :-1].shape[0]))
        while True:
            # find best feature that splits the data
            if len(coef_idxs) == 0:
                return None
            coef_vals = corr_coef[-1, coef_idxs]
            split_feat = coef_idxs[np.argmax(np.abs(coef_vals))]
            split_val = np.median(data[:, split_feat])
            ldata = data[data[:, split_feat] <= split_val]
            rdata = data[data[:, split_feat] > split_val]
            if ldata.shape[0] > 0 and rdata.shape[0] > 0:
                return split_feat

            coef_idxs.remove(split_feat)

    def query(self, points):
        """
        Estimate set of test points with previously build decision tree.
        Returns 1d array of estimated values.
        """
        Y = np.zeros((points.shape[0],))
#         for i in range(points.shape[0]):
#             pt = points[i]
#             node = self._root
#             while True:
#                 # break if leaf
#                 if node[0] is None:
#                     break
#
#                 # update node based on split val
#                 idx = node[0]
#                 split_val = node[1]
#                 if pt[idx] <= split_val:
#                     node = node[2]
#                 else:
#                     node = node[3]
#
#             Y[i] = node[1]
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
