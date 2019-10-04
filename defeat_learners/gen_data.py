"""
template for generating data to fool learners (c) 2016 Tucker Balch
Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved
Template code for CS 4646/7646
Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.
We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.
-----do not edit anything above this line---
Student Name: Christopher Fleisher
GT User ID: cfleisher3
GT ID: 903421975
"""
import numpy as np


# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    np.random.seed(seed)
    feat_cnt = np.random.randint(2, 11)
    weights = np.random.random((feat_cnt,))
    X = np.random.random((1000, feat_cnt))
    Y = np.sum(weights*X, axis=1)
    return X, Y


def best4DT(seed=1489683273):
    np.random.seed(seed)
    feat_cnt = np.random.randint(2, 5)

    # correlated features
    # MU = np.random.random((feat_cnt,))
    # norm pos semi-def with bias
    # bias = 0.25
    # C = np.random.random((feat_cnt, feat_cnt))+bias
    # C[np.diag_indices(feat_cnt)] = [C[i,:].sum()+1-C[i, i] for i in range(feat_cnt)]
    # C[np.diag_indices(feat_cnt)] = [C[i, i:].sum()+1 for i in range(feat_cnt)]
    # C = C+C.T-np.diag(C.diagonal())

    # corr data from cholesky decomposition
    # L = np.linalg.cholesky(C)
    # X_uncorr = np.random.multivariate_normal(MU, C, 1000)
    # X = (np.dot(L, X_uncorr.T)+MU[:, None]).T

    # nonlinearity
    X = np.random.random((1000, feat_cnt))
    # X_norm = (X-X.mean(axis=0))/np.std(X, axis=0)
    # Y = np.sin(X_norm.sum(axis=1))
    Y = np.prod(X, axis=1)
    
    # data = np.concatenate((X_norm, Y[:, None]), axis=1)
    # print(f'data shape:{data.shape}')
    # y_corrs = np.corrcoef(data, rowvar=False)[-1]
    # print(f'correlations: {y_corrs}')
    return X, Y


def author():
    return 'cfleisher3'


if __name__ == "__main__":
    print("they call me Tim.")
