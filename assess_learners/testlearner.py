"""
Test a learner.  (c) 2015 Tucker Balch
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
"""
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import LinRegLearner as lrl
from DTLearner import DTLearner
from BagLearner import BagLearner


def eval_sample(lrner, X, Y, title='', verbose=True):
    predY = lrner.query(X)
    rmse = math.sqrt(((Y - predY)**2).sum() / Y.shape[0])
    c = np.corrcoef(predY, y=Y)
    if verbose:
        print(f'\n{title}')
        print(f'rmse: {rmse}')
        print(f'corr: {c[0, 1]}')
    return rmse, c


def test_leaf_size(trainX, trainY, testX, testY, should_plot=False,
                   max_size=None, verbose=False):
    bound = trainX.shape[0] // 5
    if max_size is not None:
        bound = min(max_size, bound)

    rmses = np.zeros((bound,))
    xrng = np.arange(bound)
    for i in xrng:
        learner = DTLearner(leaf_size=i)
        learner.addEvidence(trainX, trainY)
        predY = learner.query(testX)
        rmses[i] = math.sqrt(((testY - predY)**2).sum()/testY.shape[0])

    if verbose:
        print(f'\n***** Leaf Size Test *****\n')
        print(f"{'Size':<10}{'RMSE':<10}")
        for i in xrng:
            print(f'{i:<10}{rmses[i]:<10.4f}')

    if should_plot:
        # plot RMSE vs leaf size
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrng, rmses, label='DTLearner')

        ax.set_xlabel('Leaf Size', fontweight='bold')
        ax.set_ylabel('RMSE', fontweight='bold')

        plt.legend()
        plt.savefig('dtl_leafsize_fig.png')
        plt.clf()

    return rmses


def test_bagging(trainX, trainY, testX, testY, should_plot=False,
                 max_size=None):
    bound = trainX.shape[0] // 5
    if max_size is not None:
        bound = min(max_size, bound)

    bags = [1, 10, 25]
    rmses = np.zeros((len(bags), bound))
    xrng = np.arange(bound)
    baseline = np.zeros((bound,))

    # DTLearner without bagging
    for i in xrng:
        learner = DTLearner(leaf_size=i)
        learner.addEvidence(trainX, trainY)
        predY = learner.query(testX)
        baseline[i] = math.sqrt(((testY - predY)**2).sum()/testY.shape[0])

    # DTLearner with bagging
    for i, cnt in enumerate(bags):
        for j in xrng:
            learner = BagLearner(learner=DTLearner, bags=cnt,
                                 kwargs={'leaf_size': j})
            learner.addEvidence(trainX, trainY)
            predY = learner.query(testX)
            rmses[i][j] = math.sqrt(((testY - predY)**2).sum()/testY.shape[0])

    if should_plot:
        # plot RMSE vs leaf size for each bag case
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrng, baseline, label='DTLearner')
        for i, cnt in enumerate(bags):
            ax.plot(xrng, rmses[i], label=f'bags={cnt}')

        ax.set_xlabel('Leaf Size', fontweight='bold')
        ax.set_ylabel('RMSE', fontweight='bold')

        plt.legend()
        plt.savefig('dtl_bagging_fig.png')
        plt.clf()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    # read in data of given filename
    inf = open(sys.argv[1])
    data = [list(s.strip().split(',')) for s in inf.readlines()[1:]]
    data = np.array([list(map(float, vals[1:])) for vals in data])

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])
    test_rows = data.shape[0] - train_rows

    # split data into training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]
    print(f"{testX.shape}")
    print(f"{testY.shape}")

    # create linreg learner and train it
    print(f'\n**** LinRegLearner ****')
    lrlearn = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    lrlearn.addEvidence(trainX, trainY)  # train it
    print(lrlearn.author())

    # evaluate learner
    eval_sample(lrlearn, trainX, trainY, title='in-sample results:')
    eval_sample(lrlearn, testX, testY, title='out-of-sample results:')

    # create dt learner and train it
    print(f'\n**** DTLearner ****')
    dtl = DTLearner(leaf_size=50)
    dtl.addEvidence(trainX, trainY)
    print(f'author: {dtl.author()}')

    # evaluate
    eval_sample(dtl, trainX, trainY, title='in-sample results')
    eval_sample(dtl, testX, testY, title='out-of-sample results:')

    # leaf size overfitting impact on DTLearner
    test_leaf_size(trainX, trainY, testX, testY, max_size=100,
                   should_plot=True)

    # bagging impact on DTLearner
    test_bagging(trainX, trainY, testX, testY, max_size=100,
                 should_plot=True)
