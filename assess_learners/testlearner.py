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
from RTLearner import RTLearner
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


def compare_dt_rt(trainX, trainY, testX, testY, should_plot=False,
                  max_size=None, data_title=''):
    bound = trainX.shape[0] // 5
    if max_size is not None:
        bound = min(max_size, bound)

    resids = np.zeros((2, testX.shape[0]))
    rsqrs = np.zeros((2, bound))
    aics = np.zeros((2, bound))
    stds = np.zeros((2, bound))
    xrng = np.arange(bound)
    n, k = trainX.shape
    for i in xrng:
        opts = {'leaf_size': i}
        learners = [BagLearner(learner=DTLearner, kwargs=opts, bags=10),
                    BagLearner(learner=RTLearner, kwargs=opts, bags=10)]
        for j, learner in enumerate(learners):
            learner.addEvidence(trainX, trainY)
            predY = learner.query(testX)
            if i == 0:
                resids[j] = (testY - predY)
            rsqrs[j][i] = np.corrcoef(predY, y=testY)[0, 1]**2
            aics[j][i] = 2*k + n*np.log(((testY - predY)**2).sum()/n)
            stds[j][i] = np.std(testY-predY)

    if should_plot:
        # plot R squared, AIC, std vs leaf size
        if data_title != '':
            data_title = f'_{data_title}'

        fig = plt.figure()
        ax = fig.add_subplot(111)
        x = xrng + 1
        ax.plot(x, rsqrs[0], label='DTLearner')
        ax.plot(x, rsqrs[1], label='RTLearner')

        ax.set_xlabel('Leaf Size', fontweight='bold')
        ax.set_ylabel('R-Squared', fontweight='bold')
        ax.set_xbound(lower=1)

        plt.legend()
        plt.savefig(f'dtl_rtl_rsqr{data_title}.png')
        plt.clf()

        ax = fig.add_subplot(111)
        ax.plot(x, aics[0], label='DTLearner')
        ax.plot(x, aics[1], label='RTLearner')

        ax.set_xlabel('Leaf Size', fontweight='bold')
        ax.set_ylabel('AIC', fontweight='bold')
        ax.set_xbound(lower=1)

        plt.legend()
        plt.savefig(f'dtl_rtl_aic{data_title}.png')
        plt.clf()

        ax = fig.add_subplot(111)
        ax.plot(x, stds[0], label='DTLearner')
        ax.plot(x, stds[1], label='RTLearner')

        ax.set_xlabel('Leaf Size', fontweight='bold')
        ax.set_ylabel('STD', fontweight='bold')
        ax.set_xbound(lower=1)

        plt.legend()
        plt.savefig(f'dtl_rtl_std{data_title}.png')
        plt.clf()

        ax = fig.add_subplot(111)
        ax.plot(range(trainX.shape[0]), trainY, 'bo')

        ax.set_ylabel('Y', fontweight='bold')

        plt.savefig(f'y_vals{data_title}.png')
        plt.clf()

        ax = fig.add_subplot(111)
        ax.plot(testY, resids[0], 'bo')

        ax.set_ylabel('Residuals', fontweight='bold')
        ax.set_xlabel('Y', fontweight='bold')

        plt.savefig(f'resids{data_title}.png')
        plt.clf()


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


def split_data(data, verbose=False):
    # compute how much data in training and testing
    train_rows = int(0.6*data.shape[0])

    # split data into training and testing
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]
    if verbose:
        print(f"{testX.shape}")
        print(f"{testY.shape}")

    return trainX, trainY, testX, testY


def check_edge_cases():
    # evaluate each data file with dtlearne
    data_files = ['3_groups', 'ripple', 'simple',
                  'winequality-red', 'winequality-white']
    for name in data_files:
        wqinf = open('Data/' + name + '.csv')
        file_data = [list(s.strip().split(',')) for s in wqinf.readlines()]
        file_data = np.array([list(map(float, vals)) for vals in file_data])

        rtrainX, rtrainY, rtestX, rtestY = split_data(file_data, verbose=True)
        rdtl = DTLearner(leaf_size=50)
        rdtl.addEvidence(rtrainX, rtrainY)
        eval_sample(rdtl, rtrainX, rtrainY, title=f'{name} IS results')
        eval_sample(rdtl, rtestX, rtestY, title=f'{name} OOS results:')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    # read in data of given filename
    inf = open(sys.argv[1])
    data = [list(s.strip().split(',')) for s in inf.readlines()[1:]]
    data = np.array([list(map(float, vals[1:])) for vals in data])

    trainX, trainY, testX, testY = split_data(data, verbose=True)

    # print correlation matrix
    c = np.corrcoef(data, rowvar=False)
    print(f'correlation matrix: Istanbul')
    print(c)

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
    # test_leaf_size(trainX, trainY, testX, testY, max_size=100,
    #               should_plot=True)

    # bagging impact on DTLearner
    # test_bagging(trainX, trainY, testX, testY, max_size=100,
    #              should_plot=True)

    # cmp dtl and rtl
    # compare_dt_rt(trainX, trainY, testX, testY, max_size=100,
    #              should_plot=True, data_title='ist')

    # cmp dtl and rtl for winequality-red dataset
    # wqinf = open('Data/winequality-red.csv')
    # data = [list(s.strip().split(',')) for s in wqinf.readlines()]
    # data = np.array([list(map(float, vals)) for vals in data])

    # print correlation matrix
    # c = np.corrcoef(data, rowvar=False)
    # print(f'correlation matrix: Wine Quality Red')
    # print(c)

    # trainX, trainY, testX, testY = split_data(data, verbose=True)
    # compare_dt_rt(trainX, trainY, testX, testY, max_size=100,
    #               should_plot=True, data_title='weq')
