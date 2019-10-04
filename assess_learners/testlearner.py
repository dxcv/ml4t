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


def get_istanbul_data():
    inf = open('Data/Istanbul.csv')
    data = [list(s.strip().split(',')) for s in inf.readlines()[1:]]
    return np.array([list(map(float, vals[1:])) for vals in data])


def split_data_into_trials():
    data = get_istanbul_data()
    train_pct = 0.6
    train_rows = int(train_pct*data.shape[0])
    trials = 10

    tms = np.array([np.full(data.shape[0], False) for _ in np.arange(trials)])
    for mask in tms:
        mask[:train_rows] = True
        np.random.shuffle(mask)

    train_data = np.array([data[mask] for mask in tms])
    test_data = np.array([data[~mask] for mask in tms])
    trainX = train_data[:, :, :-1]
    trainY = train_data[:, :, -1]
    testX = test_data[:, :, :-1]
    testY = test_data[:, :, -1]
    return trainX, trainY, testX, testY


def dtl_leaf_size_rmses(trainX, trainY, testX, testY, train_pct=0.6):
    trials = trainX.shape[0]
    leaf_rng = np.arange(1, trainX.shape[1] // 5)
    rmses_train = np.zeros((trials, leaf_rng.shape[0]))
    rmses_test = np.zeros((trials, leaf_rng.shape[0]))
    for trial_idx in np.arange(trials):
        for leaf_size, _ in enumerate(leaf_rng):
            dtl = DTLearner(leaf_size=leaf_size)
            dtl.addEvidence(trainX[trial_idx], trainY[trial_idx])
            train_predY = dtl.query(trainX[trial_idx])
            test_predY = dtl.query(testX[trial_idx])
            trobs = trainY.shape[1]
            teobs = testY.shape[1]
            tr = math.sqrt(((trainY[trial_idx] - train_predY)**2).sum()/trobs)
            te = math.sqrt(((testY[trial_idx] - test_predY)**2).sum()/teobs)
            rmses_train[trial_idx][leaf_size] = tr
            rmses_test[trial_idx][leaf_size] = te

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(leaf_rng, rmses_train.mean(axis=0),
            label=f'IS ({train_pct*100:0.0f}%)')
    ax.plot(leaf_rng, rmses_test.mean(axis=0),
            label=f'OOS ({(1-train_pct)*100:0.0f}%)')
    ax.plot(leaf_rng, rmses_test.mean(axis=0)-rmses_train.mean(axis=0),
            label=f'OOS-IS', c='m')
    ax.axvspan(6, 20, color='g', alpha=0.4)

    ax.set_xlim((1, leaf_rng[-10]))
    ax.set_xlabel('Leaf Size', fontweight='bold')
    ax.set_ylabel(f'RMSE (avg over {trials} trials)', fontweight='bold')
    ax.set_title(f'DTLearner Generalization Error', fontweight='bold')
    plt.legend()
    plt.savefig('dtl_leaf_size_rmses_v1.png')
    plt.clf()
    return rmses_train, rmses_test


def bgl_leaf_size_rmses(trainX, trainY, testX, testY,
                        rmses_train, rmses_test, train_pct=0.6):
    trials = trainX.shape[0]
    leaf_rng = np.arange(1, trainX.shape[1] // 5)
    bag_rmses_train = np.zeros((trainX.shape[0], leaf_rng.shape[0]))
    bag_rmses_test = np.zeros((trainX.shape[0], leaf_rng.shape[0]))
    bags = 25

    for trial_idx in np.arange(trials):
        for leaf_idx, leaf_size in enumerate(leaf_rng):
            bgl = BagLearner(learner=DTLearner, bags=bags,
                             kwargs=dict(leaf_size=leaf_size))
            bgl.addEvidence(trainX[trial_idx], trainY[trial_idx])

            train_predY = bgl.query(trainX[trial_idx])
            test_predY = bgl.query(testX[trial_idx])

            trobs = trainY.shape[1]
            teobs = testY.shape[1]

            trv = math.sqrt(((trainY[trial_idx]-train_predY)**2).sum()/trobs)
            tev = math.sqrt(((testY[trial_idx]-test_predY)**2).sum()/teobs)

            bag_rmses_train[trial_idx][leaf_idx] = trv
            bag_rmses_test[trial_idx][leaf_idx] = tev

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(leaf_rng, bag_rmses_train.mean(axis=0),
            label=f'IB ({train_pct*100:0.0f}%)')
    ax.plot(leaf_rng, bag_rmses_test.mean(axis=0),
            label=f'OOB ({(1-train_pct)*100:0.0f}%)')
    ax.plot(leaf_rng, bag_rmses_test.mean(axis=0)-bag_rmses_train.mean(axis=0),
            label=f'OOB-IB', c='m')

    ax.set_xlim((1, leaf_rng[-10]))
    ax.set_xlabel(f'Leaf Size', fontweight='bold')
    ax.set_ylabel(f'RMSE (avg over {trials} trials)', fontweight='bold')
    ax.set_title(f'BagLearner Generalization Error', fontweight='bold')
    plt.legend()
    plt.savefig('bgl_leaf_sizes_rmses_v1.png')
    plt.clf()

    ax = fig.add_subplot(111)
    dt_gen_err = rmses_test.mean(axis=0)-rmses_train.mean(axis=0)
    bag_gen_err = bag_rmses_test.mean(axis=0)-bag_rmses_train.mean(axis=0)
    ax.plot(leaf_rng, dt_gen_err-bag_gen_err, c='m')

    ax.set_xlim((1, leaf_rng[-10]))
    ax.set_xlabel('Leaf Size', fontweight='bold')
    ax.set_ylabel(f'RMSE (avg over {trials} trials)', fontweight='bold')
    ax.set_title(f'DTLearner - BagLearner Generalization Error',
                 fontweight='bold')
    plt.savefig('dtl_bgl_gen_err_v1.png')
    plt.clf()


def dtbg_preds(trainX, trainY, testX, testY, train_pct=0.6):
    bag_rng = np.arange(1, 10)
    leaf_size = 6
    trials = 10
    bagdt_preds_train = np.zeros((trainX.shape[0],
                                  bag_rng.shape[0],
                                  trainX.shape[1]))
    bagdt_preds_test = np.zeros((testX.shape[0],
                                 bag_rng.shape[0],
                                 testX.shape[1]))
    bagrt_preds_train = np.zeros((trainX.shape[0],
                                  bag_rng.shape[0],
                                  trainX.shape[1]))
    bagrt_preds_test = np.zeros((testX.shape[0],
                                 bag_rng.shape[0],
                                 testX.shape[1]))

    for trial_idx in np.arange(trials):
        for bag_idx, bag_size in enumerate(bag_rng):
            bgl = BagLearner(learner=DTLearner, bags=bag_size,
                             kwargs=dict(leaf_size=leaf_size))
            bgl.addEvidence(trainX[trial_idx], trainY[trial_idx])
            bgl2 = BagLearner(learner=RTLearner, bags=bag_size,
                              kwargs=dict(leaf_size=leaf_size))
            bgl2.addEvidence(trainX[trial_idx], trainY[trial_idx])

            train_predY = bgl.query(trainX[trial_idx])
            test_predY = bgl.query(testX[trial_idx])
            train2_predY = bgl2.query(trainX[trial_idx])
            test2_predY = bgl2.query(testX[trial_idx])

            bagdt_preds_train[trial_idx][bag_idx] = train_predY
            bagdt_preds_test[trial_idx][bag_idx] = test_predY
            bagrt_preds_train[trial_idx][bag_idx] = train2_predY
            bagrt_preds_test[trial_idx][bag_idx] = test2_predY

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bagdt_mean_var = (bagdt_preds_test.std(axis=0)**2).mean(axis=1)
    bagrt_mean_var = (bagrt_preds_test.std(axis=0)**2).mean(axis=1)

    ax.plot(bag_rng, bagdt_mean_var, label=f'DT OOB Var')
    ax.plot(bag_rng, bagrt_mean_var, label=f'DT OOB Var')

    ax.set_xlim((1, bag_rng[-1]))
    ax.set_xlabel('Bag Size', fontweight='bold')
    ax.set_ylabel('Prediction Variance', fontweight='bold')
    ax.set_title('DTLearner and RTLearner Prediction Variance',
                 fontweight='bold')
    plt.legend()
    plt.savefig('dtrt_pred_var_v1.png')
    plt.clf()


def dtrt_nodes(trainX, trainY, testX, testY, train_pct=0.6):
    trials = trainX.shape[0]
    dt_counts = np.zeros((trials,), dtype=np.uint)
    rt_counts = np.zeros((trials,), dtype=np.uint)
    leaf_size = 6
    for idx in np.arange(trials):
        dtl = DTLearner(leaf_size=leaf_size)
        dtl.addEvidence(trainX[idx], trainY[idx])
        dt_counts[idx] = dtl.tree[~np.isnan(dtl.tree[:, 2])].shape[0]

        rtl = RTLearner(leaf_size=leaf_size)
        rtl.addEvidence(trainX[idx], trainY[idx])
        rt_counts[idx] = rtl.tree[~np.isnan(rtl.tree[:, 2])].shape[0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xlocs = np.arange(1, trials+1)
    bar_w = 0.35

    ax.bar(xlocs-bar_w/2, dt_counts, bar_w, label='DTLearner')
    ax.bar(xlocs+bar_w/2, rt_counts, bar_w, label='RTLearner')

    ax.set_xlabel('Trial', fontweight='bold')
    ax.set_ylabel('Node Count', fontweight='bold')
    ax.set_xticklabels(np.arange(1, trials+1))
    ax.set_xticks(np.arange(1, trials+1))
    ax.set_title('DTLearner and RTLearner Node Counts', fontweight='bold')
    plt.legend(loc=4)
    plt.savefig('dtrt_nodes_v1.png')
    plt.clf()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    # read in data of given filename
    inf = open(sys.argv[1])
    data = [list(s.strip().split(',')) for s in inf.readlines()[1:]]
    data = np.array([list(map(float, vals[1:])) for vals in data])

    trainX, trainY, testX, testY = split_data_into_trials()
    rmses_train, rmses_test = dtl_leaf_size_rmses(trainX, trainY, testX, testY)
    bgl_leaf_size_rmses(trainX, trainY, testX, testY,
                        rmses_train, rmses_test)
    dtbg_preds(trainX, trainY, testX, testY)
    dtrt_nodes(trainX, trainY, testX, testY)

    # trainX, trainY, testX, testY = split_data(data, verbose=True)

    # print correlation matrix
    # c = np.corrcoef(data, rowvar=False)
    # print(f'correlation matrix: Istanbul')
    # print(c)

    # create linreg learner and train it
    # print(f'\n**** LinRegLearner ****')
    # lrlearn = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    # lrlearn.addEvidence(trainX, trainY)  # train it
    # print(lrlearn.author())

    # evaluate learner
    # eval_sample(lrlearn, trainX, trainY, title='in-sample results:')
    # eval_sample(lrlearn, testX, testY, title='out-of-sample results:')

    # create dt learner and train it
    # print(f'\n**** DTLearner ****')
    # dtl = DTLearner(leaf_size=50)
    # dtl.addEvidence(trainX, trainY)
    # print(f'author: {dtl.author()}')

    # evaluate
    # eval_sample(dtl, trainX, trainY, title='in-sample results')
    # eval_sample(dtl, testX, testY, title='out-of-sample results:')

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
