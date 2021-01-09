"""
Multi-class Kernel Perceptron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from misc import *
from Perceptron import *


def q1g_pre():
    # Import Data to get hparams
    # We will do the import at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('data/zipcombo.dat', split=True)
    num_class = 10
    n = dummy_X_train.shape[0]

    runs = 1
    ker_poly_c = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    epochs = 20

    hparams = {'kernel': 'gauss',
               'c': 0.1,
               'num_class': num_class,
               'max_epochs': epochs,
               'n_dims': n,
               'early_stopping': False,
               'patience': 5}

    # For record
    # TODO: define range for c
    avg_train_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))
    avg_test_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))
    std_train_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))
    std_test_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))

    for c_idx, c in enumerate(ker_poly_c):
        err_train_his = []
        err_test_his = []

        for run in range(runs):
            print("===== c: {}, run: {} =====".format(c, run + 1))
            hparams['c'] = c
            X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

            ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            cur_err_train_his, cur_err_test_his = ker_perceptron.train()

            err_train_his.append(cur_err_train_his)
            err_test_his.append(cur_err_test_his)

        # ker_perceptron.save_weight('./weight/q1g/pre/q1g_c' + str(c) + '_last_weight.npy')
        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[c_idx, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[c_idx, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[c_idx, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[c_idx, :] = np.std(cur_d_test_history, axis=0)

    with open('./result/q1g_pre.npy', 'wb') as f:
        np.save(f, np.array(ker_poly_c))
        np.save(f, avg_train_history)
        np.save(f, std_train_history)
        np.save(f, avg_test_history)
        np.save(f, std_test_history)

    print("==== Result ====")
    print("avg_train_history")
    print(avg_train_history)
    print("std_train_history")
    print(std_train_history)
    print()
    print("avg_test_history")
    print(avg_test_history)
    print("std_train_history")
    print(std_test_history)


def q1(multiclass):
    # Import Data to get hparams
    # We will do the import at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('data/zipcombo.dat', split=True)
    num_class = 10
    n = dummy_X_train.shape[0]

    runs = 20
    ker_poly_d = 7
    epochs = 20

    hparams = {'kernel': 'poly',
               'd': 3,
               'num_class': num_class,
               'max_epochs': epochs,
               'n_dims': n,
               'early_stopping': False,
               'patience': 5}

    # For record
    avg_train_history = np.zeros((ker_poly_d, hparams['max_epochs']))
    avg_test_history = np.zeros((ker_poly_d, hparams['max_epochs']))
    std_train_history = np.zeros((ker_poly_d, hparams['max_epochs']))
    std_test_history = np.zeros((ker_poly_d, hparams['max_epochs']))

    for d in range(ker_poly_d):
        err_train_his = []
        err_test_his = []

        for run in range(runs):
            print("===== d: {}, run: {} =====".format(d + 1, run + 1))
            hparams['d'] = d + 1
            X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

            if multiclass == '1vA':
                ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            if multiclass == '1v1':
                ker_perceptron = KPerceptron_1v1(X_train, Y_train, X_test, Y_test, hparams=hparams)
            if multiclass == 'btree':
                ker_perceptron = KPerceptron_btree(X_train, Y_train, X_test, Y_test, hparams=hparams)

            cur_err_train_his, cur_err_test_his = ker_perceptron.train()

            err_train_his.append(cur_err_train_his)
            err_test_his.append(cur_err_test_his)

        weight_file_name = './weight/q1_d_' + str(multiclass) + '/'
        ker_perceptron.save_weight(weight_file_name + 'q1_d' + str(d + 1) + '_' + str(multiclass) + '_weight.npy')
        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[d, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[d, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[d, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[d, :] = np.std(cur_d_test_history, axis=0)

    result_file_name = './result/q1_d_' + str(multiclass) + '/'
    with open(result_file_name + 'q1_d_' + str(multiclass) + '.npy', 'wb') as f:
        np.save(f, avg_train_history)
        np.save(f, std_train_history)
        np.save(f, avg_test_history)
        np.save(f, std_test_history)

    print("==== Result ====")
    print("avg_train_history")
    print(avg_train_history)
    print("std_train_history")
    print(std_train_history)
    print()
    print("avg_test_history")
    print(avg_test_history)
    print("std_train_history")
    print(std_test_history)


def q1g(multiclass):
    # Import Data to get hparams
    # We will do the import at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('data/zipcombo.dat', split=True)
    num_class = 10
    n = dummy_X_train.shape[0]

    runs = 20
    ker_poly_c = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
    # ker_poly_c = [0.0130, 0.0131, 0.0132, 0.0133, 0.0134, 0.0135, 0.0136, 0.0137, 0.0138, 0.0139]
    epochs = 20

    hparams = {'kernel': 'gauss',
               'c': 0.1,
               'num_class': num_class,
               'max_epochs': epochs,
               'n_dims': n,
               'early_stopping': False,
               'patience': 5}

    # For record
    avg_train_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))
    avg_test_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))
    std_train_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))
    std_test_history = np.zeros((len(ker_poly_c), hparams['max_epochs']))

    for c_idx, c in enumerate(ker_poly_c):
        err_train_his = []
        err_test_his = []

        for run in range(runs):
            print("===== c: {}, run: {} =====".format(c, run + 1))
            hparams['c'] = c
            X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

            if multiclass == '1vA':
                ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            if multiclass == '1v1':
                ker_perceptron = KPerceptron_1v1(X_train, Y_train, X_test, Y_test, hparams=hparams)
            if multiclass == 'btree':
                ker_perceptron = KPerceptron_btree(X_train, Y_train, X_test, Y_test, hparams=hparams)

            cur_err_train_his, cur_err_test_his = ker_perceptron.train()

            err_train_his.append(cur_err_train_his)
            err_test_his.append(cur_err_test_his)

        weight_file_name = './weight/q1_g_' + str(multiclass) + '/'
        ker_perceptron.save_weight(weight_file_name + 'q1_g' + str(c) + '_' + str(multiclass) + '_weight.npy')
        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[c_idx, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[c_idx, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[c_idx, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[c_idx, :] = np.std(cur_d_test_history, axis=0)

    result_file_name = './result/q1_g_' + str(multiclass) + '/'
    with open(result_file_name + 'q1_g_' + str(multiclass) + '.npy', 'wb') as f:
        np.save(f, np.array(ker_poly_c))
        np.save(f, avg_train_history)
        np.save(f, std_train_history)
        np.save(f, avg_test_history)
        np.save(f, std_test_history)

    print("==== Result ====")
    print("avg_train_history")
    print(avg_train_history)
    print("std_train_history")
    print(std_train_history)
    print()
    print("avg_test_history")
    print(avg_test_history)
    print("std_train_history")
    print(std_test_history)


def q2(multiclass):
    # Looping is quite annoying here, tried my best to best name the variable

    # Import Data to get hparams
    # We will split the dataset at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('data/zipcombo.dat', split=True)
    num_class = 10
    n = dummy_X_train.shape[0]

    runs = 20
    ker_poly_d = 7
    max_epochs = 50
    n_fold = 5

    hparams = {'kernel': 'poly',
               'd': 3,
               'num_class': num_class,
               'max_epochs': max_epochs,
               'n_dims': n,
               'early_stopping': True,
               'patience': 5}

    # For record
    # Record best d
    best_d_history = np.zeros((ker_poly_d, runs))

    kf = KFold(n_splits=n_fold)

    for run in range(runs):

        X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

        # store all test/val error
        cur_run_d_historu = np.zeros((ker_poly_d, 1))

        # if run < 15:
        #     continue

        for d in range(ker_poly_d):
            print("===== d: {}, run: {} =====".format(d + 1, run + 1))
            hparams['d'] = d + 1

            cur_CV_err = 0

            # Cross Validation loop
            for train_index, val_index in kf.split(X_train.T):
                # Current training set
                cur_X_train_CV = X_train[:, train_index]
                cur_Y_train_CV = Y_train[:, train_index]

                # Current validation set
                cur_X_val_CV = X_train[:, val_index]
                cur_Y_val_CV = Y_train[:, val_index]

                if multiclass == '1vA':
                    ker_perceptron = KPerceptron(cur_X_train_CV,
                                                 cur_Y_train_CV,
                                                 cur_X_val_CV,
                                                 cur_Y_val_CV,
                                                 hparams=hparams)
                if multiclass == '1v1':
                    ker_perceptron = KPerceptron_1v1(cur_X_train_CV,
                                                     cur_Y_train_CV,
                                                     cur_X_val_CV,
                                                     cur_Y_val_CV,
                                                     hparams=hparams)
                if multiclass == 'btree':
                    ker_perceptron = KPerceptron_btree(cur_X_train_CV,
                                                       cur_Y_train_CV,
                                                       cur_X_val_CV,
                                                       cur_Y_val_CV,
                                                       hparams=hparams)

                cur_err_train_his, cur_err_val_his = ker_perceptron.train()

                cur_CV_err += cur_err_val_his[-1]

            cur_CV_err /= n_fold
            cur_run_d_historu[d] = cur_CV_err

        # Save history
        result_file_name = './result/q2_d_' + str(multiclass) + '/'
        with open(result_file_name + 'q2_run' + str(run) + '_' + str(multiclass) + '_all_d_err.npy', 'wb') as f:
            np.save(f, cur_run_d_historu)

        cur_run_best_d = np.argmin(cur_run_d_historu) + 1
        print("=== Best d: {} for run: {}".format(cur_run_best_d, run + 1))
        hparams['d'] = cur_run_best_d

        if multiclass == '1vA':
            ker_perceptron = KPerceptron(X_train,
                                         Y_train,
                                         X_test,
                                         Y_test,
                                         hparams=hparams)
        if multiclass == '1v1':
            ker_perceptron = KPerceptron_1v1(X_train,
                                             Y_train,
                                             X_test,
                                             Y_test,
                                             hparams=hparams)
        if multiclass == 'btree':
            ker_perceptron = KPerceptron_btree(X_train,
                                               Y_train,
                                               X_test,
                                               Y_test,
                                               hparams=hparams)

        train_err, test_err = ker_perceptron.train()
        weight_file_name = './weight/q2_d_' + str(multiclass) + '/'
        weight_total_name = weight_file_name + 'q2_run' + str(run) + '_d' + str(cur_run_best_d) + '_' + str(
            multiclass) + '_weight.npy'
        ker_perceptron.save_weight(weight_total_name)


def q2g(multiclass):
    # Looping is quite annoying here, tried my best to best name the variable

    # Import Data to get hparams
    # We will split the dataset at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('data/zipcombo.dat', split=True)
    num_class = 10
    n = dummy_X_train.shape[0]

    runs = 20
    ker_poly_c = [0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020]
    max_epochs = 50
    n_fold = 5

    hparams = {'kernel': 'gauss',
               'c': 0.01,
               'num_class': num_class,
               'max_epochs': max_epochs,
               'n_dims': n,
               'early_stopping': True,
               'patience': 5}

    # For record
    # Record best d
    best_c_history = np.zeros((len(ker_poly_c), runs))

    kf = KFold(n_splits=n_fold)

    for run in range(runs):

        X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

        cur_run_c_history = np.zeros((len(ker_poly_c), 1))

        for c_idx, c in enumerate(ker_poly_c):
            print("===== c: {}, run: {} =====".format(c, run + 1))
            hparams['c'] = c

            cur_CV_err = 0

            # Cross Validation loop
            for train_index, val_index in kf.split(X_train.T):
                # Current training set
                cur_X_train_CV = X_train[:, train_index]
                cur_Y_train_CV = Y_train[:, train_index]

                # Current validation set
                cur_X_val_CV = X_train[:, val_index]
                cur_Y_val_CV = Y_train[:, val_index]

                if multiclass == '1vA':
                    ker_perceptron = KPerceptron(cur_X_train_CV,
                                                 cur_Y_train_CV,
                                                 cur_X_val_CV,
                                                 cur_Y_val_CV,
                                                 hparams=hparams)
                if multiclass == '1v1':
                    ker_perceptron = KPerceptron_1v1(cur_X_train_CV,
                                                     cur_Y_train_CV,
                                                     cur_X_val_CV,
                                                     cur_Y_val_CV,
                                                     hparams=hparams)
                if multiclass == 'btree':
                    ker_perceptron = KPerceptron_btree(cur_X_train_CV,
                                                       cur_Y_train_CV,
                                                       cur_X_val_CV,
                                                       cur_Y_val_CV,
                                                       hparams=hparams)

                cur_err_train_his, cur_err_val_his = ker_perceptron.train()

                cur_CV_err += cur_err_val_his[-1]

            cur_CV_err /= n_fold
            cur_run_c_history[c_idx] = cur_CV_err

        # Save history
        result_file_name = './result/q2_g_' + str(multiclass) + '/'
        with open(result_file_name + 'q2_run' + str(run) + '_' + str(multiclass) + '_all_g_err.npy', 'wb') as f:
            np.save(f, cur_run_c_history)

        cur_run_best_c = ker_poly_c[np.argmin(cur_run_c_history)]
        print("=== Best c: {} for run: {}".format(cur_run_best_c, run + 1))
        hparams['c'] = cur_run_best_c

        if multiclass == '1vA':
            ker_perceptron = KPerceptron(X_train,
                                         Y_train,
                                         X_test,
                                         Y_test,
                                         hparams=hparams)
        if multiclass == '1v1':
            ker_perceptron = KPerceptron_1v1(X_train,
                                             Y_train,
                                             X_test,
                                             Y_test,
                                             hparams=hparams)
        if multiclass == 'btree':
            ker_perceptron = KPerceptron_btree(X_train,
                                               Y_train,
                                               X_test,
                                               Y_test,
                                               hparams=hparams)

        train_err, test_err = ker_perceptron.train()
        weight_file_name = './weight/q2_g_' + str(multiclass) + '/'
        weight_total_name = weight_file_name + \
                            'q2_run' + str(run) + '_g' + str(cur_run_best_c) + '_' + str(multiclass) + '_weight.npy'
        ker_perceptron.save_weight(weight_total_name)


def q7_1_knn():
    ks = [1, 2, 3, 4, 5, 6, 7]
    runs = 20

    all_train_err = np.zeros((len(ks), runs))
    all_test_err = np.zeros((len(ks), runs))

    for k in ks:
        err_train_his = []
        err_test_his = []

        for run in range(runs):
            print("===== k: {}, run: {} =====".format(k, run + 1))

            X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

            X_train = X_train.T
            Y_train = Y_train.T[:, 0]

            X_test = X_test.T
            Y_test = Y_test.T[:, 0]

            neigh = KNeighborsClassifier(n_neighbors=k,
                                         weights='uniform',
                                         algorithm='ball_tree',
                                         p=2,
                                         n_jobs=-1)

            neigh.fit(X_train, Y_train)

            pred_train = neigh.predict(X_train)
            pred_test = neigh.predict(X_test)

            cur_err_train = np.sum(pred_train != Y_train) / len(Y_train)
            cur_err_test = np.sum(pred_test != Y_test) / len(Y_test)

            print("err {}".format(cur_err_test))

            all_train_err[k-1, run] = cur_err_train
            all_test_err[k - 1, run] = cur_err_test

    result_file_name = './result/q1_knn/'
    with open(result_file_name + 'q1_knn.npy', 'wb') as f:
        np.save(f, all_train_err)
        np.save(f, all_test_err)

    print("==== Result ====")
    # print("all_train_err")
    # print(all_train_err)
    print("all_test_err")
    print(all_test_err)


def q7_2_knn():
    ks = [1, 2, 3, 4, 5, 6, 7]
    runs = 20
    n_fold = 5

    # For record
    # ((best k, test err), runs)
    all_k_history = np.zeros((2, runs))

    kf = KFold(n_splits=n_fold)

    for run in range(runs):

        X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

        X_train = X_train.T
        Y_train = Y_train.T[:, 0]

        X_test = X_test.T
        Y_test = Y_test.T[:, 0]

        # store all test/val error
        # (number of k, CV err)
        cur_run_k_history = np.zeros((len(ks), 1))

        for k in ks:
            print("===== k: {}, run: {} =====".format(k, run + 1))
            cur_CV_err = 0

            # Cross Validation loop
            for train_index, val_index in kf.split(X_train):
                # Current training set
                cur_X_train_CV = X_train[train_index, :]
                cur_Y_train_CV = Y_train[train_index]

                # Current validation set
                cur_X_val_CV = X_train[val_index, :]
                cur_Y_val_CV = Y_train[val_index]

                neigh = KNeighborsClassifier(n_neighbors=k,
                                             weights='uniform',
                                             algorithm='ball_tree',
                                             p=2,
                                             n_jobs=-1, )

                neigh.fit(cur_X_train_CV, cur_Y_train_CV)

                # No error for 1nn
                # pred_test = neigh.predict(cur_X_train_CV)
                pred_train = neigh.predict(cur_X_val_CV)

                err = np.sum(pred_train != cur_Y_val_CV) / len(pred_train)

                cur_CV_err += err

            cur_CV_err /= n_fold
            cur_run_k_history[k - 1] = cur_CV_err
            print("CV err: {}".format(cur_CV_err))

        cur_run_best_k = np.argmin(cur_run_k_history) + 1
        print("=== Best k: {} for run: {}".format(cur_run_best_k, run + 1))

        neigh = KNeighborsClassifier(n_neighbors=cur_run_best_k,
                                     weights='uniform',
                                     algorithm='ball_tree',
                                     p=2,
                                     n_jobs=-1, )

        neigh.fit(X_train, Y_train)
        pred = neigh.predict(X_test)
        test_err = np.sum(pred != Y_test) / len(pred)

        all_k_history[0, run] = cur_run_best_k
        all_k_history[1, run] = test_err

        print("Cur run {}, Cur k {}, Test err {}".format(run, cur_run_best_k, test_err))

        result_file_name = './result/q2_knn/'
        with open(result_file_name + 'q2_knn.npy', 'wb') as f:
            np.save(f, all_k_history)


def q7_1_lg():
    C = [0.01, 0.1, 1, 10, 100]
    runs = 20

    all_train_err = np.zeros((len(C), runs))
    all_test_err = np.zeros((len(C), runs))

    for (c_idx, c) in enumerate(C):
        err_train_his = []
        err_test_his = []

        for run in range(runs):
            print("===== c: {}, run: {} =====".format(c, run + 1))

            X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

            X_train = X_train.T
            Y_train = Y_train.T[:, 0]

            X_test = X_test.T
            Y_test = Y_test.T[:, 0]

            clf = LogisticRegression(penalty='l2',
                                     tol=1e-4,
                                     multi_class='ovr',
                                     max_iter=10000,
                                     n_jobs=-1,
                                     C=c)

            clf.fit(X_train, Y_train)

            pred_train = clf.predict(X_train)
            pred_test = clf.predict(X_test)

            cur_err_train = np.sum(pred_train != Y_train) / len(Y_train)
            cur_err_test = np.sum(pred_test != Y_test) / len(Y_test)

            print("err {}".format(cur_err_test))

            all_train_err[c_idx, run] = cur_err_train
            all_test_err[c_idx, run] = cur_err_test

    result_file_name = './result/q1_lg/'
    with open(result_file_name + 'q1_lg.npy', 'wb') as f:
        np.save(f, all_train_err)
        np.save(f, all_test_err)

    print("==== Result ====")
    print("all_train_err")
    print(all_train_err)
    print("all_test_err")
    print(all_test_err)


def q7_2_lg():
    C = [0.01, 0.1, 1, 10, 100]
    runs = 20
    n_fold = 5

    # For record
    # ((best k, test err), runs)
    all_c_history = np.zeros((2, runs))

    kf = KFold(n_splits=n_fold)

    for run in range(runs):

        X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

        X_train = X_train.T
        Y_train = Y_train.T[:, 0]

        X_test = X_test.T
        Y_test = Y_test.T[:, 0]

        # store all test/val error
        # (number of k, CV err)
        cur_run_c_history = np.zeros((len(C), 1))

        for (c_idx, c) in enumerate(C):
            print("===== c: {}, run: {} =====".format(c, run + 1))
            cur_CV_err = 0

            # Cross Validation loop
            for train_index, val_index in kf.split(X_train):
                # Current training set
                cur_X_train_CV = X_train[train_index, :]
                cur_Y_train_CV = Y_train[train_index]

                # Current validation set
                cur_X_val_CV = X_train[val_index, :]
                cur_Y_val_CV = Y_train[val_index]

                clf = LogisticRegression(penalty='l2',
                                         tol=1e-4,
                                         multi_class='ovr',
                                         max_iter=10000,
                                         n_jobs=-1,
                                         C=c)

                clf.fit(cur_X_train_CV, cur_Y_train_CV)

                # No error for 1nn
                # pred_test = neigh.predict(cur_X_train_CV)
                pred_train = clf.predict(cur_X_val_CV)

                err = np.sum(pred_train != cur_Y_val_CV) / len(pred_train)

                cur_CV_err += err

            cur_CV_err /= n_fold
            cur_run_c_history[c_idx] = cur_CV_err
            print("CV err: {}".format(cur_CV_err))

        cur_run_best_c = C[np.argmin(cur_run_c_history)]
        print("=== Best k: {} for run: {}".format(cur_run_best_c, run + 1))

        clf = LogisticRegression(penalty='l2',
                                         tol=1e-4,
                                         multi_class='ovr',
                                         max_iter=10000,
                                         n_jobs=-1,
                                         C=cur_run_best_c)

        clf.fit(X_train, Y_train)
        pred = clf.predict(X_test)
        test_err = np.sum(pred != Y_test) / len(pred)

        all_c_history[0, run] = cur_run_best_c
        all_c_history[1, run] = test_err

        print("Cur run {}, Cur c {}, Test err {}".format(run, cur_run_best_c, test_err))

    result_file_name = './result/q2_lg/'
    with open(result_file_name + 'q2_lg.npy', 'wb') as f:
        np.save(f, all_c_history)


if __name__ == '__main__':
    # Result is stored and retrieve in retrieve_result.py
    # This is mainly for training and storing the result

    # q1g_pre()

    # for multiclass in ['1v1', '1vA', 'btree']:
    #     # q1(multiclass)
    #     # q1g(multiclass)
    #     q2(multiclass)
    #     q2g(multiclass)

    # q7_2_lg()

    q7_1_knn()