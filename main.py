"""
Multi-class Kernel Perceptron
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix

from Perceptron import *
from misc import *
from Perceptron import *


def q1():
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

            ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            cur_err_train_his, cur_err_test_his = ker_perceptron.train()

            err_train_his.append(cur_err_train_his)
            err_test_his.append(cur_err_test_his)

        # ker_perceptron.save_weight('./weight/q1/q1_d'+str(d+1)+'_last_weight.npy')
        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[d, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[d, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[d, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[d, :] = np.std(cur_d_test_history, axis=0)

    # with open('./result/q1/q1.npy', 'wb') as f:
    #     np.save(f, avg_train_history)
    #     np.save(f, std_train_history)
    #     np.save(f, avg_test_history)
    #     np.save(f, std_test_history)

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


def q2():
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

        cur_run_d_historu = np.zeros((ker_poly_d, 1))

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

                ker_perceptron = KPerceptron(cur_X_train_CV,
                                             cur_Y_train_CV,
                                             cur_X_val_CV,
                                             cur_Y_val_CV,
                                             hparams=hparams)

                cur_err_train_his, cur_err_val_his = ker_perceptron.train()

                cur_CV_err += cur_err_val_his[-1]

            cur_CV_err /= n_fold
            cur_run_d_historu[d] = cur_CV_err

        # Save history
        with open('./result/q2/q2_run_' + str(run) + '_all_d_err.npy', 'wb') as f:
            np.save(f, cur_run_d_historu)

        cur_run_best_d = np.argmin(cur_run_d_historu) + 1
        print("=== Best d: {} for run: {}".format(cur_run_best_d, run))
        hparams['d'] = cur_run_best_d
        ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
        ker_perceptron.train()
        ker_perceptron.save_weight('./weight/q2/q2_run' + str(run) + '_d' + str(cur_run_best_d) + '_weight.npy')


def q1g_pre():
    # Import Data to get hparams
    # We will do the import at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('data/zipcombo.dat', split=True)
    num_class = 10
    n = dummy_X_train.shape[0]

    runs = 1
    ker_poly_c = [0.001, 0.01, 0.1, 1, 10, 100]
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

        ker_perceptron.save_weight('./weight/q1g/pre/q1g_c' + str(c) + '_last_weight.npy')
        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[c_idx, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[c_idx, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[c_idx, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[c_idx, :] = np.std(cur_d_test_history, axis=0)

    with open('./result/q1g/q1g_pre.npy', 'wb') as f:
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


def q1g():
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

        ker_perceptron.save_weight('./weight/q1g/q1g_c' + str(c) + '_last_weight.npy')
        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[c_idx, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[c_idx, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[c_idx, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[c_idx, :] = np.std(cur_d_test_history, axis=0)

    with open('./result/q1g/q1g_3rd.npy', 'wb') as f:
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


def q2g():
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

                ker_perceptron = KPerceptron(cur_X_train_CV,
                                             cur_Y_train_CV,
                                             cur_X_val_CV,
                                             cur_Y_val_CV,
                                             hparams=hparams)

                cur_err_train_his, cur_err_val_his = ker_perceptron.train()

                cur_CV_err += cur_err_val_his[-1]

            cur_CV_err /= n_fold
            cur_run_c_history[c_idx] = cur_CV_err

        # Save history
        with open('./result/q2g/q2g_run_' + str(run) + '_all_c_err.npy', 'wb') as f:
            np.save(f, cur_run_c_history)

        cur_run_best_c = ker_poly_c[np.argmin(cur_run_c_history)]
        print("=== Best c: {} for run: {}".format(cur_run_best_c, run))
        hparams['c'] = cur_run_best_c
        ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
        ker_perceptron.train()
        ker_perceptron.save_weight('./weight/q2g/q2g_run' + str(run) + '_c' + str(cur_run_best_c) + '_weight.npy')


if __name__ == '__main__':
    # Result is stored and retrieve in retrieve_result.py
    q1g()

