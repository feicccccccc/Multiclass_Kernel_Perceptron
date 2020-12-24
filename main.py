"""
COMP0078 Assigment 2
Author: Cheung Yat Fei
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from Perceptron import *
from misc import *
from Perceptron import *


def q1():
    # Import Data to get hparams
    # We will do the import at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('zipcombo.dat', split=True)
    num_class = int((np.max(dummy_Y_train) - np.min(dummy_Y_train)))
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
            print("===== d: {}, run: {} =====".format(d+1, run+1))
            hparams['d'] = d+1
            X_train, X_test, Y_train, Y_test = readData('zipcombo.dat', split=True)

            ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            cur_err_train_his, cur_err_test_his = ker_perceptron.train()

            err_train_his.append(cur_err_train_his)
            err_test_his.append(cur_err_test_his)

        cur_d_train_history = np.vstack(err_train_his)
        cur_d_test_history = np.vstack(err_test_his)

        avg_train_history[d, :] = np.mean(cur_d_train_history, axis=0)
        std_train_history[d, :] = np.std(cur_d_train_history, axis=0)

        avg_test_history[d, :] = np.mean(cur_d_test_history, axis=0)
        std_test_history[d, :] = np.std(cur_d_test_history, axis=0)

    # with open('./result/q1.npy', 'wb') as f:
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
    # Import Data to get hparams
    # We will split the dataset at random in the run loop
    dummy_X_train, dummy_X_test, dummy_Y_train, dummy_Y_test = readData('zipcombo.dat', split=True)
    num_class = int((np.max(dummy_Y_train) - np.min(dummy_Y_train)))
    n = dummy_X_train.shape[0]

    runs = 1
    ker_poly_d = 7
    max_epochs = 50

    hparams = {'kernel': 'poly',
               'd': 3,
               'num_class': num_class,
               'max_epochs': max_epochs,
               'n_dims': n,
               'early_stopping': True,
               'patience': 5}

    # For record
    avg_train_result = np.zeros((ker_poly_d, 1))
    avg_val_result = np.zeros((ker_poly_d, 1))
    std_train_result = np.zeros((ker_poly_d, 1))
    std_val_result = np.zeros((ker_poly_d, 1))

    for d in range(ker_poly_d):
        err_train = []
        err_val = []

        for run in range(runs):
            print("===== d: {}, run: {} =====".format(d + 1, run + 1))
            hparams['d'] = d + 1
            X_train, X_test, Y_train, Y_test = readData('zipcombo.dat', split=True)

            ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            cur_err_train_his, cur_err_val_his = ker_perceptron.train()

            err_train.append(cur_err_train_his[-1])
            err_val.append(cur_err_val_his[-1])

        cur_d_train = np.array(err_train)
        cur_d_test = np.array(err_val)

        avg_train_result[d] = np.mean(cur_d_train)
        std_train_result[d] = np.std(cur_d_train)

        avg_val_result[d] = np.mean(cur_d_test)
        std_val_result[d] = np.std(cur_d_test)

    # with open('./result/q2.npy', 'wb') as f:
    #     np.save(f, avg_train_history)
    #     np.save(f, std_train_history)
    #     np.save(f, avg_test_history)
    #     np.save(f, std_test_history)
    #
    # print("==== Result ====")
    # print("avg_train_history")
    # print(avg_train_history)
    # print("std_train_history")
    # print(std_train_history)
    # print()
    # print("avg_test_history")
    # print(avg_test_history)
    # print("std_train_history")
    # print(std_test_history)

if __name__ == '__main__':
    q2()
