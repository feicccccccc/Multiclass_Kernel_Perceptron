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


def part_1():
    # Import Data
    X_train, X_test, Y_train, Y_test = readData('zipcombo.dat', split=True)
    num_class = int((np.max(Y_train) - np.min(Y_train)))
    n = X_train.shape[0]
    runs = 1

    hparams = {'kernel': 'poly',
               'd': 1,
               'num_class': num_class,
               'epochs': 20,
               'n_dims': n}

    for d in range(1, 7+1):
        for run in range(runs):
            print("===== d: {}, run: {} =====".format(d, run+1))
            hparams['d'] = d
            ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
            acc_train_his, acc_test_his = ker_perceptron.train()
            print(acc_train_his)
            print(acc_test_his)


if __name__ == '__main__':
    part_1()
