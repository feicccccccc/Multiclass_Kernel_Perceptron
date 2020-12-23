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

if __name__ == '__main__':
    # Import Data
    X_train, X_test, Y_train, Y_test = readData('zipcombo.dat', split=True)
    num_class = int((np.max(Y_train) - np.min(Y_train)))
    n = X_train.shape[0]

    hparams = {'kernel': 'poly',
               'd': 1,
               'num_class': num_class,
               'epochs': 20,
               'n_dims': n}

    ker_perceptron = KPerceptron(X_train, Y_train, hparams=hparams)
    print(ker_perceptron.train())
    print(ker_perceptron.test())

