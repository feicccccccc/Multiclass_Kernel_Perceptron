# Comp0078 Assignment 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import misc


def phi_pN(X, N):
    m = X.shape[0]
    if N == 1:
        out = np.ones(m).reshape(m, 1)
    else:
        # recursion for append the power
        out = np.append(phi_pN(X, N - 1), np.power(X, N - 1), axis=-1)
    return out

if __name__ == '__main__':
    # Import Data
    Xtrain, Ytrain

