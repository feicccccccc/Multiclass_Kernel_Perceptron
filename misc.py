# Helper function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def readData(path, reshape=False, split=False):
    """
    Read the data
    Specific to the given file
    :param split:
    :param reshape:
    :param path: str, path to data
    :return: (features, labels)
    """
    # (number of samples, label + data)
    # RegEx to eliminate inconsistent spacing
    arr = pd.read_csv(path, sep='\s+', header=None).to_numpy().T  # (n, m)
    m = arr.shape[1]
    if reshape:
        features = arr[1:, :].reshape((16, 16, m))
    else:
        features = arr[1:, :]

    labels = np.expand_dims(arr[0, :], axis=0).astype(int)

    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(features.T, labels.T, test_size=0.2)
        return X_train.T, X_test.T, Y_train.T, Y_test.T
    else:
        return features, labels


# Test function
if __name__ == '__main__':
    # Data import test
    X, Y = readData('dtrain123.dat', reshape=True)
    print(X.shape)
    print(Y.shape)
    print(np.max(Y))
    print(np.min(Y))

    n = 3
    for i in range(n):
        print(Y[:, i])
        plt.imshow(X[:, :, i], cmap='gray')
        plt.show()
