# Helper function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def readData(path):
    """
    Read the data
    Specific to the given file
    :param path: str, path to data
    :return: (features, labels)
    """
    # (number of samples, label + data)
    arr = pd.read_csv(path, sep='   ', header=None, engine='python').to_numpy()
    m = arr.shape[0]
    labels = np.expand_dims(arr[:, 0], axis=1)
    features = arr[:, 1:].reshape((m, 16, 16))
    return features, labels


# Test function
if __name__ == '__main__':
    X, Y = readData('dtrain123.dat')
    n = 5
    for i in range(n):
        print(Y[i])
        plt.imshow(X[i], cmap='gray')
        plt.show()