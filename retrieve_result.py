import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import glob

from Perceptron import *
from confusion_matrics import *


def q1():
    with open('result/q1/q1.npy', 'rb') as f:
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    print(avg_train_history.shape)

    print("==== Result ====")
    print("train    std     test    std")
    for i in range(avg_train_history.shape[0]):
        print("{:.4f}    {:.4f}    {:.4f}    {:.4f}".format(avg_train_history[i, -1],
                                                            std_train_history[i, -1],
                                                            avg_test_history[i, -1],
                                                            std_test_history[i, -1]))

    log_avg_train_history = np.log(avg_train_history)
    log_avg_test_history = np.log(avg_test_history)

    fig, ax = plt.subplots()
    for d in range((avg_train_history.shape[0])):
        # ax.plot(log_avg_train_history[d, :], '--', label='d = '+str(d+1))
        ax.plot(log_avg_test_history[d, :], '--', label='d = ' + str(d + 1))
        None

    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax.legend()
    plt.show()


def q2():
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    hparams = {'kernel': 'poly',
               'd': 7,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    all_d_his = np.zeros((7, 20))  # (poly, run)

    for run in range(20):
        with open('result/q2/q2_run_' + str(run) + '_all_d_err.npy', 'rb') as f:
            d_history = np.load(f)
            all_d_his[:, run] = d_history[:, 0]

    mean_d = np.mean(all_d_his, axis=1)
    std_d = np.std(all_d_his, axis=1)

    # This is bad just to do max on each 20 run
    # Here we plot the average error to get a better visualisation
    x = np.arange(1, 7 + 1)
    print(mean_d, std_d)
    fig, ax = plt.subplots()
    ax.errorbar(x, mean_d, yerr=std_d, fmt='-o')
    plt.show()


def q3():
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    hparams = {'kernel': 'poly',
               'd': 7,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    test_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)

    # This is bad, we are averaging results with different hparam(d)
    # If the point is to do ensemble on 20 model, I can understand
    # This seems nonsense to me
    all_confmat = np.zeros((10, 10, 20))  # (class, class, run)
    for i in range(20):
        path = glob.glob('./weight/q2/q2_run' + str(i) + '*.npy')[0]
        test_perceptron.load_weight(path)
        pred = test_perceptron.predict(X_test)
        all_confmat[:, :, i] = confusion_matrix(Y_test[0, :], pred)

    conf_fig = plot_confusion_matrix(all_confmat)
    conf_fig.show()


def q4():
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    hparams = {'kernel': 'poly',
               'd': 7,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    test_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
    test_perceptron.load_weight('./weight/q2/q2_run11_d4_weight.npy')
    # Refer to confusion matrrix
    # 2, 3, 4, 8, 9
    # training set have more data, or we can do it in the whole data set
    # We split and shuffle everytime so it doesn't really matter for our purpose (evaluating on data set)
    target = 2
    all_idx = np.where(Y_train == target)[1]

    pred = test_perceptron.predict(X_train[:, all_idx])
    false_idx = all_idx[np.where(pred != target)[0]]
    print(pred)
    print(false_idx)

    for i, idx in enumerate(false_idx):
        if i == 10:
            break
        img = X_train[:, idx].reshape(16, 16)
        plt.imshow(img, cmap='gray')
        plt.show()


def q1g_pre():
    with open('result/q1g/q1g_pre.npy', 'rb') as f:
        c_values = np.load(f)
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    print(avg_train_history.shape)

    print("==== Result ====")
    print("train    std     test    std")
    for i in range(avg_train_history.shape[0]):
        print("{:.4f}    {:.4f}    {:.4f}    {:.4f}".format(avg_train_history[i, -1],
                                                            std_train_history[i, -1],
                                                            avg_test_history[i, -1],
                                                            std_test_history[i, -1]))

    log_avg_train_history = np.log(avg_train_history)
    log_avg_test_history = np.log(avg_test_history)

    fig, ax = plt.subplots()
    for c_idx, c in enumerate(c_values):
        # ax.plot(log_avg_train_history[c_idx, :], '--', label='c = '+str(c))
        ax.plot(log_avg_test_history[c_idx, :], '--', label='c = ' + str(c))
        None

    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax.legend()
    plt.show()


def q1g():
    with open('result/q1g/q1g.npy', 'rb') as f:
        c_values = np.load(f)
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    print(avg_train_history.shape)

    print("==== Result ====")
    print("train    std     test    std")
    for i in range(avg_train_history.shape[0]):
        print("{:.4f}    {:.4f}    {:.4f}    {:.4f}".format(avg_train_history[i, -1],
                                                            std_train_history[i, -1],
                                                            avg_test_history[i, -1],
                                                            std_test_history[i, -1]))

    log_avg_train_history = np.log(avg_train_history)
    log_avg_test_history = np.log(avg_test_history)

    fig, ax = plt.subplots()
    for c_idx, c in enumerate(c_values):
        # ax.plot(log_avg_train_history[c_idx, :], '--', label='c = '+str(c))
        ax.plot(log_avg_test_history[c_idx, :], '--', label='c = ' + str(c))
        None

    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax.legend()
    plt.show()


def q2g():
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    hparams = {'kernel': 'Gauss',
               'c': 0.14,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    with open('result/q1g/q1g.npy', 'rb') as f:
        c_values = np.load(f)
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    all_c_his = np.zeros((len(c_values), 20))  # (c range, run)

    for run in range(20):
        with open('result/q2g/q2g_run_' + str(run) + '_all_c_err.npy', 'rb') as f:
            c_history = np.load(f)
            all_c_his[:, run] = c_history[:, 0]

    mean_c = np.mean(all_c_his, axis=1)
    std_c = np.std(all_c_his, axis=1)

    # This is bad just to do max on each 20 run
    # Here we plot the average error to get a better visualisation
    x = np.arange(0.002,0.022,0.002)
    print(mean_c, std_c)
    fig, ax = plt.subplots()
    ax.errorbar(x, mean_c, yerr=std_c, fmt='-o')
    plt.show()


if __name__ == "__main__":
    q1g()
