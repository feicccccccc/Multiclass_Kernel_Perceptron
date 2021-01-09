import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import glob

from Perceptron import *
from confusion_matrics import *


def q1():
    with open('result/q1_d_btree/q1_d_btree.npy', 'rb') as f:
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    print(avg_train_history.shape)

    print("==== Result ====")
    print("train    std     test    std")
    for i in range(avg_train_history.shape[0]):
        print("\hline")
        print("{} & {:.4f} ± {:.4f} & {:.4f} ± {:.4f} \\\ ".format(i + 1,
                                                                   avg_train_history[i, -1],
                                                                   std_train_history[i, -1],
                                                                   avg_test_history[i, -1],
                                                                   std_test_history[i, -1]))

    log_avg_train_history = np.log(avg_train_history)
    log_err_train_history = np.multiply(1 / avg_train_history, std_train_history)  # Error transformation
    log_avg_test_history = np.log(avg_test_history)

    x_step = [x + 2 for x in range(20)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for d in range((avg_train_history.shape[0])):
        # ax.errorbar(x_step, log_avg_train_history[d, :], fmt='-o', yerr=log_err_train_history[d, :], label='d = ' + str(d + 1))
        ax1.plot(x_step, log_avg_train_history[d, :], '--',
                 label='d = ' + str(d + 1))
        ax2.plot(log_avg_test_history[d, :], '--', label='d = ' + str(d + 1))
        None
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('log error')
    ax1.set_title('Train log error')

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('log error')
    ax2.set_title('Test log error')
    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax1.legend()
    ax2.legend()
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
        path = glob.glob('./result/q2_d_1VA/q2_run' + str(run) + '*.npy')[0]
        with open(path, 'rb') as f:
            d_history = np.load(f)
            all_d_his[:, run] = d_history[:, 0]

    mean_d = np.mean(all_d_his, axis=1)
    std_d = np.std(all_d_his, axis=1)

    log_mean = np.log(mean_d)
    log_std = 1 / mean_d * std_d

    # This is bad just to do max on each 20 run
    # Here we plot the average error to get a better visualisation
    x = np.arange(1, 7 + 1)
    print(mean_d, std_d)
    fig, ax = plt.subplots()
    ax.errorbar(x, log_mean, yerr=log_std, fmt='-o', label="1VA")

    all_d_his = np.zeros((7, 20))  # (poly, run)

    for run in range(20):
        path = glob.glob('./result/q2_d_1V1/q2_run' + str(run) + '*.npy')[0]
        with open(path, 'rb') as f:
            d_history = np.load(f)
            all_d_his[:, run] = d_history[:, 0]

    mean_d = np.mean(all_d_his, axis=1)
    std_d = np.std(all_d_his, axis=1)

    log_mean = np.log(mean_d)
    log_std = 1 / mean_d * std_d

    # This is bad just to do max on each 20 run
    # Here we plot the average error to get a better visualisation
    x = np.arange(1, 7 + 1)
    # print(mean_d, std_d)
    ax.errorbar(x, log_mean, yerr=log_std, fmt='-o', label="1V1")

    all_d_his = np.zeros((7, 20))  # (poly, run)

    for run in range(20):
        path = glob.glob('./result/q2_d_btree/q2_run' + str(run) + '*.npy')[0]
        with open(path, 'rb') as f:
            d_history = np.load(f)
            all_d_his[:, run] = d_history[:, 0]

    mean_d = np.mean(all_d_his, axis=1)
    std_d = np.std(all_d_his, axis=1)

    log_mean = np.log(mean_d)
    log_std = 1 / mean_d * std_d

    # This is bad just to do max on each 20 run
    # Here we plot the average error to get a better visualisation
    x = np.arange(1, 7 + 1)
    # print(mean_d, std_d)
    ax.errorbar(x, log_mean, yerr=log_std, fmt='-o', label="binary tree")

    ax.set_ylabel("average log test error")
    ax.set_xlabel("d")
    ax.legend()
    plt.show()


def q3():

    best_d_1vA = [3, 4, 4, 3, 2, 4, 6, 4, 4, 5, 5, 5, 7, 6, 5, 4, 4, 4, 5, 6]

    best_c_1vA = [0.014, 0.016, 0.018, 0.014, 0.012, 0.012, 0.016, 0.016, 0.016, 0.012, 0.02, 0.016, 0.014, 0.01, 0.02,
                  0.016, 0.014, 0.012, 0.016, 0.018]

    X, Y = readData('data/zipcombo.dat')

    hparams = {'kernel': 'poly',
               'd': 4,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    all_confmat = np.zeros((10, 10, 20))  # (class, class, run)
    all_sample_err_count = np.zeros(Y.shape)

    for i in range(20):
        print(i)
        path = glob.glob('./weight/q2_d_1VA/q2_run' + str(i) + '_d' + str(best_d_1vA[i]) + '*.npy')[0]
        hparams['c'] = best_c_1vA[i]
        test_perceptron = KPerceptron(X, Y, X, Y, hparams=hparams)
        test_perceptron.load_weight(path)

        X_test, Y_test = get_test_set(test_perceptron._stored_Xtrain)
        pred = test_perceptron.predict(X)
        err = (pred != Y).astype(int)
        all_sample_err_count += err
        # for idx, value in enumerate(result[0, :]):
        #     if not value:
        #         plt.imshow(X_test[:, idx].reshape(16, 16), cmap='gray')
        #         name = './images/' + str(Y_test[0, idx]) + '_' + str(pred[idx]) + '_' + str(i) + '.png'
        #         plt.savefig(name)

        all_confmat[:, :, i] = confusion_matrix(Y_test[0, :], pred)

    # print(all_sample_err_count) # [1570, 8261,  323, 8085, 5296]
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
    with open('result/q1g_pre.npy', 'rb') as f:
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
    # for c_idx, c in enumerate(c_values):
    #     # ax.plot(log_avg_train_history[c_idx, :], '--', label='c = '+str(c))
    #     ax.plot(log_avg_test_history[c_idx, :], '--', label='c = ' + str(c))
    #     None

    ax.plot(np.log(c_values), log_avg_test_history[:, -1], '-o', label='c = ' + str(c_values))
    ax.set_xlabel('log c')
    ax.set_ylabel('log err')
    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax.legend()
    plt.show()


def q1g():
    with open('result/q1_g_btree/q1_g_btree.npy', 'rb') as f:
        c_values = np.load(f)
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)

    print(avg_train_history.shape)

    print("==== Result ====")
    print("train    std     test    std")
    for (i, c) in enumerate(c_values):
        print("\hline")
        print("{} & {:.4f} ± {:.4f} & {:.4f} ± {:.4f} \\\ ".format(c,
                                                                   avg_train_history[i, -1],
                                                                   std_train_history[i, -1],
                                                                   avg_test_history[i, -1],
                                                                   std_test_history[i, -1]))

    log_avg_train_history = np.log(avg_train_history)
    log_avg_test_history = np.log(avg_test_history)

    x_step = [x + 2 for x in range(20)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for (i, c) in enumerate(c_values):
        # ax.errorbar(x_step, log_avg_train_history[d, :], fmt='-o', yerr=log_err_train_history[d, :], label='d = ' + str(d + 1))
        ax1.plot(x_step, log_avg_train_history[i, :], '--',
                 label='c = ' + str(c))
        ax2.plot(log_avg_test_history[i, :], '--', label='c = ' + str(c))

    ax1.set_ylabel('log error')
    ax1.set_title('Train log error')

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('log error')
    ax2.set_title('Test log error')
    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax1.legend()
    ax2.legend()
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

    # Get c vlaues
    with open('result/q1_g_1vA/q1_g_1vA.npy', 'rb') as f:
        c_values = np.load(f)
        avg_train_history = np.load(f)
        std_train_history = np.load(f)
        avg_test_history = np.load(f)
        std_test_history = np.load(f)
    fig, ax = plt.subplots()

    all_c_his = np.zeros((len(c_values), 20))  # (c range, run)
    for method in ['1VA', '1V1', 'btree']:
        for run in range(20):
            path = glob.glob('./result/q2_g_' + method + '/q2_run' + str(run) + '*.npy')[0]
            with open(path, 'rb') as f:
                c_history = np.load(f)
                all_c_his[:, run] = c_history[:, 0]

        mean_c = np.mean(all_c_his, axis=1)
        std_c = np.std(all_c_his, axis=1)

        log_mean = np.log(mean_c)
        log_std = 1 / mean_c * std_c

        x = np.arange(0.002, 0.022, 0.002)
        print(mean_c, std_c)
        ax.errorbar(x, log_mean, yerr=log_std, label=method, fmt='-o')

    ax.set_xlabel("c")
    ax.set_ylabel("average log test error")
    ax.legend()
    plt.show()


def q1d1v1():
    with open('result/q1d1v1/q1d1v1.npy', 'rb') as f:
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
        ax.plot(log_avg_train_history[d, :], '--', label='d = ' + str(d + 1))
        # ax.plot(log_avg_test_history[d, :], '--', label='d = ' + str(d + 1))
        None

    # ax.plot(log_avg_train_history[3, :], '--', label='train')
    # ax.plot(log_avg_test_history[3, :], '--', label='test')

    ax.legend()
    plt.show()


def q2d1v1():
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    hparams = {'kernel': 'poly',
               'd': 7,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    all_d_his = np.zeros((7, 20))  # (poly, run)
    best_d_1v1 = [5, 4, 4, 3, 4, 4, 4, 4, 4, 3, 3, 5, 4, 5, 4, 4, 3, 4, 4, 4]  # from stored result
    best_d_1vA = [6, 4, 4, 3, 2, 4, 6, 4, 4, 5, 5, 5, 7, 6, 5, 4, 4, 4, 5, 6]
    best_d_btree = [6, 6, 6, 7, 5, 6, 7, 6, 4, 4, 4, 6, 4, 6, 7, 6, 6, 5, 7, 5]

    for run in range(20):
        with open('result/q2_d_1v1/q2_run' + str(run) + '_1v1_all_d_err.npy', 'rb') as f:
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
    ax.set_xlabel('d')
    ax.set_ylabel('average err')
    plt.show()


def q1_knn():
    with open('result/q1_knn/q1_knn.npy', 'rb') as f:
        all_train_err = np.load(f)
        all_test_err = np.load(f)

    all_mean_train = np.mean(all_train_err, axis=1)
    all_std_train = np.std(all_train_err, axis=1)

    all_mean_test = np.mean(all_test_err, axis=1)
    all_std_test = np.std(all_test_err, axis=1)

    print("==== Result ====")
    print("train    std     test    std")
    for i in range(all_mean_train.shape[0]):
        print("\hline")
        print("{} & {:.4f} ± {:.4f} & {:.4f} ± {:.4f} \\\ ".format(i + 1,
                                                                   all_mean_train[i],
                                                                   all_std_train[i],
                                                                   all_mean_test[i],
                                                                   all_std_test[i]))

    None


def q1_lg():
    with open('result/q1_lg/q1_lg.npy', 'rb') as f:
        all_train_err = np.load(f)
        all_test_err = np.load(f)

    all_mean_train = np.mean(all_train_err, axis=1)
    all_std_train = np.std(all_train_err, axis=1)

    all_mean_test = np.mean(all_test_err, axis=1)
    all_std_test = np.std(all_test_err, axis=1)

    print("==== Result ====")
    print("train    std     test    std")
    for i in range(all_mean_train.shape[0]):
        print("\hline")
        print("{} & {:.4f} ± {:.4f} & {:.4f} ± {:.4f} \\\ ".format(i + 1,
                                                                   all_mean_train[i],
                                                                   all_std_train[i],
                                                                   all_mean_test[i],
                                                                   all_std_test[i]))

def q2_lg():
    with open('result/q2_lg/q2_lg.npy', 'rb') as f:
        all_c_history = np.load(f)

    None

def re_eva_q2():
    """
    Re test on stored weight for q2. Get the test set and testing error on stored parameters obtained by 5-fold CV
    :return:
    """
    X, Y = readData('data/zipcombo.dat')

    hparams = {'kernel': 'gauss',
               'c': 0.016,
               'num_class': 10,
               'max_epochs': 5,
               'n_dims': 256,
               'early_stopping': False,
               'patience': 5}

    best_d_1vA = [3, 4, 4, 3, 2, 4, 6, 4, 4, 5, 5, 5, 7, 6, 5, 4, 4, 4, 5, 6]
    best_d_1v1 = [4, 4, 5, 5, 4, 4, 4, 3, 2, 4, 3, 4, 2, 3, 6, 3, 5, 4, 5, 4]  # from stored result
    best_d_btree = [7, 7, 4, 5, 7, 6, 4, 7, 7, 7, 7, 6, 6, 5, 5, 6, 7, 6, 4, 5]

    best_c_1vA = [0.014, 0.016, 0.018, 0.014, 0.012, 0.012, 0.016, 0.016, 0.016, 0.012, 0.02, 0.016, 0.014, 0.01, 0.02,
                  0.016, 0.014, 0.012, 0.016, 0.018]
    best_c_1v1 = [0.012, 0.014, 0.012, 0.016, 0.014, 0.012, 0.012, 0.016, 0.012, 0.008, 0.016, 0.01, 0.008, 0.01, 0.012,
                  0.008, 0.006, 0.012, 0.012, 0.01]  # from stored result

    best_c_btree = [0.012, 0.014, 0.018, 0.018, 0.018, 0.016, 0.012, 0.016, 0.014, 0.02, 0.018, 0.012, 0.02, 0.016,
                    0.018, 0.018, 0.016, 0.02, 0.02, 0.018]

    # method = 'btree'
    # for run in range(20):
    #     path = glob.glob('./weight/q2_g_' + method + '/q2_run' + str(run) + '_*.npy')[0]
    #     start = path.find('g0')
    #     end = path.find('_' + method + '_weight')
    #     print(path[start + 1:end], end=',')
    #
    # return None

    target = best_c_btree
    all_test_err = []

    method = 'btree'
    for run in range(20):
        hparams['c'] = target[run]
        ker_perceptron = KPerceptron_btree(X, Y, X, Y, hparams=hparams)
        ker_perceptron.load_weight(
            'weight/q2_g_' + method + '/q2_run' + str(run) + '_g' + str(target[run]) + '_' + method + '_weight.npy')
        train_set_list = ker_perceptron._stored_Xtrain.T.tolist()

        X_test, Y_test = get_test_set(ker_perceptron._stored_Xtrain)

        pred_Y = ker_perceptron.predict(X_test)
        err = 1 - (np.sum(Y_test == pred_Y) / Y_test.shape[1])
        print("d: {}, test err {}".format(target[run], err))
        all_test_err.append(err)

    print("mean d {}, std d {}".format(np.mean(target), np.std(target)))
    print("mean err {}, std err {}".format(np.mean(all_test_err), np.std(all_test_err)))


def get_test_set(train_set):
    X, Y = readData('data/zipcombo.dat')
    train_set_list = train_set.T.tolist()

    all_X_list = X.T.tolist()
    all_Y_list = Y.T.tolist()

    test_set_x_list = all_X_list.copy()
    test_set_y_list = all_Y_list.copy()

    for (i, x) in enumerate(train_set_list):
        idx = test_set_x_list.index(x)
        test_set_x_list.pop(idx)
        test_set_y_list.pop(idx)

    X_test = np.array(test_set_x_list).T
    Y_test = np.array(test_set_y_list).T

    return X_test, Y_test


if __name__ == "__main__":
    # re_eva_q2()
    # q2g()
    # q3()
    q1_knn()
