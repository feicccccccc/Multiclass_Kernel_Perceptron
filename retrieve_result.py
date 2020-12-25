import numpy as np
import matplotlib.pyplot as plt
from Perceptron import *


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
    # For train, no d = 1
    # plt.ylim([-2.45, -2.25])

    # For valid, no d = 1
    plt.ylim([-2.23, -2.05])

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

    all_d_his = np.zeros((7, 20))  # (run, poly)

    for run in range(20):
        with open('result/q2/q2_run_'+str(run)+'_all_d_err.npy', 'rb') as f:
            d_history = np.load(f)
            all_d_his[:, run] = d_history[:, 0]

    mean_d = np.mean(all_d_his, axis=1)
    std_d = np.std(all_d_his, axis=1)

    x = np.arange(1, 7)
    print(mean_d, std_d)
    fig, ax = plt.subplots()
    ax.plot(x, mean_d)
    ax.errorbar(x, std_d)
    plt.show()



    # test_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)

    # test_perceptron.load_weight('./weight/q2/q2_d3_run0_weight.npy')
    # for i in range(7):
    #     test_perceptron.predict_and_visualise(X_train[:, i])


if __name__ == "__main__":
    q2()
