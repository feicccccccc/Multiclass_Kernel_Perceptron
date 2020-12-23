import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from misc import *


# Kernel Perceptron
# 1 vs All for Multi-class classification

class KPerceptron():
    def __init__(self, X_train, Y_train, X_test, Y_test, hparams=None):

        if hparams is None:
            hparams = {'kernel': 'poly',
                       'd': 3,
                       'num_class': 3,
                       'epochs': 20,
                       'n_dims': 256}

        self.X_train = X_train  # (n, m)
        self.Y_train = Y_train  # (1, m)
        self.X_test = X_test
        self.Y_test = Y_test

        self.m = X_train.shape[1]
        self.num_class = hparams['num_class']
        self.epochs = hparams['epochs']

        self.weight = np.zeros((hparams['n_dims'], 1))  # Not used in dual form
        self._alphasMatrics = np.zeros((self.m, self.num_class))  # dual form params, each col represent a hyperplane

        # Encapsulate the hyper parameter c/d
        if hparams['kernel'] == 'poly':
            self.kernel = lambda x1, x2: self._poly_ker(x1, x2, hparams['d'])
        elif hparams['kernel'] == 'gauss':
            self.kernel = lambda x1, x2: self._gauss_ker(x1, x2, hparams['c'])

        # We compute the kernel before hand to reduce repeated computation
        # Equivalence to online learning, see pdf
        self._kernelMatrics_train = self._computeKernelMatrics(X_train)
        self._kernelMatrics_test = self._computeKernelMatrics(X_test)

    def predict(self, input):
        ker_product = self.kernel(input, self.X_train)
        output = self._alphasMatrics.T @ ker_product  # Dual form computation
        return np.argmax(output)  # return the first occurrence term (break even)

    def train(self):
        # Record performance for each epoch
        acc_train_his = []
        acc_test_his = []
        cur_mistake = 0

        # Training Loop
        # Full derivation in report
        for epoch in range(self.epochs):
            cur_mistake = 0
            for i in range(self.m):  # Loop through all sample
                """
                Two variant of update rule
                1. Update only when the final output is incorrect (y =/= f(x))
                2. Update when the individual hyperplane is incorrect, even the final one is correct
                We will use the 2nd method here.
                """
                cur_X = self.X_train[:, i]
                cur_Y = self.Y_train[:, i]

                # Predict
                # y_hat = self.predict(cur_X)  # Not efficient for training
                f_x = self._alphasMatrics.T @ self._kernelMatrics_train[i]  # Look up precomputed kernel
                y_hat = np.argmax(f_x)
                if cur_Y != y_hat:
                    cur_mistake += 1

                # Compare and Update
                """
                Treat each hyperplane as individual and update accordingly
                1. when y_hat == y, ** do nothing ** for the ** alphas for that Class **
                2. when f_x[class] > 0, y == class, y=/= y_hat, ** do nothing **
                    Remark: 
                    We can say it is "Correct", since the individual plane predict the correct label
                    or "Incorrect", since the f_x[class] is not large enough (compare to other plane)
                    It depends on how we view the aggregate of hyperplane and define mistakes.
                    i.e. like WA and HEDGE, we decrease the weight of all others if it is not the correct label
                    
                3. when f_x[class] > 0, y =/= class, alphas for that class decrease by 1
                4. when f_x[class] < 0, y =/= class, do nothing (Correct prediction)
                5. when f_x[class] < 0, y == class, alphas for that class increase by 1
                """
                for c in range(self.num_class):
                    if f_x[c] > 0:
                        # Case 1, 2, 3
                        if c != cur_Y:
                            # alphaMatrics : (m, class)
                            self._alphasMatrics[i][c] -= 1  # Update in dual space, should be obvious
                    if f_x[c] <= 0:
                        # Case 4, 5
                        if c == cur_Y:
                            self._alphasMatrics[i][c] += 1  # if writing down the w update in dual form

            _, cur_acc_train = self._train_err()
            _, cur_acc_test = self._test_err()
            print("Epoch: {}, Mistakes: {}, Acc_train: {:.3f}, Acc_test: {:.3f}".format(epoch,
                                                                                        cur_mistake,
                                                                                        cur_acc_train,
                                                                                        cur_acc_test))
            acc_train_his.append(cur_acc_train)
            acc_test_his.append(cur_acc_test)

        return acc_train_his, acc_test_his

    # A little bit redundant here but it is easier to read in other loop
    def _train_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_train
        all_y_hat = np.argmax(all_fx, axis=0)
        correct = np.sum(self.Y_train == all_y_hat)
        acc = correct / self.Y_train.shape[1]
        return correct, acc

    def _test_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_test
        all_y_hat = np.argmax(all_fx, axis=0)
        correct = np.sum(self.Y_test == all_y_hat)
        acc = correct / self.Y_test.shape[1]
        return correct, acc

    def _poly_ker(self, x1, x2, d=3):
        # homogeneous Polynomial kernel (no bias)
        output = np.power((x2.T @ x1), d)
        return output

    def _gauss_ker(self, x1, x2, c=0.1):
        # TODO: implement gaussian kernel
        output = np.power((x1.T @ x2), c)
        return output

    def _computeKernelMatrics(self, input):
        # Calculate all possible inner product (at the feature space) on the data set
        kerMatrics = self.kernel(input, self.X_train)
        # (number of samples, number of samples)
        return kerMatrics

    def _test(self):
        print("Test")
        print(self.kernel)
        print(self.weight.shape)


# Test function
if __name__ == '__main__':
    X_train, Y_train = readData('zipcombo.dat')
    ker_perceptron = KPerceptron(X_train, Y_train)

    # # Test function for kernel
    # test_x1 = np.array([1, 3, 5])
    # test_x2 = np.array([2, 4, 6])
    # print(ker_perceptron.kernel(test_x1, test_x2))  # (2 + 12 + 30)^3
    #
    # # Check dim
    # print(ker_perceptron._kernelMatrics.shape)
    #
    # # Check predict function
    # print(ker_perceptron.predict(np.random.rand(256, 1)))

    # Check train and test
    ker_perceptron.train()
    print(ker_perceptron.test())
