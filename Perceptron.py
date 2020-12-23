import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from misc import *


# Kernel Perceptron
# 1 vs All for Multi-class classification

class KPerceptron():
    def __init__(self, X_train, Y_train, hparams=None):

        if hparams is None:
            hparams = {'kernel': 'poly',
                       'd': 3,
                       'num_class': 3,
                       'epochs': 20,
                       'n_dims': 256}

        self.X_train = X_train  # (n, m)
        self.Y_train = Y_train  # (1, m)
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

        self._kernelMatrics = self._computeKernelMatrics()

    def predict(self, input):
        ker_product = self.kernel(self.X_train, input)
        output = self._alphasMatrics.T @ ker_product  # Dual form computation
        return np.argmax(output)  # return the first occurrence term (break even)


    def train(self):
        mistake_his = []
        # Training Loop
        # Full derivation in report
        for epoch in range(self.epochs):
            cur_mistake = 0
            for i in range(self.m):  # Loop through all sample
                """
                Two variant of update rule
                1. Update only when the final output is incorrect (y =/= f(x))
                2. Update when the individual hyperplane is incorrect, even the final one is correct
                We will use the second method here.
                """
                cur_X = self.X_train[:, i]
                cur_Y = self.Y_train[:, i]

                # Predict
                # y_hat = self.predict(cur_X)  # Not efficient for training
                f_x = self._alphasMatrics.T @ self._kernelMatrics[i]  # Look up precomputed kernel
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

            print("Epoch: {}, Mistakes: {}, Acc: {:.3f}".format(epoch,
                                                                cur_mistake,
                                                                1-(cur_mistake/self.m)))
            mistake_his.append(cur_mistake)
        return mistake_his

    def test(self):
        # Mistakes here is the final accuracy, which does not do multiple count
        # batch prediction
        all_fx = self._alphasMatrics.T @ self._kernelMatrics
        all_y_hat = np.argmax(all_fx, axis=0)
        mistakes = np.sum(self.Y_train == all_y_hat)
        acc = mistakes / self.Y_train.shape[1]
        return mistakes, acc

    def _poly_ker(self, x1, x2, d=3):
        # homogeneous Polynomial kernel (no bias)
        output = np.power((x1.T @ x2), d)
        return output

    def _gauss_ker(self, x1, x2, c=0.1):
        # TODO: implement gaussian kernel
        output = np.power((x1.T @ x2), c)
        return output

    def _computeKernelMatrics(self):
        # Calculate all possible inner product (at the feature space) on the data set
        kerMatrics = self.kernel(self.X_train, self.X_train)
        # (number of samples, number of samples)
        return kerMatrics

    def _test(self):
        print("Test")
        print(self.kernel)
        print(self.weight.shape)


# Test function
if __name__ == '__main__':
    Xtrain, Ytrain = readData('zipcombo.dat')
    ker_perceptron = KPerceptron(Xtrain, Ytrain)

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