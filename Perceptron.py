import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from misc import *


# Kernel Perceptron
# 1 vs All for Multi-class classification
class KPerceptron:
    def __init__(self, X_train, Y_train, X_val, Y_val, hparams=None):

        if hparams is None:
            hparams = {'kernel': 'poly',
                       'd': 3,
                       'num_class': 3,
                       'max_epochs': 20,
                       'n_dims': 256,
                       'early_stopping': False,
                       'patience': 5}

        self.X_train = X_train  # (n, m)
        self.Y_train = Y_train  # (1, m)
        self.X_val = X_val  # Not following q1 naming, I prefer validation set
        self.Y_val = Y_val  # In the latter part this is use as a validation set

        self.m = X_train.shape[1]
        self.num_class = hparams['num_class']
        self.epochs = hparams['max_epochs']

        # self.weight = np.zeros((hparams['n_dims'], 1))  # Not used in dual form
        self._alphasMatrics = np.zeros((self.m, self.num_class))  # dual form params, each col represent a hyperplane
        self._stored_Xtrain = self.X_train  # Keep a copy of training set to load the weight

        # Encapsulate the hyper parameter c/d
        if hparams['kernel'] == 'poly':
            self.kernel = lambda x1, x2: self._poly_ker(x1, x2, hparams['d'])
        elif hparams['kernel'] == 'gauss':
            self.kernel = lambda x1, x2: self._gauss_ker(x1, x2, hparams['c'])

        self.earlyStopping = hparams['early_stopping']
        self.patience = hparams['patience']

        # We compute the kernel before hand to reduce repeated computation
        # Equivalence to online learning since alpha is init as 0, see pdf
        self._kernelMatrics_train = self._computeKernelMatrics(X_train)
        self._kernelMatrics_val = self._computeKernelMatrics(X_val)

    def save_weight(self, path):
        # save the best result
        with open(path, 'wb') as f:
            np.save(f, self._alphasMatrics)
            np.save(f, self._stored_Xtrain)

    def load_weight(self, path):
        with open(path, 'rb') as f:
            self._alphasMatrics = np.load(f)
            self._stored_Xtrain = np.load(f)

    def predict(self, input):
        # input shape: (n, m)
        ker_product = self.kernel(input, self._stored_Xtrain)
        output = self._alphasMatrics.T @ ker_product  # Dual form computation
        return np.argmax(output)  # return the first occurrence term (break even)

    def train(self):
        # Record performance for each epoch
        err_train_his = []
        err_val_his = []

        not_improve_count = 0
        lowest_val_err = 1

        best_weight = self._alphasMatrics
        best_epoch = 0

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
                            self._alphasMatrics[i][c] -= 1  # Update dual params, should be obvious
                    if f_x[c] <= 0:
                        # Case 4, 5
                        if c == cur_Y:
                            self._alphasMatrics[i][c] += 1  # if writing down the w update in dual form

            _, cur_err_train = self._train_err()
            _, cur_err_val = self._val_err()
            print("Epoch: {}, Mistakes: {}, err_train: {:.3f}, err_val: {:.3f}".format(epoch,
                                                                                        cur_mistake,
                                                                                        cur_err_train,
                                                                                        cur_err_val))
            if self.earlyStopping:
                if lowest_val_err <= cur_err_val:
                    not_improve_count += 1
                else:
                    not_improve_count = 0
                    lowest_val_err = cur_err_val
                    best_weight = self._alphasMatrics
                    best_epoch = epoch

                if not_improve_count == self.patience:
                    print("=== Early Stopping at epoch {}, best result at epoch {} ===".format(epoch, best_epoch))
                    self._alphasMatrics = best_weight
                    break

            err_train_his.append(cur_err_train)
            err_val_his.append(cur_err_val)

        return err_train_his, err_val_his

    def predict_and_visualise(self, input):
        # For 1 sample only
        # TODO: Raise error for incorrect dimension
        result = self.predict(input)
        print("Predicted Label: {}".format(result))
        img = input.reshape(16, 16)
        plt.imshow(img, cmap='gray')
        plt.show()

    # A little bit redundant here but it is easier to read in other loop with different name
    def _train_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_train
        all_y_hat = np.argmax(all_fx, axis=0)
        correct = np.sum(self.Y_train == all_y_hat)
        err = 1 - correct / self.Y_train.shape[1]
        return correct, err

    def _val_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_val
        all_y_hat = np.argmax(all_fx, axis=0)
        correct = np.sum(self.Y_val == all_y_hat)
        err = 1 - correct / self.Y_val.shape[1]
        return correct, err

    # Best I can do without GPU, take up almost 50% of running time
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
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    hparams = {'kernel': 'poly',
               'd': 3,
               'num_class': 10,
               'max_epochs': 10,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    # ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)

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

    # # Check train and test
    # ker_perceptron.train()
    # ker_perceptron.save_weight('./weight/test.npy')

    # # Predict and Visualise
    # ker_perceptron.load_weight('./weight/test.npy')
    # for i in range(7):
    #     ker_perceptron.predict_and_visualise(X_train[:, i])

    # # Check K-fold Validation
    # kf = KFold(n_splits=5)
    # for train_index, val_index in kf.split(X_train.T):
    #     # print(train_index, val_index)
    #     print(X_train[:, train_index].shape)
    #     print(X_train[:, val_index].shape)
