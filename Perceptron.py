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

    def predict(self, input, conf = False):
        # input shape: (n, m)
        ker_product = self.kernel(input, self._stored_Xtrain)
        output = self._alphasMatrics.T @ ker_product  # Dual form computation
        if conf:
            return np.argmax(output, axis=0), np.choose(np.argmax(output, axis=0), output)
        else:
            return np.argmax(output, axis=0)  # return the first occurrence term (break even)

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
                            self._alphasMatrics[i, c] -= 1  # Update dual params, should be obvious
                    if f_x[c] <= 0:
                        # Case 4, 5
                        if c == cur_Y:
                            self._alphasMatrics[i, c] += 1  # if writing down the w update in dual form

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
    def _poly_ker(self, t, x, d=4):
        # homogeneous Polynomial kernel (no bias)
        output = np.power((x.T @ t), d)
        return output

    def _gauss_ker(self, t, x, c=0.01):
        # Vectorised at best by numpy
        t_sq = np.sum(np.power(t, 2), axis=0, keepdims=True)  # (num_test, 1)
        x_sq = np.sum(np.power(x, 2), axis=0, keepdims=True)  # (num_train, 1)
        xTt = x.T @ t
        output = np.exp(-c * (t_sq - 2 * xTt + x_sq.T))  # col broadcast x1sq, row broadcast x2sq
        return output

    def _computeKernelMatrics(self, input):
        # Calculate all possible inner product (at the feature space) on the data set
        kerMatrics = self.kernel(input, self.X_train)
        # (number of input samples, number of output samples)
        return kerMatrics

    def _test(self):
        print("Test")
        print(self.kernel)
        print(self.weight.shape)


# TODO: should inherit from KPerceptron instead to reduce repeated code
class KPerceptron_1v1:
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
        self.Y_val = Y_val

        self.m = X_train.shape[1]
        self.num_class = hparams['num_class']
        self.epochs = hparams['max_epochs']

        # self.weight = np.zeros((hparams['n_dims'], 1))  # Not used in dual form
        total_plane = int(self.num_class * (self.num_class - 1) / 2)
        self._allpairs = []  # Use list for ease of implementation for searching
        for i in range(hparams['num_class']):
            for j in range(i + 1, hparams['num_class']):
                self._allpairs.append((i, j))
        self._allpairs = np.array(self._allpairs)  # Turn into np array for ease of computation

        self._alphasMatrics = np.zeros((self.m, total_plane))  # (number of sample, n C 2)
        self._stored_Xtrain = self.X_train  # Keep a copy of training set, part of the "weight"

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

    # Not for training time and testing time. There are separate method to speed up the calculation
    def predict(self, input):
        # input shape: (n, m)
        """
        1 vs 1 Prediction:
        There will be 45 hyperplane for 10 class label (10 C 2)
        Each hyperplane represent binary classification for all possible combination
        i.e. (0,1) (0,2) .... (0,9) (1,2) (1,3) ... (1,9)... (8,9)
        This is slow and just for the sake of the report. Scale with O(n^2)
        * Not implemented, not worth it to do *
        Why would I do extra thing if I will get my mark deducted and discouraged?
        I did it anyway =/, coz of my OCD
        We can use do a binary tree like to achieve best memory and computational complexity.
        i.e. (0:4, 5:9) -> (0:2, 3:4) , (5:7, 8:9) -> ...
        It only required O(log n) hyperplane and its corresponding computation for each hyperplane.
        """
        ker_product = self.kernel(input, self._stored_Xtrain)
        output = self._alphasMatrics.T @ ker_product  # Dual form computation

        all_pred = np.zeros(np.shape(output))
        all_pred[np.where(output > 0)] = self._allpairs[np.where(output > 0)[0]][:, 0]
        all_pred[np.where(output <= 0)] = self._allpairs[np.where(output <= 0)[0]][:, 1]
        all_pred = all_pred.astype(int)

        # Vote for each sample
        all_y_hat = np.zeros((1, input.shape[1]))  # TODO: Better define row and col vector assignment
        for idx, sample in enumerate(all_pred.T):
            bin_count = np.bincount(sample, minlength=self.num_class)
            y_hat = np.argmax(bin_count)  # Break even for choosing the lower class, This is not a good practise.
            all_y_hat[:, idx] = y_hat

        return all_y_hat

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
                It is not much different from 1vsAll
                Now we have 45 classifier instead of 10 and we update accordingly
                i.e. if the true label is 1, we update all classifier with 1 in it
                Remark: We can still use all training data even if they are not in the classifier,
                since we do not update alpha and not related alpha will always be zero,
                which contribute nothing to the solution (for particular 1vs1)
                """
                cur_X = self.X_train[:, i]
                cur_Y = self.Y_train[:, i]

                # Predict, comment out to speed up computation
                # y_hat = self.predict(cur_X)  # Not efficient for training
                f_x = self._alphasMatrics.T @ self._kernelMatrics_train[i]  # Look up precomputed kernel
                # we only need f_x for training for each hyperplane
                # y_hat = np.argmax(f_x)
                # if cur_Y != y_hat:
                #     cur_mistake += 1

                for plane_idx, pair in enumerate(self._allpairs):  # Loop through all rows
                    """
                    Update when it match the pair
                    i.e. (1,3) cur_Y:2 do nothing
                    (1,3) cur_Y:3 f_x:1 -> wrong, discourage
                    (1,3) cur_Y:3 f_x:3 -> correct, encourage
                    Interested reader can compare to binary case perceptron and show they are the same update
                    """

                    if pair[0] == cur_Y:
                        # Doing exhaustive search on each pair, could be optimise
                        if f_x[plane_idx] <= 0:  # False negative
                            self._alphasMatrics[i, plane_idx] += 1

                    if pair[1] == cur_Y:
                        if f_x[plane_idx] > 0:  # False positive
                            self._alphasMatrics[i, plane_idx] -= 1

            _, cur_err_train = self._train_err()
            _, cur_err_val = self._val_err()
            print("Epoch: {}, Mistakes: {}, err_train: {:.3f}, err_val: {:.3f}".format(epoch,
                                                                                       int(cur_err_train * self.m),
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
        if len(input.shape) != 2:  # Make sure dimension matches
            input = np.expand_dims(input, axis=1)
        result = self.predict(input)
        print("Predicted Label: {}".format(result))
        img = input.reshape(16, 16)
        plt.imshow(img, cmap='gray')
        plt.show()

    # A little bit redundant here but it is easier to read in other loop with different name
    def _train_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_train
        # Get the indices, turn the element in that indices base on the pair generated (label if fx>0, vice versa)
        # It is nice to set a break point here and see how it works
        all_pred = np.zeros(np.shape(all_fx))
        all_pred[np.where(all_fx > 0)] = self._allpairs[np.where(all_fx > 0)[0]][:, 0]
        all_pred[np.where(all_fx <= 0)] = self._allpairs[np.where(all_fx <= 0)[0]][:, 1]
        all_pred = all_pred.astype(int)

        # Vote for each sample
        all_y_hat = np.zeros(np.shape(self.Y_train))
        for idx, sample in enumerate(all_pred.T):
            bin_count = np.bincount(sample, minlength=self.num_class)
            y_hat = np.argmax(bin_count)  # Break even for choosing the lower class, This is not a good practise.
            all_y_hat[:, idx] = y_hat

        correct = np.sum(self.Y_train == all_y_hat)
        err = 1 - correct / self.Y_train.shape[1]
        return correct, err

    def _val_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_val

        all_pred = np.zeros(np.shape(all_fx))
        all_pred[np.where(all_fx > 0)] = self._allpairs[np.where(all_fx > 0)[0]][:, 0]
        all_pred[np.where(all_fx <= 0)] = self._allpairs[np.where(all_fx <= 0)[0]][:, 1]
        all_pred = all_pred.astype(int)

        # Vote for each sample
        all_y_hat = np.zeros(np.shape(self.Y_val))
        for idx, sample in enumerate(all_pred.T):
            bin_count = np.bincount(sample, minlength=self.num_class)
            y_hat = np.argmax(bin_count)  # Break even for choosing the lower class, This is not a good practise.
            all_y_hat[:, idx] = y_hat

        correct = np.sum(self.Y_val == all_y_hat)
        err = 1 - correct / self.Y_val.shape[1]
        return correct, err

    # Best I can do without GPU, take up almost 50% of running time
    def _poly_ker(self, t, x, d=4):
        # homogeneous Polynomial kernel (no bias)
        output = np.power((x.T @ t), d)
        return output

    def _gauss_ker(self, t, x, c=0.01):
        # Vectorised at best by numpy
        t_sq = np.sum(np.power(t, 2), axis=0, keepdims=True)  # (num_test, 1)
        x_sq = np.sum(np.power(x, 2), axis=0, keepdims=True)  # (num_train, 1)
        xTt = x.T @ t
        output = np.exp(-c * (t_sq - 2 * xTt + x_sq.T))  # col broadcast x1sq, row broadcast x2sq
        return output

    def _computeKernelMatrics(self, input):
        # Calculate all possible inner product (at the feature space) on the data set
        kerMatrics = self.kernel(input, self.X_train)
        # (number of input samples, number of output samples)
        return kerMatrics

    def _test(self):
        print("Test")
        print(self.kernel)
        print(self.weight.shape)


class KPerceptron_btree:
    # achieve lowest computational time by splitting the class into binary tree
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
        self.Y_val = Y_val

        self.m = X_train.shape[1]
        self.num_class = hparams['num_class']
        self.epochs = hparams['max_epochs']

        # self.weight = np.zeros((hparams['n_dims'], 1))  # Not used in dual form
        total_plane = int(np.ceil(np.log2(hparams['num_class'])))  # depth of the tree
        total_node = 9

        # split the tree from middle just for trying out
        # actually we can do a tree splitting optimisation
        # not generalised to any number of class
        # we dont really need this, the tree is hardcoded.
        # send me a pull request if you implement base on user define tree =)
        # should be easy to implement with tree search algo

        # we can also use gini to split the tree on all possible split. It is going to be really time consuming
        # for multiple run

        self._all_node = [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                          (0, 1, 2, 3, 4),
                          (5, 6, 7, 8, 9),
                          (0, 1),
                          (2, 3, 4),
                          (5, 6),
                          (7, 8, 9),
                          (3, 4),
                          (8, 9)]  # Just for sake of illustration, should use a matrices for the graph(tree)
        # search space scale with O(2^N)

        self._alphasMatrics = np.zeros((self.m, total_node))  # (number of sample, number of node in tree)
        self._stored_Xtrain = self.X_train  # Keep a copy of training set, part of the "weight"

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

    # Not for training time and testing time. There are separate method to speed up the calculation
    def predict(self, input):
        # input shape: (n, m)
        """
        btree Prediction
        """
        ker_product = self.kernel(input, self._stored_Xtrain)
        output = self._alphasMatrics.T @ ker_product  # Dual form computation

        # Cannot vectorised (across samples) since we don't know which node each sample belong to
        # this is going to be really slow...
        all_y_hat = np.zeros((1, input.shape[1]))
        for m in range(output.shape[1]):
            if output[0, m] > 0:
                if output[1, m] > 0:
                    if output[3, m] > 0:
                        all_y_hat[:, m] = 0
                    else:
                        all_y_hat[:, m] = 1
                elif output[4, m] > 0:
                    all_y_hat[:, m] = 2
                elif output[7, m] > 0:
                    all_y_hat[:, m] = 3
                else:
                    all_y_hat[:, m] = 4
            else:
                if output[2, m] > 0:
                    if output[5, m] > 0:
                        all_y_hat[:, m] = 5
                    else:
                        all_y_hat[:, m] = 6
                elif output[6, m] > 0:
                    all_y_hat[:, m] = 7
                elif output[8, m] > 0:
                    all_y_hat[:, m] = 8
                else:
                    all_y_hat[:, m] = 9

        return all_y_hat

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
                cur_X = self.X_train[:, i]
                cur_Y = self.Y_train[:, i]

                f_x = self._alphasMatrics.T @ self._kernelMatrics_train[i]  # Look up precomputed kernel

                # Hardcode is bad practise =(
                """
                i.e.  (5,6,7,8,9)
                f_x>0|           |f_x <=0
                (5,6)           (7,8,9)
                """
                for hplane_idx, node in enumerate(self._all_node):
                    if cur_Y in node:  # to skip necessary loop
                        boundary = int((max(node) + min(node) + 1) / 2)
                        if cur_Y < boundary and f_x[hplane_idx] <= 0:  # Beware of >= and >
                            self._alphasMatrics[i, hplane_idx] += 1
                        if cur_Y >= boundary and f_x[hplane_idx] > 0:
                            self._alphasMatrics[i, hplane_idx] -= 1

            _, cur_err_train = self._train_err()
            _, cur_err_val = self._val_err()
            print("Epoch: {}, Mistakes: {}, err_train: {:.3f}, err_val: {:.3f}".format(epoch,
                                                                                       int(cur_err_train * self.m),
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
        if len(input.shape) != 2:  # Make sure dimension matches
            input = np.expand_dims(input, axis=1)
        result = self.predict(input)
        print("Predicted Label: {}".format(result))
        img = input.reshape(16, 16)
        plt.imshow(img, cmap='gray')
        plt.show()

    # A little bit redundant here but it is easier to read in other loop with different name
    def _train_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_train
        all_y_hat = np.zeros(self.Y_train.shape)
        for m in range(all_fx.shape[1]):
            if all_fx[0, m] > 0:
                if all_fx[1, m] > 0:
                    if all_fx[3, m] > 0:
                        all_y_hat[:, m] = 0
                    else:
                        all_y_hat[:, m] = 1
                elif all_fx[4, m] > 0:
                    all_y_hat[:, m] = 2
                elif all_fx[7, m] > 0:
                    all_y_hat[:, m] = 3
                else:
                    all_y_hat[:, m] = 4
            else:
                if all_fx[2, m] > 0:
                    if all_fx[5, m] > 0:
                        all_y_hat[:, m] = 5
                    else:
                        all_y_hat[:, m] = 6
                elif all_fx[6, m] > 0:
                    all_y_hat[:, m] = 7
                elif all_fx[8, m] > 0:
                    all_y_hat[:, m] = 8
                else:
                    all_y_hat[:, m] = 9

        correct = np.sum(self.Y_train == all_y_hat)
        err = 1 - correct / self.Y_train.shape[1]
        return correct, err

    def _val_err(self):
        all_fx = self._alphasMatrics.T @ self._kernelMatrics_val
        all_y_hat = np.zeros(self.Y_val.shape)

        for m in range(all_fx.shape[1]):
            if all_fx[0, m] > 0:
                if all_fx[1, m] > 0:
                    if all_fx[3, m] > 0:
                        all_y_hat[:, m] = 0
                    else:
                        all_y_hat[:, m] = 1
                elif all_fx[4, m] > 0:
                    all_y_hat[:, m] = 2
                elif all_fx[7, m] > 0:
                    all_y_hat[:, m] = 3
                else:
                    all_y_hat[:, m] = 4
            else:
                if all_fx[2, m] > 0:
                    if all_fx[5, m] > 0:
                        all_y_hat[:, m] = 5
                    else:
                        all_y_hat[:, m] = 6
                elif all_fx[6, m] > 0:
                    all_y_hat[:, m] = 7
                elif all_fx[8, m] > 0:
                    all_y_hat[:, m] = 8
                else:
                    all_y_hat[:, m] = 9

        correct = np.sum(self.Y_val == all_y_hat)
        err = 1 - correct / self.Y_val.shape[1]
        return correct, err

    # Best I can do without GPU, take up almost 50% of running time
    def _poly_ker(self, t, x, d=4):
        # homogeneous Polynomial kernel (no bias)
        output = np.power((x.T @ t), d)
        return output

    def _gauss_ker(self, t, x, c=0.01):
        # Vectorised at best by numpy
        t_sq = np.sum(np.power(t, 2), axis=0, keepdims=True)  # (num_test, 1)
        x_sq = np.sum(np.power(x, 2), axis=0, keepdims=True)  # (num_train, 1)
        xTt = x.T @ t
        output = np.exp(-c * (t_sq - 2 * xTt + x_sq.T))  # col broadcast x1sq, row broadcast x2sq
        return output

    def _computeKernelMatrics(self, input):
        # Calculate all possible inner product (at the feature space) on the data set
        kerMatrics = self.kernel(input, self.X_train)
        # (number of input samples, number of output samples)
        return kerMatrics

    def _test(self):
        print("Test")
        print(self.kernel)
        print(self.weight.shape)


# Test function
if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)

    # hparams = {'kernel': 'poly',
    #            'd': 3,
    #            'num_class': 10,
    #            'max_epochs': 10,
    #            'n_dims': 256,
    #            'early_stopping': True,
    #            'patience': 5}

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

    # # Gauss kernel
    # hparams = {'kernel': 'gauss',
    #            'd': 3,
    #            'c': 0.1,
    #            'num_class': 10,
    #            'max_epochs': 20,
    #            'n_dims': 256,
    #            'early_stopping': False,
    #            'patience': 5}
    #
    # ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
    # print(ker_perceptron.train())

    # # 1 vs 1 Multiclass perceptron
    hparams = {'kernel': 'poly',
               'd': 4,
               'num_class': 10,
               'max_epochs': 20,
               'n_dims': 256,
               'early_stopping': True,
               'patience': 5}

    # hparams = {'kernel': 'gauss',
    #            'c': 0.014,
    #            'num_class': 10,
    #            'max_epochs': 20,
    #            'n_dims': 256,
    #            'early_stopping': False,
    #            'patience': 5}
    #
    ker_perceptron = KPerceptron_btree(X_train, Y_train, X_test, Y_test, hparams=hparams)
    ker_perceptron.train()
    print(ker_perceptron.predict(X_test[:, 0:7]))
    # print(Y_test[:, 0:7])
    # ker_perceptron = KPerceptron_1v1(X_train, Y_train, X_test, Y_test, hparams=hparams)
    # ker_perceptron.train()
    # print(ker_perceptron.predict(X_test[:, 0:7]))
    # print(Y_test[:, 0:7])

    None
