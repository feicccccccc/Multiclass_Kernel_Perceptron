"""
Specific Method for
- train: Train the parameters
- predict: Predict the parameters
- reset: Init the parameters (i.e. shape / init value)
and the classifier class will do the rest

i.e.
linear_classifier = Linear_Classifier()
linear_classifier.get_min_m()
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_data_set(n=10, m=100, choice=(-1, 1)):
    # shape (m, n)
    return np.random.choice(choice, (m, n))


class Classifier:
    def __init__(self):
        self.total_run = 100  # total number of run for choose n, result of m is average out
        self.dim = -1
        self.data = -1
        self.min_acc = 0.9  # generalisation error

    def gen_data(self, max_m=1000, n=10, choice=(-1, 1)):
        # and set up corresponding shape
        self.dim = n
        self.data = np.random.choice(choice, (max_m, n))

    def train(self, m):
        # Change the weight here
        raise NotImplementedError

    def predict(self, test_set):
        # return the predicted label here (m, 1)
        raise NotImplementedError

    def reset(self, n):
        # Reset all parameters
        raise NotImplementedError

    def get_acc_test(self, input_samples):
        return np.sum(self.predict(input_samples) == input_samples[:, [0]]) / input_samples.shape[0]

    def get_min_m(self, test_set, start_m=1, choice=(-1, 1)):
        all_m = []
        for run in range(self.total_run):
            # Each run have different sample
            self.gen_data(2000, self.dim, choice=choice)
            for m in range(1, 2000 + 1):
                # a rough lower bound guess can speed up the process
                if m < start_m:
                    continue

                self.train(m)
                cur_acc = self.get_acc_test(test_set)
                # print("cur m {}, cur run {}, cur acc {}".format(m, run, cur_acc))

                if cur_acc > self.min_acc:
                    all_m.append(m)
                    break
            if run == len(range(self.total_run)):
                raise Exception  # Not enough samples
            self.reset(self.dim)
        return np.mean(all_m), np.std(all_m)


class LinearRegression(Classifier):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.weight = -1  # not init

    def train(self, m=10):
        self.weight = np.zeros((self.dim, 1))
        x_pinv = np.linalg.pinv(self.data[:m, :])
        self.weight[:, ] = x_pinv @ self.data[:m, [0]]

    def predict(self, input_samples):
        return np.sign(input_samples @ self.weight)

    def reset(self, n):
        self.dim = n
        self.weight = np.zeros((n, 1))


class Winnow(Classifier):
    def __init__(self):
        super(Winnow, self).__init__()
        self.weight = -1  # not init

    def train(self, m=10):
        for cur_sample_idx in range(m):
            cur_data = self.data[cur_sample_idx, :]
            cur_y = self.data[cur_sample_idx, 0]
            y_hat = self.predict(cur_data)
            if y_hat != cur_y:
                temp = np.expand_dims((cur_y - y_hat) * cur_data, axis=1)
                self.weight = np.multiply(self.weight, np.power(2., temp))

    def predict(self, input_samples):
        return (input_samples @ self.weight >= self.dim).astype(int)

    def reset(self, n):
        self.dim = n
        self.weight = np.ones((n, 1))


class OneNN(Classifier):
    def __init__(self):
        super(OneNN, self).__init__()
        self.history = -1  # not init

    def train(self, m=10):
        # TODO: can optimise by checking if there's any duplicate
        self.history = self.data[:m, :]

    def predict(self, input_samples):
        # We can count the number of zeros but it should be clear with norm
        result = np.zeros((input_samples.shape[0], 1))
        all_dist = np.zeros((input_samples.shape[0], self.history.shape[0]))
        for k in range(input_samples.shape[0]):
            all_dist[k, :] = np.linalg.norm(self.history - input_samples[k, :], axis=1)
        all_idx = np.argmin(all_dist, axis=1)
        result = np.expand_dims(self.history[all_idx, [0]], axis=1)
        return result

    def reset(self, n, m=1):
        self.history = np.zeros((m, n))
        self.dim = n


class Perceptron(Classifier):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.weight = -1  # not init

    def train(self, m=10):
        for cur_sample_idx in range(m):
            cur_data = self.data[cur_sample_idx, :]
            cur_y = self.data[cur_sample_idx, 0]
            y_hat = self.predict(cur_data)
            if y_hat != cur_y:
                self.weight = self.weight + cur_y * np.expand_dims(cur_data, axis=1)

    def predict(self, input_samples):
        return np.sign(input_samples @ self.weight)

    def reset(self, n):
        self.dim = n
        self.weight = np.zeros((n, 1))



def sample_complexity(classifier, choice=(-1, 1)):
    n_history = [x + 1 for x in range(16)]
    m_mean_history = []
    m_std_history = []

    for n in n_history:
        # This is our test set to estimate generalisation error
        test_set = generate_data_set(n=n, m=100, choice=choice)

        classifier.reset(n)

        cur_mean, cur_std = classifier.get_min_m(test_set, start_m=1, choice=choice)

        print("current n: {}, m: {}".format(n, cur_mean))

        m_mean_history.append(cur_mean)
        m_std_history.append(cur_std)

    fig, ax = plt.subplots()
    ax.errorbar(n_history, m_mean_history, yerr=m_std_history, fmt='-o')
    ax.set_xlabel('n')
    ax.set_ylabel('m')
    name = classifier.__class__.__name__
    ax.set_title(name + 'Sample complexity')
    plt.show()
    fig.savefig(name)


if __name__ == "__main__":
    # linear = LinearRegression()
    # sample_complexity(linear)

    # winnow = Winnow()
    # sample_complexity(winnow, choice=(0, 1))

    oneNN = OneNN()
    sample_complexity(oneNN)

    # perceptron = Perceptron()
    # sample_complexity(perceptron)

    # Example training sequence
    # test_data_set = generate_data_set(3, 7)
    # perceptron.reset(3)
    # perceptron.gen_data(1000, 3)
    # perceptron.train(10)
    # print(perceptron.get_acc_test(test_data_set))
    # print(perceptron.predict(test_data_set))
    # print(test_data_set[:, [0]])

