from Perceptron import KPerceptron
from misc import readData

if __name__ == "__main__":
    hparams = {'kernel': 'poly',
               'd': 4,
               'num_class': 10,
               'max_epochs': 5,
               'n_dims': 256,
               'early_stopping': False,
               'patience': 5}

    X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)
    ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
    ker_perceptron.train()
    pred = ker_perceptron.predict(X_test)

    # ker_perceptron.save_weight('./weight/test.npy')
    # ker_perceptron.load_weight('./weight/test.npy')