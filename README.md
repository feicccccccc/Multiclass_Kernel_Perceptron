# Multiclass Kernel Perceptron Library
Simple Library for Multiclass Kernel Perceptron

# Very Simple **Library** on Multiclass Kernel Perceptron implement with pure NumPy

Easy 5 step training sequence.

## 1. import library

```python
from Perceptron import KPerceptron
from misc import readData
```

## 2. define hyper parameters

```python
hparams = {'kernel': 'poly',
           'd': 4,
           'num_class': 10,
           'max_epochs': 5,
           'n_dims': 256,
           'early_stopping': False,
           'patience': 5}
```

| hyperparameters       |          |   |
| ------------- |:-------------:| -----:|
| kernel      | 'poly' or 'gauss' | polynomial kernel or gaussain kernel |
| d      | integer      |   if using polynomial kernel, d specify degree |
| c | float    |    if using gaussian kernel, c specify γ |

Other parameters are self-explanatory.

## 3. Define the instance
```python
X_train, X_test, Y_train, Y_test = readData('data/zipcombo.dat', split=True)
ker_perceptron = KPerceptron(X_train, Y_train, X_test, Y_test, hparams=hparams)
```

## 4. one line train
```python
train_history, test_history = ker_perceptron.train()
```
you can get the history on training error and test error for every epoch

## 5. Get prediction
```python
pred = ker_perceptron.predict(X_test)
```

## 6. save weight for future use
```python
ker_perceptron.save_weight('./weight/test.npy')
```
Warning: The data set is part of the weight. (Kernel perceptron in dual form)

If the data set is huge, the stored parameters will occupie a large space.

## 7. load weight
```python
ker_perceptron.load_weight('./weight/test.npy')
```
