from functools import partial
from multiprocessing import Pool
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import sys


N_LABELS = 83


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def jacobian(theta, X, y):
    n, m = X.shape

    alpha = sigmoid(X.dot(theta))
    gradient = (X.T.dot(alpha - y) + 1.0 * theta) / (2.0 * n)

    return gradient


def likelihood(theta, X, y):
    n, m = X.shape
    alpha = sigmoid(X.dot(theta))

    val = (
        -y.dot(np.log(alpha)) - (1.0 - y).dot(np.log(1.0 - alpha)) +
        (1.0 * theta.dot(theta) / 2.0)  # regularization term
    ) / n
    return val


def train_single_label(X, y):
    n, m = X.shape
    theta = np.zeros(m)
    return minimize(
        likelihood,
        theta,
        args=(X, y),
        method='L-BFGS-B',
        options={'disp': True},
        jac=jacobian,
    )


def mle(X, labels, i):
    print '***** Processing column:', i
    result = train_single_label(X, labels[:, i])
    np.savetxt(
        'params/%d.csv' % i,
        result.x,
        delimiter=','
    )


def read_csv(filename):
    return pd.read_csv(
        filename,
        delimiter='\t',
        header=None,
    ).values


def train():
    X = read_csv('data/trainingData.csv')
    nrows, ncols = X.shape

    labels = np.zeros((nrows, N_LABELS))
    with open('data/trainingLabels.txt', 'r') as fp:
        for i, line in enumerate(fp):
            row_labels = map(int, line.split(','))
            labels[i, np.array(row_labels) - 1] = 1

    pool = Pool(5)
    pool.map(partial(mle, X, labels), range(N_LABELS))


def predict():
    X = read_csv('data/testData.csv')
    nrows, ncols = X.shape

    Theta = np.zeros((nrows, N_LABELS))
    for i in xrange(N_LABELS):
        filename = '%d.csv' % i
        Theta[:, i] = np.loadtxt(filename)

    result = sigmoid(X.dot(Theta)) >= 0.5
    np.savetxt('prediction.csv', result, delimiter='\t')


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train()
    else:
        predict()
