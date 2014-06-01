import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


N_LABELS = 83


df = pd.io.parsers.read_csv(
    'data/trainingData.csv',
    sep='\t',
    header=None,
)
nrows, ncols = df.shape

labels = pd.DataFrame(np.zeros((nrows, N_LABELS)))
with open('data/trainingLabels.txt', 'r') as fp:
    for i, line in enumerate(fp):
        row_labels = map(int, line.split(','))
        labels[i: i + 1][pd.Series(row_labels) - 1] = 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def likelihood2(theta, X, y):
    alpha = sigmoid(X * theta)
    return y * np.log(alpha) + (1.0 - y) * np.log(1.0 - alpha)


def likelihood(theta, X, y):
    ret = 0.0
    n, m = X.shape
    for i in xrange(n):
        alpha = sigmoid(theta.T * X.iloc[[i]])
        ret += y[i] * np.log(alpha) + (1.0 - y[i]) * np.log(1.0 - alpha)
    return -ret


def train(X, y):
    n, m = X.shape
    theta = np.zeros(m)
    theta, _, _ = fmin_l_bfgs_b(
        likelihood2,
        theta,
        args=(X, y),
    )
