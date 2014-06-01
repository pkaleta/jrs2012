import pandas as pd
import numpy as np
from scipy.optimize import minimize


N_LABELS = 83

X = pd.read_csv(
    'data/trainingData.csv',
    delimiter='\t',
    header=None,
).values
nrows, ncols = X.shape

labels = np.zeros((nrows, N_LABELS))
with open('data/trainingLabels.txt', 'r') as fp:
    for i, line in enumerate(fp):
        row_labels = map(int, line.split(','))
        labels[i, np.array(row_labels) - 1] = 1


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def likelihood(theta, X, y):
    alpha = sigmoid(X.dot(theta))
    val = y.dot(np.log(alpha)) + (1.0 - y).dot(np.log(1.0 - alpha))
    return val


def train(X, y):
    n, m = X.shape
    theta = np.zeros(m)
    theta, _, _ = minimize(
        likelihood,
        theta,
        args=(X, y),
        method='L-BFGS-B',
    )


if __name__ == '__main__':
    train(X, labels[:, 0])
