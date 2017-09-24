import math
import numpy as np

# One-hot encoding of nominal values
def one_hot_encode(y):
    y.astype("int")
    n_col = np.amax(y) + 1
    binarized = np.zeros((len(y), n_col))
    for i in range(len(y)):
        binarized[i, y[i]] = 1

    return binarized
# Split the data into train and test sets

def shuffle_data(X, y, seed=None):
    if seed:
        np.random.seed(seed)
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    return X, y

def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    x_train, x_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return x_train, x_test, y_train, y_test
