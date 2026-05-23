import numpy as np

def accuracy(Y_true, Y_pred):
    return np.mean(Y_true == Y_pred)

def precision(Y_true, Y_pred):
    tp = np.sum((Y_true == 1) & (Y_pred == 1))
    fp = np.sum((Y_true == 0) & (Y_pred == 1))
    return tp / (tp + fp + 1e-15)

def recall(Y_true, Y_pred):
    tp = np.sum((Y_true == 1) & (Y_pred == 1))
    fn = np.sum((Y_true == 1) & (Y_pred == 0))
    return tp / (tp + fn + 1e-15)
