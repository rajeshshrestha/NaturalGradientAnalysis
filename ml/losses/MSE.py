import numpy as np


class MSE:
    '''Mean Square Error Loss'''
    def __init__(self) -> None:
        self.back = None

    def compute(self, y_pred, true_y):
        m = y_pred.shape[1]
        out = np.mean((y_pred-true_y)**2)
        self.back = 1/m * 2 * (y_pred-true_y)
        return out

    def backward(self):
        return self.back
