import numpy as np


class BCE:
    '''Binary Cross Entropy Loss'''
    def __init__(self) -> None:
        self.back = None

    def compute(self, y_pred, y_labels):
        out = np.mean(np.where(y_labels == 0, -np.log2(np.clip(1 -
                                                               y_pred, 1e-6, 1-1e-6)), -np.log2(np.clip(y_pred, 1e-6, 1-1e-6))))
        self.back = np.where(
            y_labels == 0, 1/(1-y_pred + 1e-6), -1/(y_pred+1e-6))
        return out

    def backward(self):
        return self.back
