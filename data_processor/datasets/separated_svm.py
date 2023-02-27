import scipy.io as sio
import numpy as np


def get_data():
    '''Load separable data'''
    train_separable = sio.loadmat(
        "data/svm data/separable_case/train_separable.mat")
    test_separable = sio.loadmat(
        "data/svm data/separable_case/test_separable.mat")
    train_data, test_data = train_separable, test_separable

    A, B = train_data['A'], train_data['B']
    train_labels = np.concatenate(
        [np.ones(A.shape[1]), np.zeros(B.shape[1])], axis=0)
    test_X, test_labels = test_data['X_test'], test_data['true_labels'].flatten(
    )
    test_labels = np.where(test_labels == 1, 1, 0)

    train_data = np.block([[A, B], [train_labels]])
    test_data = np.block([[test_X], [test_labels]])

    return train_data, test_data
