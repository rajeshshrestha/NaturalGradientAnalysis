# %%
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from time import time
from tqdm import tqdm
from scipy.linalg import khatri_rao
import pdb
import pandas as pd

seed = np.random.randint(0, 10000)
np.random.seed(seed)

DATASET = 'weather'
TEST_PROPORTION = 0.2
TASK = 'regression'

running_loss_window = 3

assert DATASET in ['separated_svm', 'overlapped_svm',
                   'weather'], f"Unknown value of Dataset: {DATASET} set"

# %%
if DATASET == 'separated_svm':
    '''Load separable data'''
    train_separable = sio.loadmat(
        "data/svm data/separable_case/train_separable.mat")
    test_separable = sio.loadmat(
        "data/svm data/separable_case/test_separable.mat")
    train_data, test_data = train_separable, test_separable
elif DATASET == 'overlapped_svm':
    '''Load overlap data'''
    train_overlap = sio.loadmat("data/svm data/overlap_case/train_overlap.mat")
    test_overlap = sio.loadmat("data/svm data/overlap_case/test_overlap.mat")
    train_data, test_data = train_overlap, test_overlap
elif DATASET == 'weather':
    '''Load weather data'''
    weather_data = pd.read_csv("data/weather/weatherHistory.csv")[["Apparent Temperature (C)", "Humidity",
                                                                   "Wind Speed (km/h)", "Wind Bearing (degrees)", "Visibility (km)", "Pressure (millibars)", "Temperature (C)"]]
    weather_data = (weather_data - weather_data.mean()) / weather_data.std()


# %%
if DATASET in ['separated_svm', 'overlapped_svm']:
    A, B = train_data['A'], train_data['B']
    train_labels = np.concatenate(
        [np.ones(A.shape[1]), np.zeros(B.shape[1])], axis=0)
    test_X, test_labels = test_data['X_test'], test_data['true_labels'].flatten(
    )
    test_labels = np.where(test_labels == 1, 1, 0)

    train_data = np.block([[A, B], [train_labels]])
    test_data = np.block([[test_X], [test_labels]])

    d, train_num, test_num = train_data.shape[0] - \
        1, train_data.shape[1], test_data.shape[1]

elif DATASET == 'weather':
    data_indices = set(weather_data.index)
    test_indices = set(np.random.choice(list(data_indices), size=int(
        len(data_indices)*TEST_PROPORTION), replace=False))
    train_indices = data_indices - test_indices
    train_data, test_data = weather_data.loc[list(train_indices)].to_numpy(
    ).T, weather_data.loc[list(test_indices)].to_numpy().T
    d, train_num, test_num = train_data.shape[0] - \
        1, train_data.shape[1], test_data.shape[1]


# %% [markdown]
# ## Model

# %%
class Linear:
    def __init__(self, input_size, output_size, activation='relu', lambda_reg=0) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.lambda_reg = lambda_reg

        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1, output_size)

        assert activation in [
            None, "relu", "sigmoid"], f"Unknown activation Type passed: {activation}"
        self.activation = activation
        self.activation_fn = lambda x: np.maximum(x, 0) if self.activation == 'relu' else 1/(
            1+np.exp(-np.clip(x, -30, 30))) if self.activation == 'sigmoid' else x

        self.m = 0
        self.X = 0
        self.Y = None
        self.out = None
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        self.m = X.shape[1]
        self.X = X
        self.Y = self.W.T@self.X+self.b.T
        self.out = self.activation_fn(self.Y)
        return self.out

    def get_agumented_X(self):
        return np.concatenate([self.X, np.ones((1, self.m))], axis=0)

    def get_agumented_params(self):
        return np.concatenate([self.W, self.b], axis=0)

    def get_agumented_grads(self):
        return np.concatenate([self.grad_W, self.grad_b], axis=0)

    def backward(self, back_grad):
        # print(f"back grad shape: {back_grad.shape}")
        if self.activation is None:
            self.G = back_grad
        elif self.activation == 'relu':
            self.G = np.where(self.out >= 0, 1, 0)*back_grad
        else:
            self.G = self.activation_fn(
                self.Y)*(1-self.activation_fn(self.Y))*back_grad

        ''' Clip G'''
        # self.G = np.clip(self.G, -10, 10)

        # pdb.set_trace()
        grad = (1/self.m)*self.get_agumented_X()@(self.G.T)
        grad_W_val = grad[:-1, :]
        grad_b_val = grad[-1:, :]
        if self.grad_W is None or self.grad_b is None:
            self.grad_W = grad_W_val + self.lambda_reg * 2 * self.W
            self.grad_b = grad_b_val
        else:
            self.grad_W += grad_W_val
            self.grad_b += grad_b_val

        # print(self.W.shape, self.G.shape)
        return (1/self.m)*self.W@back_grad

    def clear_gradient(self):
        self.grad_W = None
        self.grad_b = None

    def first_order_update(self, lr):
        # print(lr)
        # print(self.W.shape, self.grad_W.shape)
        # print(self.b.shape, self.grad_b.shape)
        self.W -= lr*self.grad_W
        self.b -= lr*self.grad_b
        self.clear_gradient()

    def get_jacobian(self):
        return khatri_rao(self.get_agumented_X(), self.G).T

    def get_all_params(self):
        return np.concatenate([self.W.flatten(), self.b.flatten()], axis=0)

    def get_all_grads(self):
        return np.concatenate([self.grad_W.flatten(), self.grad_b.flatten()], axis=0)

    def get_params_num(self):
        return len(self.W.flatten())+len(self.b.flatten())

    def set_params(self, params):
        # print(np.linalg.norm(params-self.get_all_params()))
        W_param_num = len(self.W.flatten())
        self.W = params[:W_param_num].reshape(self.W.shape)
        self.b = params[W_param_num:].reshape(self.b.shape)

    def kfac_update(self, lr, alpha, min_FIM_det, max_FIM_det):
        J_k = self.get_jacobian()
        F_k = 1/self.m * J_k.T@J_k
        F_k = F_k + alpha*np.eye(F_k.shape[0])
        FIM = np.linalg.inv(F_k)

        # '''Normalize FIM if the determinant is too high or too low'''
        # det = np.linalg.det(FIM)
        # if det > max_FIM_det:
        #     FIM *= (max_FIM_det/det)**(1/FIM.shape[0])
        # elif det < min_FIM_det:
        #     FIM *= (min_FIM_det/det)**(1/FIM.shape[0])

        params = self.get_all_params()
        grads = self.get_all_grads()
        params -= lr*FIM@grads

        self.set_params(params)
        self.clear_gradient()

    def tengrad_update(self, lr, alpha, min_A_det, max_A_det):
        J_JT = (self.get_agumented_X().T@self.get_agumented_X()) * \
            (self.G.T@self.G)
        A = np.linalg.inv(J_JT/self.m+alpha*np.eye(J_JT.shape[0]))

        # '''Normalize A if the determinant is too high'''
        # det = np.linalg.det(A)
        # if det > max_A_det:
        #     A *= (max_A_det/det)**(1/A.shape[0])
        # elif det < min_A_det:
        #     A *= (min_A_det/det)**(1/A.shape[0])

        grads = self.get_all_grads()
        params = self.get_all_params()

        b = ((self.get_agumented_grads().T@self.get_agumented_X())
             * self.G).T@np.ones((self.G.shape[0], 1))

        v = A@b

        params -= lr/alpha*(grads-1/self.m*(self.get_agumented_X() @
                            ((v@np.ones((1, self.G.shape[0])))*self.G.T)).reshape(grads.shape))
        self.set_params(params)
        self.clear_gradient()

    # def update_using_approx_FIM(self, lr, flattened_J):
    #     J_W = flattened_J[:self.input_size*self.output_size].reshape(self.W.shape)
    #     J_b = flattened_J[self.input_size*self.output_size:].reshape(self.b.shape)


# %%
class Model:
    def __init__(self, node_nums, activations, lambda_reg=0):
        self.layers = []
        for i in range(len(node_nums)-1):
            self.layers.append(
                Linear(node_nums[i], node_nums[i+1], activations[i], lambda_reg=lambda_reg))

    def forward(self, X):
        self.m = X.shape[1]
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, back_grad):
        for layer in self.layers[::-1]:
            back_grad = layer.backward(back_grad)

    def clear_gradient(self):
        for layer in self.layers:
            layer.clear_gradient()

    def update(self, lr, degree="first_order", alpha=1e-3, min_det=1e-3, max_det=100):
        if degree == 'first_order':
            for layer in self.layers:
                layer.first_order_update(lr)
        elif degree == 'natural_gradient':
            J = np.concatenate([layer.get_jacobian()
                               for layer in self.layers], axis=1)
            F = (1/self.m)*J.T@J
            F_ = F+alpha*np.eye(F.shape[0])
            FIM = np.linalg.inv(F_)

            # '''Normalize FIM if the determinant is too high or too low'''
            # det = np.linalg.det(FIM)
            # if det > max_det:
            #     FIM *= (max_det/det)**(1/FIM.shape[0])
            # elif det < min_det:
            #     FIM *= (min_det/det)**(1/FIM.shape[0])

            all_params = np.concatenate(
                [layer.get_all_params() for layer in self.layers], axis=0)
            all_grads = np.concatenate(
                [layer.get_all_grads() for layer in self.layers], axis=0)
            param_nums = [layer.get_params_num() for layer in self.layers]
            # FIM = np.eye(all_grads.shape[0])
            all_params -= lr * FIM@all_grads

            '''Set new param numbers in layers'''
            initial_param_idx = 0
            for i, layer in enumerate(self.layers):
                layer.set_params(
                    all_params[initial_param_idx: initial_param_idx+param_nums[i]])
                initial_param_idx += param_nums[i]
                layer.clear_gradient()
            # return np.linalg.det(FIM)
        elif degree == 'kfac':
            for layer in self.layers:
                layer.kfac_update(
                    lr, alpha=alpha, min_FIM_det=min_det, max_FIM_det=max_det)

        elif degree == 'tengrad':
            for layer in self.layers:
                layer.tengrad_update(
                    lr, alpha=alpha, min_A_det=min_det, max_A_det=max_det)


# %%
class BinaryCrossEntropy:
    def __init__(self) -> None:
        self.out = None
        self.back = None

    def compute(self, y_pred, y_labels):
        self.out = np.mean(np.where(y_labels == 0, -np.log2(np.clip(1 -
                           y_pred, 1e-6, 1-1e-6)), -np.log2(np.clip(y_pred, 1e-6, 1-1e-6))))
        self.back = np.where(
            y_labels == 0, 1/(1-y_pred + 1e-6), -1/(y_pred+1e-6))
        return self.out

    def backward(self):
        return self.back


class MSE:
    def __init__(self) -> None:
        self.out = None
        self.back = None

    def compute(self, y_pred, true_y):
        m = y_pred.shape[1]
        self.out = np.mean((y_pred-true_y)**2)
        self.back = 1/m * 2 * (y_pred-true_y)
        return self.out

    def backward(self):
        return self.back

# %%


def train(update_type="first_order", task='classification', lr=1e-4, epochs=10000, mini_batch_size=32, lambda_reg=0):
    model = Model([d, 128, 64, 1], ['relu', 'relu', None],
                  lambda_reg=lambda_reg)

    if task == 'classification':
        loss = BinaryCrossEntropy()
        train_info = {'iter_wise': {'iterations': [], 'loss': [], 'running_loss': []}, 'epoch_wise': {
            'epochs': [], 'loss': [], 'val_loss': [], 'val_accuracy': []}}
        validation_accuracy = 0

    elif task == 'regression':
        loss = MSE()
        train_info = {'iter_wise': {'iterations': [], 'loss': [], 'running_loss': []},
                      'epoch_wise': {'epochs': [], 'loss': [], 'val_loss': []}}

    pbar = tqdm(list(range(1, epochs+1)))
    iteration = 0

    train_features, train_labels = train_data[:d], train_data[d:]
    test_features, test_labels = test_data[:d], test_data[d:]

    average_loss = float("inf")
    val_loss = float("inf")

    '''Log Initial epoch wise train info'''
    initial_train_pred, initial_test_pred = model.forward(
        train_features), model.forward(test_features)
    train_info['epoch_wise']['epochs'].append(0)
    train_info['epoch_wise']['loss'].append(
        loss.compute(initial_train_pred, train_labels))
    train_info['epoch_wise']['val_loss'].append(
        loss.compute(initial_test_pred, test_labels))

    if task == 'classification':
        train_info['epoch_wise']['val_accuracy'].append(
            np.mean(np.where(initial_test_pred > 0.5, 1, 0) == test_labels))

    lr_ = lr
    for epoch in pbar:
        data_idxs = set(range(train_num))
        epoch_loss_lst = []
        while data_idxs:
            iteration += 1
            lr_ = lr/(1+0.1*iteration//10)
            sample_idx = np.random.choice(list(data_idxs), size=min(
                mini_batch_size, len(data_idxs)), replace=False).tolist()
            data_idxs -= set(sample_idx)
            data, labels = train_features[:,
                                          sample_idx], train_labels[:, sample_idx]
            out = model.forward(data)
            loss_val = loss.compute(out, labels)
            epoch_loss_lst.append(loss_val)
            loss_grad = loss.backward()
            model.backward(loss_grad)
            res = model.update(lr_, update_type, alpha=1e-6, max_det=1000)
            # if res is not None:
            #     print(res)

            if np.isnan(loss_val) or np.log(loss_val) > 20:
                break

            '''Log iter wise train info'''
            train_info['iter_wise']['iterations'].append(iteration)
            train_info['iter_wise']['loss'].append(loss_val)
            train_info['iter_wise']['running_loss'].append(
                np.mean(train_info['iter_wise']['loss'][-running_loss_window:]))

            if task == "classification":
                pbar.set_description(" Update Type: %s LR: %f Iteration: %d Epoch: %d Loss: %f Epoch Loss: %f Val Loss: %f Validation Accuracy %f" % (update_type, lr_,
                                                                                                                                     iteration, epoch, round(loss_val, 6), round(average_loss, 6), round(val_loss, 6), round(validation_accuracy, 6)))

            elif task == "regression":
                pbar.set_description("Update Type: %s LR: %f Iteration: %d Epoch: %d Loss: %f Epoch Loss: %f Val Loss: %f" % (update_type, lr_,
                                                                                                              iteration, epoch, round(loss_val, 6), round(average_loss, 6), round(val_loss, 6)))
        if np.isnan(loss_val) or np.log(loss_val) > 20:
            break
        # pdb.set_trace()
        '''Lr decay'''
        # lr_ /= 10
        average_loss = np.mean(epoch_loss_lst)
        if task == 'classification':
            pred_prob = model.forward(test_features)
            preds = np.where(pred_prob > 0.5, 1, 0)
            val_loss = loss.compute(pred_prob, test_labels)
            validation_accuracy = np.mean(test_labels == preds)
            train_info['epoch_wise']['val_accuracy'].append(
                validation_accuracy)
        else:
            pred = model.forward(test_features)
            val_loss = loss.compute(pred, test_labels)

        '''Log epoch wise train info'''
        train_info['epoch_wise']['epochs'].append(epoch)
        train_info['epoch_wise']['loss'].append(average_loss)
        train_info['epoch_wise']['val_loss'].append(val_loss)

    return train_info


# plt.ion()
plt.figure("training", figsize=(20, 40))

if TASK == "classification":
    fig_num = 4
else:
    fig_num = 3


# for lr in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-6, 3e-5, 3e-4, 3e-3, 3e-2, 3e-1][::-1]:
for lr in [ 1e-3, 1e-2, 1e-1][::-1]:

    # %%
    tengrad_grad_descent_train_info = train(
        "tengrad", task=TASK, lr=lr, epochs=5, mini_batch_size=128, lambda_reg=1e-6)

    # %%
    grad_descent_train_info = train(
        "first_order", task=TASK, lr=lr, epochs=5, mini_batch_size=128, lambda_reg=1e-6)

    # # %%
    # kfac_grad_descent_train_info = train(
    #     "kfac", task=TASK, lr=1e-3, epochs=3, mini_batch_size=64)

    # # %%
    # natural_grad_descent_train_info = train(
    #     "natural_gradient", task=TASK, lr=1e-5, epochs=3, mini_batch_size=64)

    # %%
    # [(key, val[0]) for key, val in grad_descent_train_info['epoch_wise'].items()], [(key, val[0]) for key, val in natural_grad_descent_train_info['epoch_wise'].items()]

    # %% [markdown]
    # ## Plot train metrics

    # %%
    plt.subplot(fig_num, 1, 1)
    plt.plot(grad_descent_train_info['iter_wise']['iterations'], np.log(
        grad_descent_train_info['iter_wise']['running_loss']), '-', label=f'Without natural gradient lr={lr}')
    # plt.plot(natural_grad_descent_train_info['iter_wise']['iterations'], np.log(
    #     natural_grad_descent_train_info['iter_wise']['loss']), label='With natural gradient')
    # plt.plot(kfac_grad_descent_train_info['iter_wise']['iterations'], np.log(
    #     kfac_grad_descent_train_info['iter_wise']['loss']), label='KFAC')
    plt.plot(tengrad_grad_descent_train_info['iter_wise']['iterations'], np.log(
        tengrad_grad_descent_train_info['iter_wise']['running_loss']), '--', label=f'TENGRAD lr={lr}')
    plt.xlabel("Iterations")
    plt.ylabel("Log of Batch Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.subplot(fig_num, 1, 2)
    plt.plot(grad_descent_train_info['epoch_wise']['epochs'], np.log(
        grad_descent_train_info['epoch_wise']['loss']), '-', label=f'Without natural gradient lr={lr}')
    # plt.plot(natural_grad_descent_train_info['epoch_wise']['epochs'], np.log(
    #     natural_grad_descent_train_info['epoch_wise']['loss']), label='With natural gradient')
    # plt.plot(kfac_grad_descent_train_info['epoch_wise']['epochs'], np.log(
    #     kfac_grad_descent_train_info['epoch_wise']['loss']), label='KFAC')
    plt.plot(tengrad_grad_descent_train_info['epoch_wise']['epochs'], np.log(
        tengrad_grad_descent_train_info['epoch_wise']['loss']), '--', label=f'TENGRAD lr={lr}')
    plt.xlabel("Epochs")
    plt.ylabel("Log of Average Train Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.subplot(fig_num, 1, 3)
    plt.plot(grad_descent_train_info['epoch_wise']['epochs'], np.log(
        grad_descent_train_info['epoch_wise']['val_loss']), '-', label=f'Without natural gradient lr={lr}')
    # plt.plot(natural_grad_descent_train_info['epoch_wise']['epochs'], np.log(
    #     natural_grad_descent_train_info['epoch_wise']['val_loss']), label='With natural gradient')
    # plt.plot(kfac_grad_descent_train_info['epoch_wise']['epochs'], np.log(
    #     kfac_grad_descent_train_info['epoch_wise']['val_loss']), label='KFAC')
    plt.plot(tengrad_grad_descent_train_info['epoch_wise']['epochs'], np.log(
        tengrad_grad_descent_train_info['epoch_wise']['val_loss']), '--', label=f'TENGRAD lr={lr}')
    plt.xlabel("Epochs")
    plt.ylabel("Log of Validation Loss")
    plt.legend(loc='upper right')
    plt.tight_layout()

    if TASK == "classification":
        plt.subplot(fig_num, 1, 4)
        plt.plot(grad_descent_train_info['epoch_wise']['epochs'],
                 grad_descent_train_info['epoch_wise']['val_accuracy'], '-', label=f'Without natural gradient lr={lr}')
        # plt.plot(natural_grad_descent_train_info['epoch_wise']['epochs'],
        #          natural_grad_descent_train_info['epoch_wise']['val_accuracy'], label='With natural gradient')
        # plt.plot(kfac_grad_descent_train_info['epoch_wise']['epochs'],
        #          kfac_grad_descent_train_info['epoch_wise']['val_accuracy'], label='KFAC')
        plt.plot(tengrad_grad_descent_train_info['epoch_wise']['epochs'],
                 tengrad_grad_descent_train_info['epoch_wise']['val_accuracy'], '--', label=f'TENGRAD lr={lr}')
        plt.xlabel("Epochs")
        plt.ylabel("Validation Accuracy")
        plt.legend(loc='upper right')
        plt.tight_layout()
plt.show()

# %%
