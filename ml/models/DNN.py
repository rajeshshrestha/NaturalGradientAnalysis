import numpy as np
from .layers.Linear import Linear


class DNN:
    '''Fully connected DNN'''
    def __init__(self, node_nums, activations, lambda_reg=0, is_ngd=True):
        self.layers = []
        for i in range(len(node_nums)-1):
            self.layers.append(
                Linear(node_nums[i], node_nums[i+1], activations[i], lambda_reg=lambda_reg, is_ngd=is_ngd))

    def forward(self, X):
        '''Forward pass in the model'''
        self.m = X.shape[1]
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, back_grad):
        '''Back pass in the model for gradient propagation'''
        for layer in self.layers[::-1]:
            back_grad = layer.backward(back_grad)

    def clear_gradient(self):
        '''Clear accumulated gradients for all layers in the model'''
        for layer in self.layers:
            layer.clear_gradient()

    def get_param_num(self):
        return sum([len(layer.get_all_params()) for layer in self.layers])


    def update(self, lr, degree="first_order", alpha=1e-3):
        if degree == 'first_order':
            '''SGD update'''
            for layer in self.layers:
                layer.first_order_update(lr)
        elif degree == 'natural_gradient':
            '''Exact NGD update without blockwise FIM assumption'''
            J = np.concatenate([layer.get_jacobian()
                               for layer in self.layers], axis=1)
            F = (1/self.m)*J.T@J
            F_ = F+alpha*np.eye(F.shape[0])
            FIM = np.linalg.inv(F_)

            all_params = np.concatenate(
                [layer.get_all_params() for layer in self.layers], axis=0)
            all_grads = np.concatenate(
                [layer.get_all_grads() for layer in self.layers], axis=0)
            param_nums = [layer.get_params_num() for layer in self.layers]
            all_params -= lr * FIM@all_grads

            '''Set new param numbers in layers'''
            initial_param_idx = 0
            for i, layer in enumerate(self.layers):
                layer.set_params(
                    all_params[initial_param_idx: initial_param_idx+param_nums[i]])
                initial_param_idx += param_nums[i]
                layer.clear_gradient()

        elif degree == 'kfac':
            '''Update model parameters using Blockwise NGD method'''
            for layer in self.layers:
                layer.kfac_update(
                    lr, alpha=alpha)

        elif degree == 'tengrad':
            '''Update model parameters using TENGraD method'''
            for layer in self.layers:
                layer.tengrad_update(
                    lr, alpha=alpha)
