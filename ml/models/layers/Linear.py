import numpy as np
from scipy.linalg import khatri_rao


class Linear:
    '''Fully connected Linear layer'''
    def __init__(self, input_size, output_size, activation='relu', lambda_reg=0, is_ngd=False) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.lambda_reg = lambda_reg
        self.is_ngd = is_ngd
       
        '''Initialize parameters of the layer'''
        self.W = np.random.randn(input_size, output_size)
        self.b = np.random.randn(1, output_size)


        assert activation in [
            'None', "relu", "sigmoid"], f"Unknown activation Type passed: {activation}"
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
        '''Forward pass in layer'''
        self.m = X.shape[1]
        self.X = X
        self.Y = self.W.T@self.X+self.b.T
        self.out = self.activation_fn(self.Y)
        return self.out

    def get_agumented_X(self):
        '''Agumented X with 1 for bias'''
        return np.concatenate([self.X, np.ones((1, self.m))], axis=0)

    def get_agumented_params(self):
        '''Get W agumented with b'''
        return np.concatenate([self.W, self.b], axis=0)

    def get_agumented_grads(self):
        '''Get grad_W agumented with b'''
        return np.concatenate([self.grad_W, self.grad_b], axis=0)

    def backward(self, back_grad):
        '''Backward pass in a layer for gradient computation'''
        if self.activation == 'None':
            G = back_grad
        elif self.activation == 'relu':
            G = np.where(self.out >= 0, 1, 0)*back_grad
        else:
            G = self.activation_fn(
                self.Y)*(1-self.activation_fn(self.Y))*back_grad

        grad = (1/self.m)*self.get_agumented_X()@(G.T)

        if self.is_ngd:
            self.G = G
        else:
            del G

        grad_W_val = grad[:-1, :]
        grad_b_val = grad[-1:, :]
        if self.grad_W is None or self.grad_b is None:
            self.grad_W = grad_W_val + self.lambda_reg * 2 * self.W
            self.grad_b = grad_b_val
        else:
            self.grad_W += grad_W_val
            self.grad_b += grad_b_val
        return (1/self.m)*self.W@back_grad

    def clear_gradient(self):
        '''Clear accumulated gradient of the parameters of the layer'''
        self.grad_W = None
        self.grad_b = None

    def first_order_update(self, lr):
        '''Update based on SGD'''
        self.W -= lr*self.grad_W
        self.b -= lr*self.grad_b
        self.clear_gradient()

    def get_jacobian(self):
        '''Compute Jacobian'''
        return khatri_rao(self.get_agumented_X(), self.G).T

    def get_all_params(self):
        return np.concatenate([self.W.flatten(), self.b.flatten()], axis=0)

    def get_all_grads(self):
        return np.concatenate([self.grad_W.flatten(), self.grad_b.flatten()], axis=0)

    def get_params_num(self):
        return len(self.W.flatten())+len(self.b.flatten())

    def set_params(self, params):
        W_param_num = len(self.W.flatten())
        self.W = params[:W_param_num].reshape(self.W.shape)
        self.b = params[W_param_num:].reshape(self.b.shape)

    def kfac_update(self, lr, alpha):
        '''Block-wise NGD Update for the layer parameters'''
        J_k = self.get_jacobian()
        F_k = 1/self.m * J_k.T@J_k
        F_k = F_k + alpha*np.eye(F_k.shape[0])
        FIM = np.linalg.inv(F_k)

        params = self.get_all_params()
        grads = self.get_all_grads()
        params -= lr*FIM@grads

        self.set_params(params)
        self.clear_gradient()

    def tengrad_update(self, lr, alpha):
        '''TENGraD update for the layer parameters'''
        J_JT = (self.get_agumented_X().T@self.get_agumented_X()) * \
            (self.G.T@self.G)
        A = np.linalg.inv(J_JT/self.m+alpha*np.eye(J_JT.shape[0]))

        grads = self.get_all_grads()
        params = self.get_all_params()

        b = ((self.get_agumented_grads().T@self.get_agumented_X())
             * self.G).T@np.ones((self.G.shape[0], 1))

        v = A@b

        params -= lr/alpha*(grads-1/self.m*(self.get_agumented_X() @
                            ((v@np.ones((1, self.G.shape[0])))*self.G.T)).reshape(grads.shape))
        self.set_params(params)
        self.clear_gradient()
