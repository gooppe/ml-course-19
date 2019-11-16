import numpy as np

from .module import Module


class Linear(Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()

        self.inpu_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        if bias:
            input_dim += 1

        k = np.sqrt(1 / input_dim)
        self.parameters = np.random.uniform(-k, k, (input_dim, output_dim))

    def forward(self, input):
        batch, _ = input.shape
        if self.bias:
            bias = np.ones((batch, 1))
            input = np.concatenate((input, bias), axis=-1)
        self.save_for_backward(input)

        return input @ self.parameters

    def backward(self, grad_output):
        grad, = self.ctx
        self.grad = grad.T @ grad_output
        assert self.grad.shape == self.parameters.shape
        parameters = self.parameters[:-1, :] if self.bias else self.parameters
        return grad_output @ parameters.T
