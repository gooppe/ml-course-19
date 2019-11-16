import numpy as np

from .module import Module


class Sum(Module):
    def forward(self, input, *args):
        self.save_for_backward(input.shape)
        return np.sum(input)

    def backward(self, grad_output):
        (shape,) = self.ctx
        return np.ones(shape)


class ReLU(Module):
    def forward(self, input):
        self.save_for_backward(input)
        return np.maximum(input, 0)

    def backward(self, grad_output):
        (input,) = self.ctx
        return grad_output * (input > 0).astype(float)


class Dropout(Module):
    def __init__(self, prob=0.3):
        super().__init__()
        self.keep_prob = 1 - prob
        self.inv_prob = 1 / (prob)

    def forward(self, input):
        if self.traininig:
            mask = np.random.binomial(1, self.keep_prob, input.shape)
            self.save_for_backward(mask)
            return input * mask * self.inv_prob
        else:
            return input

    def backward(self, grad_output):
        (mask,) = self.ctx
        return grad_output * mask * self.inv_prob


class Sigmoid(Module):
    def forward(self, input):
        sigmoid = 1 / (1 + np.exp(-input))
        self.save_for_backward(sigmoid)
        return sigmoid

    def backward(self, grad_output):
        (sigmoid,) = self.ctx
        return grad_output * sigmoid * (1 - sigmoid)


class Softmax(Module):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, input):
        exp = np.exp(input - np.max(input, axis=-1, keepdims=True))
        softmax = exp / np.sum(exp, axis=self.axis, keepdims=True)
        self.save_for_backward(softmax)
        return softmax

    def backward(self, grad_output):
        (softmax,) = self.ctx

        return (
            grad_output - np.reshape(np.sum(grad_output * softmax, 1), [-1, 1])
        ) * softmax


class Log(Module):
    def forward(self, input):
        self.save_for_backward(input)
        return np.log(input)

    def backward(self, grad_output):
        (input,) = self.ctx
        return grad_output / input


class MSELoss(Module):
    def forward(self, input, target):
        diff = input - target
        self.save_for_backward(diff)
        return np.mean(diff ** 2, keepdims=True), input

    def backward(self, grad_output):
        (diff,) = self.ctx
        return 2 * diff
