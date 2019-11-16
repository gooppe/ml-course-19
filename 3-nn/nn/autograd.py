class ComputationGraph:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input, target):
        *layers, last_layer = self.layers
        for layer in layers:
            input = layer(input)
        return last_layer(input, target)

    def backward(self):
        grad = None
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def zero_grad(self):
        for layer in self.layers:
            layer.ctx = []
            layer.grad = None

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()

    def __call__(self, input, target=None):
        return self.forward(input, target)

    def __len__(self):
        return sum(layer.parameters is not None for layer in self.layers)

    def __iter__(self):
        for layer in self.layers:
            if layer.parameters is not None:
                yield layer
