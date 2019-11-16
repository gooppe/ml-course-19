class SGD:
    def __init__(self, graph, lr=1e-3):
        self.lr = lr
        self.graph = graph

    def step(self):
        for layer in self.graph:
            layer.parameters -= layer.grad * self.lr
