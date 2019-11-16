class Module:
    def __init__(self):
        self.ctx = []
        self.grad = None
        self.parameters = None
        self.traininig = True

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    def save_for_backward(self, value):
        self.ctx.append(value)

    def train(self):
        self.traininig = True

    def eval(self):
        self.traininig = False
