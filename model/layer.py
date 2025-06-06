# Base Class Layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.time_usage = None

    # compute the output of a layer for a given input
    def forward_propagation(self, input):
        raise NotImplementedError

    # conpute dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
