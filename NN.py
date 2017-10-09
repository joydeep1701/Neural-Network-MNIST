import numpy as np

def bit_vector_to_label(bv):
    label = []
    j = 0
    for i in bv:
        if i != 0:
            label.append(str(j))
        j += 1
    return ",".join(label)

# #### Activation functions
# During training we need to compute the activation fuction values for each layer in addition to their derivatives.

def f_sigmoid(X, deriv=False):
    if not deriv:
        return 1 / (1 + np.exp(-X))
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))

def f_softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z

class Layer:
    def __init__(self, size, minibatch_size, is_input=False, is_output=False, activation=f_sigmoid):
        self.is_input = is_input
        self.is_output = is_output

        # Z matrix stores the output values for the network
        self.Z = np.zeros((minibatch_size, size[0]))

        # the activation fuction is an externally defined function  with a derivative
        # that is stored here
        self.activation = activation

        # W is the outgoing weight matrix for this layer
        self.W = None

        # S is the matrix that holds the inputs for this layer
        self.S = None

        # D is the matrix holding the deltas for this layer
        self.D  = None

        # Fp is a matrix containin the derivatives of the activation function
        self.fp = None

        if not is_input:
            self.S = np.zeros((minibatch_size, size[0]))
            self.D = np.zeros((minibatch_size, size[0]))

        if not is_output:
            self.W = np.random.normal(size=size, scale=1E-4)

        if not is_input and not is_output:
            self.Fp = np.zeros((size[0], minibatch_size))

    def forward_propagate(self):
        if self.is_input:
            return self.Z.dot(self.W)

        self.Z = self.activation(self.S)
        if self.is_output:
            return self.Z
        else:
            # For hidden layers, we add the bias values here
            self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
            self.Fp = self.activation(self.S, deriv=True).T
            return self.Z.dot(self.W)
