import numpy as np
import utils

class Propagator:
    def __init__(self):
        pass

    @profile
    def forward(self, model):
        """
        Forward progropation of the model.
        """
        m = model

        # number of layers
        L = len(m.parameters) // 2

        # initialize activations
        activations = {}
        activations["A0"] = m.inputs

        # forward propagation with ReLu activation
        for l in range(1, L):
            activations['Z'+str(l)] = np.dot(m.parameters['W'+str(l)], activations['A'+str(l-1)]) + m.parameters['b'+str(l)]
            activations['A'+str(l)] = self.relu(activations['Z'+str(l)])

        # last layer with Softmax activation
        activations['Z'+str(L)] = np.dot(m.parameters['W'+str(L)], activations['A'+str(L-1)]) + m.parameters['b'+str(L)]
        activations['A'+str(L)] = self.softmax(activations['Z'+str(L)])

        model.activations = activations
        
        return model

    def backward(self, model):
        """
        Backward propagation of the model.
        """
        m = model
        r = m.encoded_outputs.shape[1]

        # number of layers
        L = len(m.parameters) // 2

        # initialize gradients
        derivatives = {}
        gradients = {}

        # last layer
        derivatives['dZ'+str(L)] = (m.activations['A'+str(L)] - m.encoded_outputs)
        gradients['dW'+str(L)] = (1./r) * np.dot(derivatives['dZ'+str(L)], m.activations['A'+str(L-1)].T)
        gradients['db'+str(L)] = (1./r) * np.sum(derivatives['dZ'+str(L)])

        # other layers
        for l in reversed(range(1,L)):
            derivatives['dZ'+str(l)] = np.dot(m.parameters['W'+str(l+1)].T, derivatives['dZ'+str(l+1)]) * self.deriv_relu(m.activations['Z'+str(l)])
            gradients['dW'+str(l)] = (1./r) * np.dot(derivatives['dZ'+str(l)], m.activations['A'+str(l-1)].T)
            gradients['db'+str(l)] = (1./r) * np.sum(derivatives['dZ'+str(l)], axis=1, keepdims=True)

        model.gradients = gradients

        return model

    def relu(self, x):
        """
        ReLu activation function.
        """
        return np.maximum(0,x)

    def deriv_relu(self, x):
        """
        Derivative of ReLu activation function.
        """
        return 1. * (x > 0)

    def softmax(self, x):
        """
        Softmax activation function.
        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def deriv_softmax(self, x):
        """
        Derivative of Softmax activation function.
        """
        return self.softmax(x) * (1 - self.softmax(x))
