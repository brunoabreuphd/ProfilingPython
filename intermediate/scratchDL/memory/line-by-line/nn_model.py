import numpy as np
import utils

class NNModel:
    def __init__(self, layers_size, data_loader, learning_rate):
        self.layers_size = layers_size
        self.learning_rate = learning_rate
        
        self.parameters = None
        self.activations = None
        self.derivatives = None
        self.gradients = None
        
        self.data_loader = data_loader
        self.inputs = None
        self.outputs = None
        self.encoded_outputs = None

        self.predictor = None
        self.metrics = None

    def initialize_random_params(self):
        """
        Initialize parameters for the model.
        """
        parameters = {}

        ls = self.layers_size

        for layer in range(1, len(ls)):
            parameters['W' + str(layer)] = np.random.randn(ls[layer], ls[layer-1]) * np.sqrt(1. / ls[layer])
            parameters['b' + str(layer)] = np.random.randn(ls[layer], 1) * np.sqrt(1. / ls[layer])

        self.parameters = parameters

    def load_train_data(self):
        """
        Load data from the data loader.
        """
        inputs, outputs = self.data_loader.load_train_data()
        self.inputs = inputs
        self.outputs = outputs

        if self.data_loader.one_hot_encode:
            self.encoded_outputs = utils.one_hot_encode(self.outputs)

    def check_readiness(self):
        """
        Check if the model is ready to be trained.
        """
        if self.inputs is None or self.outputs is None:
            raise Exception("The model is not ready to be trained. Please load data first.")
        elif self.predictor is None:
            raise Exception("The model is not ready to be trained. Please set a predictor first.")
        elif self.metrics is None:
            raise Exception("The model is not ready to be trained. Please set metrics first.")
        elif self.parameters is None:
            raise Exception("The model is not ready to be trained. Please initialize parameters first.")
        else:
            print("Model is ready to be trained.")

