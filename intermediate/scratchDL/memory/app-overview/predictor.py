import numpy as np

class Predictor:
    def __init__(self):
        pass

    def predict(self, last_activation):
        """
        Computes the predictions of a neural network by taking the highest activation.

        Arguments:
            - last_activation: array of the last activations

        Returns:
            - index: the index of the highest activation
        """
        return np.argmax(last_activation, axis=0)