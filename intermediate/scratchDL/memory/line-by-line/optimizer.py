import utils

class Optimizer:
    def __init__(self, propagator):
        self.propagator = propagator

    @profile
    def gradient_descent(self, model, iters):
        """
        Trains the model using gradient descent.
    
        Arguments:
            - model: the model to be trained
            - iters: the number of iterations to train the model
        """
        m = model
        L = len(m.parameters) // 2

        accuracies = []
        losses = []

        for it in range(1, iters + 1):
            # compute activations
            m = self.propagator.forward(m)

            # make predictions (right-most set of activations)
            Y_hat = m.predictor.predict(m.activations["A"+str(L)])

            # compute accuracy
            accuracy = m.metrics.accuracy(Y_hat, m.outputs)
            accuracies.append(accuracy)

            # compute loss
            loss = m.metrics.cross_entropy(m.encoded_outputs, m.activations["A"+str(L)])
            losses.append(loss)

            # compute gradients
            m = self.propagator.backward(m)

            # update parameters
            m = self.update_params(m)

            if it % 100 == 0:
                print("Iteration: ", it, "Loss: ", loss, "Accuracy: ", accuracy)

        return m, accuracies, losses


    def update_params(self, model):
        """
        Updates the parameters of the network using gradient descent.
            
        Arguments:
            - model: the model that is being trained.
    
        Returns:
            - model: the updated model.
        """
        m = model
        
        # number of layers
        L = len(m.parameters) // 2
        
        params_updated = {}
        for l in range(1, L+1):
            params_updated["W"+str(l)] = m.parameters["W"+str(l)] - m.learning_rate * m.gradients["dW"+str(l)]
            params_updated["b"+str(l)] = m.parameters["b"+str(l)] - m.learning_rate * m.gradients["db"+str(l)]

        m.parameters = params_updated

        return m
