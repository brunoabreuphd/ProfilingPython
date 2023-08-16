from nn_model import NNModel
from data_loader import MNISTDataLoader
from propagator import Propagator
from predictor import Predictor
from metrics import Metrics
from optimizer import Optimizer

import time


if __name__ == "__main__":
    data_loader = MNISTDataLoader(one_hot_encode=True)
    learning_rate = 0.1
    layers_size = [784, 10, 10]

    model = NNModel(layers_size, data_loader, learning_rate)

    predictor = Predictor()
    metrics = Metrics()

    model.predictor = predictor
    model.metrics = metrics

    model.initialize_random_params()
    model.load_train_data()

    propagator = Propagator()
    optimizer = Optimizer(propagator)

    start = time.perf_counter()
    model, accuracies, losses = optimizer.gradient_descent(model, 200)
    stop = time.perf_counter()
    print("Time to train (s): %.2f" % (stop - start))
