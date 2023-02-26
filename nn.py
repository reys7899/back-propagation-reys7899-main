"""
File Name: nn.py
Purpose: The main code for the back propagation assignment. See README.md for details.
Author(s): Rey Sanayei
Version: 1.0 (10/03/2022)
"""
import math
from typing import List

import numpy as np


def sigmoid(input_param):
    """The Sigmoid activation function for our Nural Network

       :param input_param: the input of the activation function
       :return: The transformed input
    """
    return 1 / (1 + np.exp(-input_param))


def sigmoid_gradient(input_param):
    """The Sigmoid activation function's gradient.

           :param input_param: the input
           :return: The transformed input
        """
    return sigmoid(input_param) * (1 - sigmoid(input_param))


class SimpleNetwork:
    """A simple feedforward network where all units have sigmoid activation.
    """

    @classmethod
    def random(cls, *layer_units: int):
        """Creates a feedforward neural network with the given number of units
        for each layer.

        :param layer_units: Number of units for each layer
        :return: the neural network
        """

        def uniform(n_in, n_out):
            epsilon = math.sqrt(6) / math.sqrt(n_in + n_out)
            return np.random.uniform(-epsilon, +epsilon, size=(n_in, n_out))

        pairs = zip(layer_units, layer_units[1:])
        return cls(*[uniform(i, o) for i, o in pairs])

    def __init__(self, *layer_weights: np.ndarray):
        """Creates a neural network from a list of weight matrices.
        The weights correspond to transformations from one layer to the next, so
        the number of layers is equal to one more than the number of weight
        matrices.

        :param layer_weights: A list of weight matrices
        """
        self.num_layers = len(layer_weights)
        # list of all the layers' weights.
        self.weight = []
        # list of activation for each layer. Each element is the input of next layer.
        self.activation_list = []
        # list of outputs of each layer.
        self.output_list = []
        # populating weight list.
        for weight in layer_weights:
            self.weight.append(weight)

    def predict(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix.

        Each unit's output should be calculated by taking a weighted sum of its
        inputs (using the appropriate weight matrix) and passing the result of
        that sum through a logistic sigmoid activation function.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each in the range (0, 1) - for the corresponding row in the
        input matrix.
        """
        self.activation_list = [input_matrix.T]
        self.output_list = [None]

        for weight in self.weight:
            self.output_list.append(weight.T.dot(self.activation_list[-1]))
            self.activation_list.append(sigmoid(self.output_list[-1]))

        return self.activation_list[-1].T

    def predict_zero_one(self, input_matrix: np.ndarray) -> np.ndarray:
        """Performs forward propagation over the neural network starting with
        the given input matrix, and converts the outputs to binary (0 or 1).

        Outputs will be converted to 0 if they are less than 0.5, and converted
        to 1 otherwise.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :return: A matrix of predictions, where each row is the predicted
        outputs - each either 0 or 1 - for the corresponding row in the input
        matrix.
        """
        predictions = self.predict(input_matrix)

        return (predictions > 0.5).astype("uint8")

    def gradients(self,
                  input_matrix: np.ndarray,
                  output_matrix: np.ndarray) -> List[np.ndarray]:
        """Performs back-propagation to calculate the gradients for each of
        the weight matrices.

        This method first performs a pass of forward propagation through the
        network, then applies the following procedure to calculate the
        gradients. In the following description, × is matrix multiplication,
        ⊙ is element-wise product, and ⊤ is matrix transpose.

        First, calculate the error, error_L, between last layer's activations,
        h_L, and the output matrix, y:

        error_L = h_L - y

        Then, for each layer l in the network, starting with the layer before
        the output layer and working back to the first layer (the input matrix),
        calculate the gradient for the corresponding weight matrix as follows.
        First, calculate g_l as the element-wise product of the error for the
        next layer, error_{l+1}, and the sigmoid gradient of the next layer's
        weighted sum (before the activation function), a_{l+1}.

        g_l = (error_{l+1} ⊙ sigmoid'(a_{l+1}))⊤

        Then calculate the gradient matrix for layer l as the matrix
        multiplication of g_l and the layer's activations, h_l, divided by the
        number of input examples, N:

        grad_l = (g_l × h_l)⊤ / N

        Finally, calculate the error that should be backpropagated from layer l
        as the matrix multiplication of the weight matrix for layer l and g_l:

        error_l = (weights_l × g_l)⊤

        Once this procedure is complete for all layers, the grad_l matrices
        are the gradients that should be returned.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :return: two matrices of gradients, one for the input-to-hidden weights
        and one for the hidden-to-output weights
        """
        input_example_count = input_matrix.shape[0]
        # list of calculated gradients for each weight matrix.
        delta_w = []
        # same list, returned at the end of calculation.
        delta_w_return = []
        # Applying forward propagation.
        y_hat = self.predict(input_matrix)
        _error = np.sum((y_hat - output_matrix) ** 2)

        for index in reversed(range(1, self.num_layers + 1)):
            if index == self.num_layers:
                activation_gradient = self.activation_list[index] - output_matrix.T
            else:
                activation_gradient = self.weight[index].dot(output_gradient)

            output_gradient = activation_gradient * sigmoid_gradient(self.output_list[index])
            # The final gradient to calculate, which we use to update weights.
            weight_gradient = output_gradient.dot(self.activation_list
                                                  [index - 1].T) / input_example_count
            delta_w.append(weight_gradient)
            # Reversing the list. First layer's weights comes first.
            delta_w_return.append(weight_gradient.T)

        return list(reversed(delta_w_return))

    def train(self,
              input_matrix: np.ndarray,
              output_matrix: np.ndarray,
              iterations: int = 10,
              learning_rate: float = 0.1) -> None:
        """Trains the neural network on an input matrix and an expected output
        matrix.

        Training should repeatedly (`iterations` times) calculate the gradients,
        and update the model by subtracting the learning rate times the
        gradients from the model weight matrices.

        :param input_matrix: The matrix of inputs to the network, where each
        row in the matrix represents an instance for which the neural network
        should make a prediction
        :param output_matrix: A matrix of expected outputs, where each row is
        the expected outputs - each either 0 or 1 - for the corresponding row in
        the input matrix.
        :param iterations: The number of gradient descent steps to take.
        :param learning_rate: The size of gradient descent steps to take, a
        number that the gradients should be multiplied by before updating the
        model weights.
        """
        # Training the network and updating the weights.
        i = 0
        while i in range(iterations):
            self.predict(input_matrix)
            gradients = self.gradients(input_matrix, output_matrix)
            for index in range(self.num_layers):
                self.weight[index] -= learning_rate * gradients[index]
            i += 1
