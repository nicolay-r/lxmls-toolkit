import numpy as np
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_class_probabilities, _ = self.log_forward(input)
        return np.argmax(np.exp(log_class_probabilities), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """

        gradients = self.backpropagation(input, output)

        learning_rate = self.config['learning_rate']
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):

            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]

            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]

    def log_forward(self, input):
        """Forward pass for sigmoid hidden layers and output softmax"""

        # Input
        tilde_z = input
        layer_inputs = []

        # Hidden layers
        num_hidden_layers = len(self.parameters) - 1
        for n in range(num_hidden_layers):

            # Store input to this layer (needed for backpropagation)
            layer_inputs.append(tilde_z)

            # Linear transformation
            weight, bias = self.parameters[n]
            z = np.dot(tilde_z, weight.T) + bias

            # Non-linear transformation (sigmoid)
            tilde_z = 1.0 / (1 + np.exp(-z))

        # Store input to this layer (needed for backpropagation)
        layer_inputs.append(tilde_z)

        # Output linear transformation
        weight, bias = self.parameters[num_hidden_layers]
        z = np.dot(tilde_z, weight.T) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

        return log_tilde_z, layer_inputs

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        num_examples = input.shape[0]
        log_probability, _ = self.log_forward(input)
        return -log_probability[range(num_examples), output].mean()

    def backpropagation(self, input, output):
        """Gradients for sigmoid hidden layers and output softmax"""

        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input)
        prob_y = np.exp(log_prob_y)

        num_examples, num_clases = prob_y.shape
        num_hidden_layers = len(self.parameters) - 1

        # For each layer in reverse store the backpropagated error, then compute
        # the gradients from the errors and the layer inputs
        errors = []

        # ----------
        # Solution to Exercise 2
        I = index2onehot(output, num_clases)

        is_last = True
        for layer_ind in reversed(range(num_hidden_layers + 1)):
            if is_last:
                errors.append(I - prob_y)
                is_last = False
            else:
                w = self.parameters[layer_ind + 1][0]

                e = np.matmul(np.transpose(w), np.transpose(errors[-1]))
                e = np.transpose(e)

                layer = layer_inputs[layer_ind + 1]
                invert_layer = 1 - layer

                e_r = np.multiply(e, layer, invert_layer)
                errors.append(e_r)

        gradients = []
        for i, e in enumerate(reversed(errors)):
            # print(layer_inputs[i].shape)
            # print(e.shape)
            g_w = (- 1.0 / num_examples) * np.matmul(np.transpose(e), layer_inputs[i])
            g_b = (- 1.0 / num_examples) * np.sum(e, axis=0)
            gradients.append([g_w, g_b])

        # End of solution to Exercise 2
        # ----------

        return gradients
