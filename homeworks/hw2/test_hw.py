import math
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


class ReLU(object):
    def __repr__(self):
        return f"Relu"

    def forwards(self, x):
        return np.where(x > 0, x, 0)

    def backwards(self, x, grad_output):
        assert x.shape == grad_output.shape

        # Return the gradient with respect to the input.
        return np.where(x > 0, grad_output, 0)

    def backwards_param(self, x, grad_output):
        return None  # We return none because there are no learnable parameters.


class FullyConnectedLayer(object):
    def __init__(self, in_dim, out_dim):
        self.out_dim = out_dim  # During forward pass, the resulting output vector will have out_dim dimensions
        self.in_dim = (
            in_dim  # During forward pass, each input vector will have in_dim dimensions
        )

        # Create an initial guess for the parameters w by sampling from a Gaussian
        # distribution.
        self.w = np.random.normal(
            0, math.sqrt(2 / (in_dim + out_dim)), [out_dim, in_dim]
        )

    def __repr__(self):
        return f"FullyConnectedLayer({self.in_dim} , {self.out_dim})"

    def forwards(self, x):
        # Computes the forward pass of a linear layer (which is also called
        # fully connected). The input x will be a matrix that has the shape:
        # (in_dim)x(batch_size). The output should be a matrix that has the
        # shape (out_dim)x(batch_size). Note: in this implementation, there is
        # no bias term in order to keep it simple.
        assert x.shape[0] == self.in_dim
        # Return the result of the forwards pass.
        return self.w @ x

    def backwards(self, x, grad_output):
        assert grad_output.shape[0] == self.out_dim
        # Return the gradient with respect to the input.
        return self.w.T @ grad_output

    def backwards_param(self, x, grad_output):
        assert grad_output.shape[0] == self.out_dim
        # Return the gradient with respect to the parameters.
        return grad_output @ x.T

    def update_param(self, grad):
        # Given the gradient with respect to the parameters, perform a gradient step.
        # This function should modify self.w based on grad. You should implement
        # the basic version of gradient descent. The function does not return anything.
        self.w -= 0.1 * grad


def test_gradient_output(name, layer, x, epsilon=1e-7):
    grad_approx = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] += epsilon
            fxph = layer.forwards(x)
            x[i][j] -= 2 * epsilon
            fxmh = layer.forwards(x)
            x[i][j] += epsilon
            grad_approx[i][j] = (fxph - fxmh).sum() / (2 * epsilon)
    grad_backprop = layer.backwards(x, np.ones(layer.forwards(x).shape))

    numerator = np.linalg.norm(grad_backprop - grad_approx)
    denominator = np.linalg.norm(grad_backprop) + np.linalg.norm(grad_approx)
    difference = numerator / (denominator)
    if difference < 1e-7:
        print("OK\t", name)
    else:
        print("FAIL\t", name, "Difference={}".format(difference))


def sample_batch(x):
    y = np.expand_dims(np.abs(x[0, :]) - np.abs(x[1, :]), axis=0)
    return y


def test_equality(name, actual, expected):
    result = (np.abs(actual - expected) < 1e-7).all()
    if result:
        print("OK\t", name)
    else:
        print("FAIL\t", name)
        print("Actual:")
        print(actual)
        print("Expected:")
        print(expected)


test_input = np.array([[10.0, -5.0, 3.0, 0.0, 2.0, -1.0]])
expected_output = np.array([[10.0, 0.0, 3.0, 0.0, 2.0, 0.0]])
actual_output = ReLU().forwards(test_input)
test_equality("ReLU Forward 1", actual_output, expected_output)

test_input = np.array(
    [[10.0, -5.0, 3.0, 0.0, 2.0, -1.0], [3.0, 2.0, 1.0, 0.0, -1.0, -2.0]]
)
expected_output = np.array(
    [[10.0, 0.0, 3.0, 0.0, 2.0, 0.0], [3.0, 2.0, 1.0, 0.0, 0.0, 0.0]]
)
actual_output = ReLU().forwards(test_input)
test_equality("ReLU Forward Batch", actual_output, expected_output)

layer = ReLU()
test_input = np.array([[3.0, -4, 2]]).T
test_gradient_output("ReLU Output Gradient", layer, test_input)

layer = ReLU()
test_input = np.array([[6, -1, -3], [4, 0.1, 2]]).T
test_gradient_output("ReLU Output Gradient Batch", layer, test_input)

layer = FullyConnectedLayer(6, 2)
layer.w[0, :] = -1
layer.w[1, :] = 2
test_input = np.array([[10, -5, 3, 0, 2, -1]]).T
expected_output = np.array([[-test_input.sum(), 2 * test_input.sum()]]).T
actual_output = layer.forwards(test_input)
test_equality("Fully Connected Forward 2", actual_output, expected_output)

layer = FullyConnectedLayer(3, 2)
layer.w[0, :] = 1
layer.w[1, :] = 0.5
test_input = np.array([[1, 2, 3], [-4, -5, -6]]).T
expected_output = np.array(
    [
        [test_input[:, 0].sum(), 0.5 * test_input[:, 0].sum()],
        [test_input[:, 1].sum(), 0.5 * test_input[:, 1].sum()],
    ]
).T
actual_output = layer.forwards(test_input)
test_equality("Fully Connected Forward Batch", actual_output, expected_output)

print("Done.")
##########################################################################################

layer = FullyConnectedLayer(1, 1)
layer.w[:] = 2.0
test_input = np.array([[3.0]]).T
test_gradient_output("Fully Connected Output Gradient 1", layer, test_input)

layer = FullyConnectedLayer(6, 2)
test_input = np.array([[-1.0, 2.0, -3.0, 4.0, 5.0, 6.0]]).T
test_gradient_output("Fully Connected Output Gradient 2", layer, test_input)

layer = FullyConnectedLayer(100, 100)
test_input = np.random.randn(100, 1)
test_gradient_output("Fully Connected Output Gradient 3", layer, test_input)

layer = FullyConnectedLayer(100, 100)
test_input = np.random.randn(100, 50)
test_gradient_output("Fully Connected Output Gradient Batch", layer, test_input)

print("Done.")


#####################################################################################
def test_gradient_param(name, layer, x, epsilon=1e-7):
    grad_approx = np.zeros(layer.w.shape)
    for i in range(layer.w.shape[0]):
        for j in range(layer.w.shape[1]):
            layer.w[i][j] += epsilon
            fxph = layer.forwards(x)
            layer.w[i][j] -= 2 * epsilon
            fxmh = layer.forwards(x)
            layer.w[i][j] += epsilon
            grad_approx[i][j] = (fxph - fxmh).sum() / (2 * epsilon)
    grad_backprop = layer.backwards_param(x, np.ones(layer.forwards(x).shape))
    numerator = np.linalg.norm(grad_backprop - grad_approx)
    denominator = np.linalg.norm(grad_backprop) + np.linalg.norm(grad_approx)
    difference = numerator / (denominator + epsilon)
    if difference < 1e-7:
        print("OK\t", name)
    else:
        print("FAIL\t", name, "Difference={}".format(difference))


layer = FullyConnectedLayer(1, 1)
test_input = np.array([[1]]).T
test_gradient_param("Fully Connected Params Gradient 1", layer, test_input)

layer = FullyConnectedLayer(6, 2)
test_input = np.array([[-1, 2, -3, 4, 5, 6]]).T
test_gradient_param("Fully Connected Params Gradient 2", layer, test_input)

layer = FullyConnectedLayer(100, 100)
test_input = np.random.randn(100, 1)
test_gradient_param("Fully Connected Params Gradient 3", layer, test_input)

layer = FullyConnectedLayer(100, 100)
test_input = np.random.randn(100, 50)
test_gradient_param("Fully Connected Params Gradient Batch", layer, test_input)

print("Done.")


###############PROBLEM2################################
def euclidean_loss(prediction, target):
    assert prediction.shape == target.shape
    # TODO: Implement a function that computes:
    # - the loss (scalar)
    # - the gradient of the loss with respect to the prediction (tensor)
    # The function should return these two values as a tuple.
    loss = np.sum((target - prediction) ** 2) / target.shape[1]
    grad = (prediction - target) / target.shape[0]
    return loss, grad


def test_gradient_loss(name, x, target, epsilon=1e-7):
    grad_approx = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i][j] += epsilon
            fxph, _ = euclidean_loss(x, target)
            x[i][j] -= 2 * epsilon
            fxmh, _ = euclidean_loss(x, target)
            x[i][j] += epsilon
            grad_approx[i][j] = (fxph - fxmh).sum() / (2 * epsilon)
    _, grad_exact = euclidean_loss(x, target)
    numerator = np.linalg.norm(grad_exact - grad_approx)
    denominator = np.linalg.norm(grad_exact) + np.linalg.norm(grad_approx)
    difference = numerator / (denominator + epsilon)
    if difference < 1e-7:
        print("OK\t", name)
    else:
        print("FAIL\t", name, "Difference={}".format(difference))


test_a = np.array([[-1.0, 5, -2, 0], [4.3, -10, 7.8, 8.4]])
test_b = np.array([[1.0, -3, -2, 2], [8.4, 7.8, -10, 4.3]])
loss, _ = euclidean_loss(test_a, test_a)
test_equality("Euclidean Loss 1", loss, 0)
loss, _ = euclidean_loss(test_a, test_b)
test_equality("Euclidean Loss 2", loss, 739.3 / 4.0)

test_gradient_loss("Euclidean Loss Gradient 1", test_a, test_a)
test_gradient_loss("Euclidean Loss Gradient 2", test_a, test_b)

print("Done.")


class NeuralNetwork(object):
    def __init__(self):
        self.layers: List[Union[ReLU, FullyConnectedLayer]] = []
        self.inputs = []
        self.grad_params = []

    def add(self, layer):
        self.layers.append(layer)

    def forwards(self, x):
        self.inputs = []
        out = x
        for layer in self.layers:
            self.inputs.append(out)
            out = layer.forwards(out)
        return out

    def backwards(self, grad_output):
        assert len(self.inputs) == len(self.layers)
        self.grad_params = []  # store gradients of the parameters for each layer

        n = len(self.layers) - 1
        for layerid in range(n, -1, -1):
            x = self.inputs[layerid]
            grad_params = self.layers[layerid].backwards_param(x, grad_output)
            grad_output = self.layers[layerid].backwards(x, grad_output)

            if grad_params is not None:  # store grad_params for update_param() below
                self.grad_params.append((layerid, grad_params))
        return grad_output

    def update_param(self, step_size=0.001):
        for layerid, grad_param in self.grad_params:
            self.layers[layerid].update_param(grad_param * step_size)


nn = NeuralNetwork()
nn.add(FullyConnectedLayer(2, 50))
nn.add(ReLU())
nn.add(FullyConnectedLayer(50, 1))


def sample_batch(batch_size):
    x = np.random.randn(2, batch_size)
    y = np.expand_dims(np.abs(x[0, :]) - np.abs(x[1, :]), axis=0)
    return x, y


loss_values = []

for iter in range(100001):
    input_data, target = sample_batch(100)

    output = nn.forwards(input_data)
    loss, grad_loss = euclidean_loss(output, target)
    nn.backwards(grad_loss)
    nn.update_param(step_size=0.0001)

    if iter % 10000 == 0:
        print(f"Iter{iter} Loss={loss}")

    loss_values.append(loss)

plt.clf()
plt.plot(loss_values, color="blue")
plt.ylabel("Loss Value")
plt.xlabel("Iteration")
plt.show()
