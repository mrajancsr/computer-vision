from dataclasses import dataclass, field
import numpy as np


@dataclass
class NeuralNetMLP:
    """Implementation of 2 Layer Neural Network,
    1 hidden layer and an output layer
    """

    d_features: int
    m_hidden_units: int
    k_classes: int
    random_seed: int = 123

    def __post_init__(self):
        rng = np.random.RandomState(self.random_seed)
        # initialize hidden layer weights
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(self.m_hidden_units, self.d_features)
        )
        self.bias_h = np.zeros(self.m_hidden_units)

        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(self.m_hidden_units, self.k_classes)
        )
        self.bias_out = np.zeros(self.k_classes)

    def __repr__(self):
        return f"NueralNetMLP(d_features={self.d_features}, m_hidden_units={self.m_hidden_units}, k_classes={self.k_classes}"

    def forward(self, x):
        pass

    def backward(self, x, a_h, a_out, y):
        pass


net = NeuralNetMLP(10, 5, 3)
print(net)
