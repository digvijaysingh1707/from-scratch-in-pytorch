import torch
from torch.nn import CrossEntropyLoss

# Define Layer in NN
class Layer:
    def __init__(
        self,
        n_inputs_per_neuron: int,
        neurons_per_layer: int,
        activation_function=torch.tanh,
    ) -> None:
        self.weights = torch.randn(
            neurons_per_layer,
            n_inputs_per_neuron,
            requires_grad=True,
            dtype=torch.float32,
        )
        self.biases = torch.randn(
            neurons_per_layer, 1, requires_grad=True, dtype=torch.float32
        )
        self.activation_function = activation_function

    def return_model_params(self) -> list:
        return [self.weights, self.biases]

    def forward_pass(self, x: torch.tensor) -> torch.tensor:
        # Get weighted sum + bias
        weighted_sum = self.weights @ x + self.biases
        if self.activation_function == "none":
            return weighted_sum
        else:
            return self.activation_function(weighted_sum)


class MLP:
    def __init__(self, input_dim: int, layers_structure: list) -> None:
        # layers_structure = [16, 16, 16, 3]
        self.layers = []
        input_neurons = input_dim
        for structure in layers_structure[:-1]:
            self.layers.append(Layer(input_neurons, structure))
            input_neurons = structure

        # Last layer
        self.layers.append(
            Layer(
                input_neurons, layers_structure[-1], activation_function="none"
            )
        )

    def return_model_params(self) -> list:
        model_params = []
        for layer in self.layers:
            model_params += layer.return_model_params()
        return model_params

    def forward_pass(self, x) -> torch.tensor:
        input_x = x
        for layer in self.layers:
            input_x = layer.forward_pass(input_x)
        return input_x

    def train(
        self, xs: torch.tensor, ys: torch.tensor, lr=0.1, epochs=100
    ) -> None:
        loss_func = CrossEntropyLoss()
        model_params = self.return_model_params()

        for epoch in range(epochs):
            # Do the forward pass
            loss = torch.zeros(1)
            for i in range(len(xs)):
                y_pred = self.forward_pass(xs[i].reshape(xs[i].shape[0], 1))
                loss += loss_func(y_pred.reshape(y_pred.shape[0]), ys[i])

            # normalize loss
            loss /= len(xs)
            print(f"Epoch: {epoch}; Loss: {loss.item()}...")

            # Do gradient descent
            for param in model_params:
                param.grad = None

            loss.backward()

            # update params
            for param in model_params:
                param.data -= lr * param.grad
