import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam


# Define Layer in NN
class Layer(nn.Module):
    def __init__(
        self,
        n_inputs_per_neuron: int,
        neurons_per_layer: int,
        activation_function=torch.tanh,
    ) -> None:
        """Initialize an MLP Layer

        Args:
            n_inputs_per_neuron (int): Inputs per neuron.
            neurons_per_layer (int): Neurons in the layer
            activation_function (_type_, optional): Activation function to use for the layer.
                Defaults to torch.tanh.
        """
        super(Layer, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.randn(neurons_per_layer, n_inputs_per_neuron)
        )
        self.biases = torch.nn.Parameter(torch.randn(neurons_per_layer, 1))
        self.activation_function = activation_function

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Forward Pass of the layer

        Args:
            x (torch.tensor): Input to the layer

        Returns:
            torch.tensor: Output of the layer
        """
        weighted_sum = self.weights @ x + self.biases
        if self.activation_function:
            return self.activation_function(weighted_sum)
        else:
            return weighted_sum


class MLP(nn.Module):
    def __init__(self, input_dim: int, layers_structure: list) -> None:
        """Initialize the MLP

        Args:
            input_dim (int): Input dimension
            layers_structure (list): Layer Structure.
                Eg: [16, 16, 16, 3] -> (3 is the output dim)
        """
        super().__init__()
        self.layers = []
        self.losses = []
        self.model_performance = []
        self.input_dim = input_dim
        self.output_dim = layers_structure[-1]

        input_neurons = input_dim
        for layer_idx in range(len(layers_structure)):
            if layer_idx < len(layers_structure) - 1:
                activation_function = torch.tanh
            else:
                activation_function = None

            layer_name = "layer_" + str(layer_idx)
            setattr(
                self,
                layer_name,
                Layer(
                    input_neurons,
                    layers_structure[layer_idx],
                    activation_function,
                ),
            )
            self.layers.append(getattr(self, layer_name))
            input_neurons = layers_structure[layer_idx]

    def __call__(self, x) -> torch.tensor:
        """Forward Pass of the model

        Args:
            x (torch.tensor): Input to the model

        Returns:
            torch.tensor: Output of the forward pass
        """
        input_x = x
        for layer in self.layers:
            input_x = layer(input_x)
        return input_x

    def train(
        self, xs: torch.tensor, ys: torch.tensor, lr=0.1, epochs=100
    ) -> None:
        """Train Model

        Args:
            xs (torch.tensor): Input features to train on.
                Shape: (len(x_train) * num_x_params)
            ys (torch.tensor): Labels corresponding to the input features.
                Shape: (len(x_train) * num_label_classes)
            lr (float, optional): Learning Rate. Defaults to 0.1.
            epochs (int, optional): Number of Epochs to train on. Defaults to 100.
        """
        loss_func = CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # model_params = self.return_model_params()

        for epoch in range(epochs):
            y_preds = torch.zeros((len(xs), self.output_dim))

            # Do the forward pass
            loss = torch.zeros(1)
            for i in range(len(xs)):
                y_preds[i] = self(xs[i].reshape(self.input_dim, 1)).flatten()
                loss += loss_func(y_preds[i], ys[i])

            # normalize loss
            loss /= len(xs)

            # Calculate accuracy
            accuracy = (
                torch.argmax(y_preds, dim=1) == torch.argmax(ys, dim=1)
            ).sum().item() / len(xs)

            # Log performance
            print(
                f"Epoch: {epoch}; Loss: {round(loss.item(), 2)}; Accuracy: {round(accuracy, 2)}..."
            )

            # Track metric
            self.losses.append(loss.item())
            self.model_performance.append(accuracy)

            # Do gradient descent
            optimizer.zero_grad()
            loss.backward()

            # update params
            optimizer.step()

    def plot_model(self, metric: str) -> plt:
        """Plot Model

        Args:
            metric (str): Metric to plot.
                Either "loss" or "accuracy"

        Returns:
            plt: A plot of Metric Vs. Epochs
        """
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        if metric == "loss":
            plt.plot(self.losses)
        elif metric == "accuracy":
            plt.plot(self.model_performance)
        else:
            print("Metric not found.")

    def predict(self, x: torch.tensor) -> int:
        """Predict Model Output for a given x

        Args:
            x (torch.tensor): Input

        Returns:
            int: Predicted class index
        """
        # Reshape if required
        if tuple(x.shape) == (self.input_dim,):
            x = x.reshape(self.input_dim, 1)
        return self(x).flatten().argmax().item()
