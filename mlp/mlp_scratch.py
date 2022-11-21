import torch
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss

# Define Layer in NN
class Layer:
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

    def return_layer_params(self) -> list:
        """Return a list of layer params

        Returns:
            list: List of layer params
        """
        return [self.weights, self.biases]

    def __call__(self, x: torch.tensor) -> torch.tensor:
        """Forward Pass of the layer

        Args:
            x (torch.tensor): Input to the layer

        Returns:
            torch.tensor: Output of the layer
        """
        # Get weighted sum + bias
        weighted_sum = self.weights @ x + self.biases
        if self.activation_function == "none":
            return weighted_sum
        else:
            return self.activation_function(weighted_sum)


class MLP:
    def __init__(self, input_dim: int, layers_structure: list) -> None:
        """Initialize the MLP

        Args:
            input_dim (int): Input dimension
            layers_structure (list): Layer Structure.
                Eg: [16, 16, 16, 3] -> (3 is the output dim)
        """
        self.layers = []
        self.losses = []
        self.model_performance = []
        self.input_dim = input_dim
        self.output_dim = layers_structure[-1]

        input_neurons = self.input_dim
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
        """Return a list of model params

        Returns:
            list: List of model params
        """
        model_params = []
        for layer in self.layers:
            model_params += layer.return_layer_params()
        return model_params

    def __call__(self, x: torch.tensor) -> torch.tensor:
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
        model_params = self.return_model_params()

        for epoch in range(epochs):
            y_preds = torch.zeros((len(xs), self.output_dim))

            # Do the forward pass
            loss = torch.zeros(1)
            for i in range(len(xs)):
                y_preds[i] = self(xs[i].reshape(xs[i].shape[0], 1)).flatten()
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
            for param in model_params:
                param.grad = None

            loss.backward()

            # update params
            for param in model_params:
                param.data -= lr * param.grad

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
