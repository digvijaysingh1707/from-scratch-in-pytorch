# Import libraries
import os
import shutil
from datetime import datetime
import torch
from torch import nn


class SequentialDataGenerator:
    def __init__(self, file_path: str) -> None:
        """Load Dataset

        Args:
            file_path (str): File path
        """
        with open(file_path) as f:
            self.data = f.read()

        chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(chars)

        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

    def char_seq_to_int_seq(self, start_idx: int, seq_length: int) -> list:
        """Generate Character

        Args:
            start_idx (int): Start Index
            end_idx (int): End Index

        Returns:
            list: List of Int Mapped to char sequence
        """
        return [
            self.char_to_ix[ch]
            for ch in self.data[start_idx : start_idx + seq_length]
        ]


class RNNCell:
    def __init__(
        self,
        hidden_size: int,
        data_generator: SequentialDataGenerator,
        model_name="lex",
    ) -> None:
        """Initialize an RNN Cell

        Args:
            hidden_size (int): Hidden State of the RNN Cell
            input_size (int): Input Size of the RNN Cell
        """
        # Assuming that input_size = output_size
        # self.input_size = input_size
        self.data_generator = data_generator
        self.hidden_size = hidden_size
        self.model_name = model_name

        self.W_xh = torch.randn(
            hidden_size,
            self.data_generator.vocab_size,
            requires_grad=True,
            dtype=torch.float32,
        )  # input to hidden
        self.W_hh = torch.randn(
            hidden_size, hidden_size, requires_grad=True, dtype=torch.float32
        )  # hidden to hidden
        self.W_hy = torch.randn(
            self.data_generator.vocab_size,
            hidden_size,
            requires_grad=True,
            dtype=torch.float32,
        )  # hidden to output

        # scale down the weights by a factor of 0.01
        scaling_factor = 1e-2
        for param in [self.W_xh, self.W_hh, self.W_hy]:
            param = param * scaling_factor

        # biases
        self.bh = torch.randn(
            hidden_size, 1, requires_grad=True, dtype=torch.float32
        )
        self.by = torch.randn(
            self.data_generator.vocab_size,
            1,
            requires_grad=True,
            dtype=torch.float32,
        )

        self.model_params = {
            "W_xh": self.W_xh,
            "W_hh": self.W_hh,
            "W_hy": self.W_hy,
            "bh": self.bh,
            "by": self.by,
        }

        # Initialize Model Hidden States
        self.reset_hidden_states()

    def reset_hidden_states(self) -> None:
        # Start with H0
        # self.hidden_states = [torch.zeros((self.hidden_size, 1))]
        self.hidden_states = []

    def forward(
        self, inputs: list, targets: list, h_prev: torch.tensor
    ) -> torch.tensor:
        """Forward Function to return the y_pred

        Args:
            inputs (list): Input Characters (List of Ints)
            targets (list): Output Characters (List of Ints)
            h_prev (list): Prev Hidden State

        Returns:
            torch.tensor: Loss
        """
        # hs, ys = {}, {}
        self.reset_hidden_states()
        self.hidden_states.append(h_prev)

        cross_entropy_loss = nn.CrossEntropyLoss()
        loss = torch.tensor(0, dtype=torch.float32)

        # forward pass
        for t in range(len(inputs)):
            # One-Hot encode input
            x_one_hot = torch.zeros(self.data_generator.vocab_size, 1)
            x_one_hot[inputs[t]] = 1

            # Calc hidden state & output
            self.hidden_states.append(
                torch.tanh(
                    (self.W_xh @ x_one_hot)
                    + (self.W_hh @ self.hidden_states[t])
                    + self.bh
                )
            )
            y_hat = (self.W_hy @ self.hidden_states[t + 1]) + self.by
            loss += cross_entropy_loss(
                y_hat.flatten(), torch.tensor(targets[t])
            )

        # take average
        loss = loss / len(inputs)
        loss.backward(retain_graph=True)

        # Clip Grads
        for param in list(self.model_params.values()):
            param.grad = torch.clamp(param.grad, -5, 5)

        return loss

    def save_model_params(self, model_dir: str, model_iter: int) -> None:
        """Save Model Params"""

        # Make Dir for the model iteration
        os.makedirs(f"{model_dir}/{model_iter}")

        # Save Model Params
        for param_name, param in self.model_params.items():
            torch.save(param, f"{model_dir}/{model_iter}/{param_name}.pt")

    def load_model_params(self, model_dir=str, model_ckpt="latest") -> None:
        if not os.path.isdir(model_dir):
            print("Model Path doesn't exist")

        model_ckpts = next(os.walk(model_dir))[1]
        if model_ckpt == "latest":
            if "final" in model_ckpts:
                model_ckpt = "final"
            else:
                model_ckpt = max([int(x) for x in model_ckpts])
        else:
            if str(model_ckpt) not in model_ckpts:
                print("Model checkpoint not found!")
                return None

        # Load Weights
        for name, param in self.model_params.items():
            param = torch.load(f"{model_dir}/{model_ckpt}/weights_{name}.pt")

    def train(
        self,
        num_epochs: int,
        seq_length=25,
        learning_rate=1e-1,
        checkpoint_range=1000,
        verbose=False,
        save_model=True,
    ) -> None:
        """Train Model

        Args:
            data (str): Data to train upon
            num_epochs (int): Number of epochs to train on
            seq_length (int): Sequence Length. Defaults to 25.
            learning_rate (float): Learning Rate of the model. Defaults to 1e-1.
            checkpoint_range (int): Checkpoint range. Defaults to 100.
            verbose (bool): Whether to print predictions at every checkpoint. Defaults to False.
            save_model_weights (bool): Whether to save model weights. Defaults to True.
        """

        print(
            f"Data size: {self.data_generator.data_size}; Vocab Size: {self.data_generator.vocab_size}"
        )

        # Ensure checkpoints directory exists
        curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
        if self.model_name == "miniRNN":
            self.model_name += f"_{curr_time}"
        model_dir = f"saved_models/{self.model_name}"

        if os.path.isdir(model_dir):
            # to clear the existing
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        curr_epoch = 0
        while curr_epoch < num_epochs:
            curr_idx = 0
            h0 = torch.zeros((self.hidden_size, 1))
            while curr_idx + seq_length < self.data_generator.data_size:

                # Generate Current Inputs & Targets
                inputs = self.data_generator.char_seq_to_int_seq(
                    curr_idx, seq_length
                )
                targets = self.data_generator.char_seq_to_int_seq(
                    curr_idx + 1, seq_length
                )

                # Set the grad to None
                for param in list(self.model_params.values()):
                    param.grad = None

                loss = self.forward(inputs, targets, h0)

                # Update grad
                for param in list(self.model_params.values()):
                    param.data -= learning_rate * param.grad

                # Save Model
                if curr_epoch % checkpoint_range == 0:
                    learning_rate = learning_rate * 0.99
                    print(
                        f"Iteration {curr_epoch}; loss: {loss.item()}"
                    )  # print progress

                    if save_model:
                        self.save_model_params(model_dir, curr_epoch)

                curr_epoch += 1
                # Use the last hidden state
                # h0 = self.hidden_states[-1]

                curr_idx += seq_length

    def sample(self, h: torch.tensor, seed_char: str, seq_len: int) -> str:
        """Sample from the model

        Args:
            h (torch.tensor): _description_
            seed_ix (int): _description_
            seq_len (int): _description_

        Returns:
            str: _description_
        """
        curr_char = seed_char
        sample_str = seed_char
        for _ in range(seq_len):
            x_one_hot = torch.zeros(self.data_generator.vocab_size, 1)
            x_one_hot[self.data_generator.char_to_ix[curr_char]] = 1

            h = torch.tanh((self.W_xh @ x_one_hot) + (self.W_hh @ h) + self.bh)
            y_hat = (self.W_hy @ h) + self.by
            y_idx = torch.argmax(y_hat.flatten())
            curr_char = self.data_generator.ix_to_char[y_idx.item()]
            sample_str += curr_char
        return sample_str
