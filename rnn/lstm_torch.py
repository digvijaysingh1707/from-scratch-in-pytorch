# import libraries
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam

torch.autograd.set_detect_anomaly(True)
device = torch.device("cpu")


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

    def char_seq_to_int_seq(
        self, start_idx: int, seq_length: int
    ) -> torch.tensor:
        """Generate Character

        Args:
            start_idx (int): Start Index
            end_idx (int): End Index

        Returns:
            list: List of Int Mapped to char sequence
        """
        return torch.tensor(
            [
                self.char_to_ix[ch]
                for ch in self.data[start_idx : start_idx + seq_length]
            ],
            device=device,
        )

    def char_seq_to_one_hot(
        self, start_idx: int, seq_length: int
    ) -> torch.tensor:
        """One Hot Encode Chars

        Args:
            int_sequence (list): Sequence of Char Ints

        Returns:
            torch.tensor: One hot encoded tensor
        """
        one_hot_encoded = torch.zeros(
            seq_length, self.vocab_size, device=device
        )
        for char_idx in range(seq_length):
            one_hot_encoded[char_idx][
                self.char_to_ix[self.data[char_idx + start_idx]]
            ] = 1
        return one_hot_encoded


# Single layer LSTM for now
# TODO: Convert this into n-layered LSTM
class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_state_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.lstm_cell = nn.LSTMCell(
            input_size=input_size, hidden_size=hidden_state_size
        )
        self.fc = nn.Linear(hidden_state_size, input_size)
        self.reset_lstm_state()

    def reset_lstm_state(self) -> None:
        """Reset LSTM state (h_x, c_x)"""
        self.h_x = torch.zeros(self.hidden_state_size, device=device)
        self.c_x = torch.zeros(self.hidden_state_size, device=device)

    def train_model(
        self,
        datagen: SequentialDataGenerator,
        num_epochs: int,
        seq_length=50,
        lr=1e-1,
        checkpoint_range=1,
        # save_model=True,
    ) -> None:
        """Train Model

        Args:
            datagen (SequentialDataGenerator): Data Generator for LSTM.
            num_epochs (int): Number of epochs to train on
            seq_length (int): Sequence Length. Defaults to 25.
            learning_rate (float): Learning Rate of the model.
                Defaults to 1e-1.
            checkpoint_range (int): Checkpoint range. Defaults to 100.
            save_model_weights (bool): Whether to save model weights.
                Defaults to True.
        """

        print(
            f"Data size: {datagen.data_size}; Vocab Size: {datagen.vocab_size}"
        )

        self.model_name = "harsh"
        # Ensure checkpoints directory exists
        curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
        if self.model_name == "miniRNN":
            self.model_name += f"_{curr_time}"
        model_dir = f"saved_models/{self.model_name}"

        if os.path.isdir(model_dir):
            # to clear the existing
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        # Define Loss function & Optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        curr_epoch = 0
        max_batch_idx = int(len(datagen.data) / seq_length)

        self.losses = []

        while curr_epoch < num_epochs:
            curr_batch = 0
            while curr_batch < max_batch_idx - 1:
                # Generate inputs and targets
                x_data = datagen.char_seq_to_one_hot(
                    curr_batch * seq_length, seq_length
                )
                y_data = datagen.char_seq_to_int_seq(
                    curr_batch * seq_length + 1, seq_length
                )

                # Set grads to Zero
                optimizer.zero_grad()

                # Detatch hidden states computation graphs
                self.h_x = self.h_x.detach()
                self.c_x = self.c_x.detach()

                # forward pass
                out = []
                for i in range(x_data.size()[0]):
                    self.h_x, self.c_x = self.lstm_cell(
                        x_data[i], (self.h_x, self.c_x)
                    )
                    out.append(self.fc(self.h_x))

                loss = loss_func(torch.stack(out, dim=0), y_data)
                loss.backward()
                self.losses.append(loss.item())

                # Update weights
                optimizer.step()
                curr_batch += 1

            # Save Model
            if curr_epoch % checkpoint_range == 0:
                print(
                    f"Iteration {curr_epoch}; loss: {loss.item()}"
                )  # print progress
                torch.save(self.state_dict(), f"{model_dir}/{curr_epoch}")
                # if save_model:
                #     self.save_model_params(model_dir, curr_epoch)

            self.reset_lstm_state()
            curr_epoch += 1
