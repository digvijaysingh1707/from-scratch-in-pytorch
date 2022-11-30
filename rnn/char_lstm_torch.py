"""
The code is a reimplementation of https://www.kaggle.com/code/francescapaulin/character-level-lstm-in-pytorch/notebook
which in turn is an implementation of Andrej Karpathy's implementation of char-RNN in torch (lua) -
https://github.com/karpathy/char-rnn
"""

# import libraries
import os
import shutil
from datetime import datetime
import numpy as np
import torch
from torch import nn
import torch.functional as F
from torch.optim import Adam
import os
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using Device: {device}")


class SequentialDataGenerator:
    def __init__(self, file_path: str) -> None:
        """Load Dataset

        Args:
            file_path (str): File path
        """
        with open(file_path) as f:
            self.data = f.read()

        self.unique_chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(self.unique_chars)

        self.char_to_ix = {ch: i for i, ch in enumerate(self.unique_chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.unique_chars)}
        self.encoded_data = torch.tensor(
            [self.char_to_ix[x_] for x_ in self.data]
        ).to(device)
        self.dataset_size = len(self.encoded_data)

    def one_hot_encode(self, arr: torch.tensor) -> torch.tensor:
        """Encodes a series of encoded ints of n dimension
            on self.vocab_size.

        Args:
            arr (torch.tensor): N-dimensional array to one-hot encode

        Returns:
            torch.tensor: One-Hot Encoded torch.tensor
                with shape of original tensor * self.vocab_size
        """

        # Initialize the the encoded array
        one_hot = torch.zeros(
            (torch.multiply(*arr.shape), self.vocab_size), dtype=torch.float32
        ).to(device)
        # Fill the appropriate elements with ones
        one_hot[torch.arange(one_hot.shape[0]), arr.flatten()] = 1.0

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, self.vocab_size))
        return one_hot

    def generate_batches(
        self, batch_size: int, seq_length: int, start_idx=0, stop_idx=-1
    ) -> tuple:
        """Return a generator that yields x & y

        Args:
            batch_size (int): Batch Size
            seq_length (int): Sequence Length
            start_idx (int): Start index of the data
            end_idx (int): End index of the data

        Yields:
            Iterator[tuple]: (one-hot encoded) x and y
        """
        trimmed_dataset = (
            self.encoded_data[start_idx:stop_idx]
            if stop_idx != -1
            else self.encoded_data[start_idx:]
        )
        chars_in_batch = batch_size * seq_length
        n_batches = len(trimmed_dataset) // chars_in_batch
        trimmed_dataset = trimmed_dataset[: n_batches * chars_in_batch + 1]

        x_batched = trimmed_dataset[:-1].reshape(batch_size, -1)
        y_batched = trimmed_dataset[1:].reshape(batch_size, -1)

        # reshape to add the batch dimension
        curr_idx = 0
        for __ in range(n_batches):
            yield (
                # one-hot encode x
                self.one_hot_encode(
                    x_batched[:, curr_idx : curr_idx + seq_length]
                ),
                # don't encode y
                y_batched[:, curr_idx : curr_idx + seq_length],
            )
            curr_idx += seq_length


class CharLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: int,
        batch_first=True,
    ) -> None:
        """Initialize char LSTM

        Args:
            input_size (int): Num Features of every input. Eg: Vocab Size.
            hidden_size (int): Size of the hidden layer
            num_layers (int): Num of LSTMs stacked on top of each other
            dropout (int): Dropout probability between the layers of the LSTM.
            batch_first (bool): Have batched training. Defaults to True.
        """
        super().__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Add the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=batch_first,
        )

        # Add the dropout layer
        self.dropout = nn.Dropout(dropout)

        # Add the Fully Connected layer
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.tensor, hidden_state: tuple) -> tuple:
        """Define the forward function of the network

        Args:
            x (torch.tensor): Input to the model.
                - Shape: (Batch Size * self.input_size).
                - Should be one-hot encoded.
            hidden_state (tuple): Hidden State to the LSTM.
                Includes (h_x, c_x) of size self.hidden_size * batch_size

        Returns:
            tuple: containing output and modified hidden state
        """
        # forward pass of LSTM
        out_lstm, hidden_state = self.lstm(x, hidden_state)

        # forward pass of Dropout
        out_dropout = self.dropout(out_lstm)

        # forward pass of FC layer
        # TODO: check the need of contiguos
        out_fc = self.fc(out_dropout.contiguous().view(-1, self.hidden_size))

        return out_fc, hidden_state

    def init_hidden(self, batch_size: int) -> tuple:
        """Initialize the hidden state of LSTM

        Args:
            batch_size (int): Batch Size.

        Returns:
            tuple: Hidden State to the LSTM.
                Includes (h_x, c_x) of size n_layer * batch_size * hidden_size
        """
        hidden_state = (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=torch.float32,
            ).to(device),
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=torch.float32,
            ).to(device),
        )
        return hidden_state

    def train_model(
        self,
        datagen: SequentialDataGenerator,
        epochs=100,
        batch_size=100,
        seq_length=50,
        lr=0.001,
        clip=5,
        val_frac=0.1,
        save_model=True,
        model_name="charLSTM",
    ):
        """Train Model class

        Args:
            epochs (int): Number of epochs to train. Defaults to 100.
            batch_size (int): Number of sequences in a batch. Defaults to 100.
            seq_length (int): Sequence Length. Defaults to 50.
            lr (float): Learning Rate. Defaults to 0.001.
            clip (int): Gradient Clipping. Defaults to 5.
            val_frac (float): Fraction of data to train on. Defaults to 0.1.
            print_every (int): Num steps to print train and validation loss. Defaults to 10.
        """
        print(
            f"Data size: {datagen.data_size}; Vocab Size: {datagen.vocab_size}"
        )

        if save_model:
            # Ensure checkpoints directory exists
            if model_name == "charLSTM":
                curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
                model_name += f"_{curr_time}"
            model_dir = f"saved_models/{model_name}"

            if os.path.isdir(model_dir):
                # to clear the existing
                shutil.rmtree(model_dir)
            os.makedirs(model_dir)

        # Specify the model to be in train state
        self.to(device=device)
        self.train()

        # define optimizer and loss function
        adam = Adam(self.parameters(), lr=lr)
        loss_function = nn.CrossEntropyLoss()

        split_idx = int(datagen.dataset_size * (1 - val_frac))

        for epoch in range(1, epochs + 1):
            counter = 0
            # initialize hidden state to zero for every epoch
            h = self.init_hidden(batch_size)

            data_generator = datagen.generate_batches(
                batch_size, seq_length, stop_idx=split_idx
            )
            num_batches = split_idx // (batch_size * seq_length)
            with tqdm(
                data_generator, total=num_batches, unit="batch"
            ) as tepoch:
                for x, y in tepoch:
                    counter += 1
                    tepoch.set_description(f"Epoch {epoch}")

                    # detatch h to ensure we don't backpropagate through history
                    h = tuple([each.data for each in h])

                    # zero accumulated gradients
                    self.zero_grad()

                    # get the output from the model
                    output, h = self(x, h)

                    # calculate the loss and perform backprop
                    loss = loss_function(
                        output, y.contiguous().view(batch_size * seq_length)
                    )
                    loss.backward()

                    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                    nn.utils.clip_grad_norm_(self.parameters(), clip)
                    adam.step()

                    tepoch.set_postfix(loss=loss.item(), val_loss=0)

                    if counter == num_batches:
                        # Get validation loss
                        val_h = self.init_hidden(batch_size)
                        val_losses = []
                        self.eval()
                        for x_test, y_test in datagen.generate_batches(
                            batch_size, seq_length, start_idx=split_idx
                        ):
                            val_h = tuple([each.data for each in val_h])
                            output, val_h = self(x_test, val_h)
                            val_loss = loss_function(
                                output,
                                y_test.contiguous().view(
                                    batch_size * seq_length
                                ),
                            )
                            val_losses.append(val_loss.item())

                        # Get mean of the losses
                        tepoch.set_postfix(
                            loss=loss.item(),
                            val_loss=round(
                                sum(val_losses) / len(val_losses), 2
                            ),
                        )

                        # Save Model
                        if save_model:
                            checkpoint = {
                                "n_hidden": self.hidden_size,
                                "n_layers": self.num_layers,
                                "state_dict": self.state_dict(),
                                "tokens": datagen.unique_chars,
                            }
                            with open(
                                f"{model_dir}/epoch_{epoch}.net", "wb"
                            ) as f:
                                torch.save(checkpoint, f)
                        self.train()

    def predict_next_char(
        self,
        datagen: SequentialDataGenerator,
        char: str,
        h: torch.tensor,
        temperature=1,
        top_k=5,
    ):
        """
        Given a character, predict the next character.
        Returns the predicted character and the hidden state.

        Args:
            datagen (SequentialDataGenerator): Data generator
            char (str): Character to generate on
            h (torch.tensor): Hidden state
            temperature (int, optional): Temperature. Defaults to 1.
            top_k (int, optional): Top K. Defaults to 5.

        Returns:
            str: predicted character
        """

        # tensor inputs
        x = torch.tensor([datagen.char_to_ix[char]]).view(1, -1).to(device)
        x = datagen.one_hot_encode(x)

        # detach hidden state from history
        h = tuple([each.data for each in h])
        # get the output of the model
        out, h = self(x, h)

        # scale output by a temperature
        out /= temperature

        # get the character probabilities
        # apply softmax to get p probabilities for the likely next character giving x
        p = torch.softmax(out, dim=1).data
        if device == "cuda":
            p = p.cpu()  # move to cpu

        # get top characters
        # considering the k most probable characters with topk method
        if top_k is None:
            top_ch = torch.arange(len(datagen.unique_chars))

        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.squeeze()

        # select the likely next character with some element of randomness
        p = p.squeeze()
        char = np.random.choice(top_ch, p=(p / p.sum()).numpy())

        # return the encoded value of the predicted char and the hidden state
        return datagen.ix_to_char[char], h

    def sample(
        self,
        datagen: SequentialDataGenerator,
        sample_len: int,
        primer="Il",
        temperature=1.0,
        top_k=None,
    ) -> str:
        """Sample from the model

        Args:
            datagen (SequentialDataGenerator): Data generator
            sample_len (int): Sample length to generate
            primer (str): Primer to start with. Defaults to "Il".
            temperature (float): Temperature for sampling. Defaults to 1.0.
            top_k (int): Top K for sampling. Defaults to None.

        Returns:
            str: Model Predictions
        """
        self.to(device=device)
        self.eval()  # eval mode

        # First off, run through the primer
        chars = [ch for ch in primer]
        h = self.init_hidden(1)
        for ch in primer:
            char, h = self.predict_next_char(
                datagen, ch, h, top_k=top_k, temperature=temperature
            )

        chars.append(char)

        # Now pass in the previous character and get a new one
        for __ in range(sample_len):
            char, h = self.predict_next_char(
                datagen, chars[-1], h, top_k=top_k, temperature=temperature
            )
            chars.append(char)

        return "".join(chars)

    def load_model(self, model_dir: str) -> None:
        """Load model params

        Args:
            model_dir (str): Saved model file path
        """
        with open(model_dir, "rb") as f:
            checkpoint = torch.load(f)
        self.load_state_dict(checkpoint["state_dict"])
