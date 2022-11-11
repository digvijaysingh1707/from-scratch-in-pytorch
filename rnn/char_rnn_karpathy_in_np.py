import os
import shutil
from datetime import datetime
import numpy as np


class MiniRNN:
    def __init__(
        self,
        n_hidden_neurons: int,
        train_data: str,
        seq_length=25,
        model_name="miniRNN",
    ) -> None:
        """Initialize Mini RNN for character level generation

        Args:
            n_hidden_neurons (int): Number of Neurons in the hidden layer
            vocab_size (int): Number of unique characters for the RNN to predict
            seq_length (int): number of steps to unroll the RNN for.
                Defaults to 25.
            model_name (str): Model Name. Defaults to miniRNN
        """
        self.data = train_data
        chars = list(set(train_data))
        self.data_size, self.vocab_size = len(train_data), len(chars)

        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        self.seq_length = seq_length
        self.n_hidden_neurons = n_hidden_neurons
        self.model_name = model_name

        # Initialize weights
        self.model_weights = {
            # input -> hidden_state
            "xh": np.random.randn(n_hidden_neurons, self.vocab_size) * 0.01,
            # prev_hidden_state -> next_hidden_state
            "hh": np.random.randn(n_hidden_neurons, n_hidden_neurons) * 0.01,
            # hidden_state -> output
            "hy": np.random.randn(self.vocab_size, n_hidden_neurons) * 0.01,
        }

        # Initialize Biases
        self.model_biases = {
            # input to hidden
            "xh": np.zeros((n_hidden_neurons, 1)),
            # hidden to output
            "hy": np.zeros((self.vocab_size, 1)),
        }

    def calc_gradients(self, inputs: list, targets: list, h_prev):
        """_summary_

        Args:
            inputs (list): _description_
            targets (list): _description_
            h_prev (_type_): _description_
        """
        inputs_x, hidden_states_h, preds_y, probabs_p = {}, {}, {}, {}

        # Last hidden state vars is hprev
        hidden_states_h[-1] = np.copy(h_prev)
        loss = 0

        # forward pass
        for t in range(len(inputs)):
            # one-hot encoding of input
            # create a list of zeros of size: vocab_size
            inputs_x[t] = np.zeros((self.vocab_size, 1))
            # set the index of the char to 1
            inputs_x[t][inputs[t]] = 1

            # calculate the hidden state
            hidden_states_h[t] = np.tanh(
                np.dot(self.model_weights["xh"], inputs_x[t])
                + np.dot(self.model_weights["hh"], hidden_states_h[t - 1])
                + self.model_biases["xh"]
            )

            # calculate the next output
            # unnormalized log probabilities for next chars
            preds_y[t] = (
                np.dot(self.model_weights["hy"], hidden_states_h[t])
                + self.model_biases["hy"]
            )

            # normalized probabilities for next chars
            probabs_p[t] = np.exp(preds_y[t]) / np.sum(np.exp(preds_y[t]))

            # Calculate loss: softmax (cross-entropy loss)
            loss += -np.log(probabs_p[t][targets[t], 0])

        # backward pass: compute gradients going backwards
        # set gradient matrices to be the same size as weights
        dWxh, dWhh, dWhy = (
            np.zeros_like(self.model_weights["xh"]),
            np.zeros_like(self.model_weights["hh"]),
            np.zeros_like(self.model_weights["hy"]),
        )
        # set bias matrices
        dbh, dby = (
            np.zeros_like(self.model_biases["xh"]),
            np.zeros_like(self.model_biases["hy"]),
        )
        # set a dummy dh_next var
        dhnext = np.zeros_like(hidden_states_h[0])

        # Go back every step and calculate grads
        for t in reversed(range(len(inputs))):
            # grad of y
            dy = np.copy(probabs_p[t])
            # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dy[targets[t]] -= 1
            dWhy = np.dot(dy, hidden_states_h[t].T)
            dby += dy
            dh = np.dot(self.model_weights["hy"].T, dy) + dhnext
            dhraw = (
                1 - hidden_states_h[t] * hidden_states_h[t]
            ) * dh  # grad of tanh
            dbh += dhraw
            dWxh += np.dot(dhraw, inputs_x[t].T)
            dWhh += np.dot(dhraw, hidden_states_h[t - 1].T)
            dhnext = np.dot(self.model_weights["hh"].T, dhraw)

        # Clip gradients to mitigate exploding grad problem
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)

        return (
            loss,
            dWxh,
            dWhh,
            dWhy,
            dbh,
            dby,
            hidden_states_h[len(inputs) - 1],
        )

    def sample(self, h: np.array, seed_ix: int, n: int) -> list:
        """_summary_

        Args:
            h (np.array): _description_
            seed_ix (int): _description_
            n (int): _description_

        Returns:
            list: _description_
        """
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1

        ixes = []
        for _ in range(n):
            # calc hidden state based on x, ws & bh, and prev state 'h'
            h = np.tanh(
                np.dot(self.model_weights["xh"], x)
                + np.dot(self.model_weights["hh"], h)
                + self.model_biases["xh"]
            )

            # calc predictions through the updated hidden state
            y = np.dot(self.model_weights["hy"], h) + self.model_biases["hy"]

            # normalize probabilites
            p = np.exp(y) / np.sum(np.exp(y))

            # TODO: Understand this piece of code.
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def save_model_params(self, model_dir: str, model_iter: int) -> None:
        """Save Model Weights"""

        # Make Dir for the model iteration
        os.makedirs(f"{model_dir}/{model_iter}")

        # Save Model Weights
        for param_name, param in self.model_weights.items():
            with open(
                f"{model_dir}/{model_iter}/weights_{param_name}.npy", "wb"
            ) as f:
                np.save(f, param)

        # Save Model Biases
        for param_name, param in self.model_biases.items():
            with open(
                f"{model_dir}/{model_iter}/biases_{param_name}.npy", "wb"
            ) as f:
                np.save(f, param)

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
        for w_ in list(self.model_weights.keys()):
            with open(f"{model_dir}/{model_ckpt}/weights_{w_}.npy", "rb") as f:
                self.model_weights[w_] = np.load(f)

        # Load Biases
        for b_ in list(self.model_biases.keys()):
            with open(f"{model_dir}/{model_ckpt}/biases_{b_}.npy", "rb") as f:
                self.model_biases[b_] = np.load(f)

    def train(
        self,
        num_epochs: int,
        learning_rate=1e-1,
        checkpoint_range=1000,
        verbose=False,
        save_model=True,
    ) -> None:
        """Train Model

        Args:
            data (str): Data to train upon
            num_epochs (int): Number of epochs to train on
            learning_rate (float): Learning Rate of the model. Defaults to 1e-1.
            checkpoint_range (int): Checkpoint range. Defaults to 100.
            verbose (bool): Whether to print predictions at every checkpoint. Defaults to False.
            save_model_weights (bool): Whether to save model weights. Defaults to True.
        """

        print(f"Data size: {self.data_size}; Vocab Size: {self.vocab_size}")

        # Ensure checkpoints directory exists
        curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
        if self.model_name == "miniRNN":
            self.model_name += f"_{curr_time}"
        model_dir = f"saved_models/{self.model_name}"

        if os.path.isdir(model_dir):
            # to clear the existing
            shutil.rmtree(model_dir)
        os.makedirs(model_dir)

        p = 0
        # declare memory variables for Adagrad
        mWxh, mWhh, mWhy = (
            np.zeros_like(self.model_weights["xh"]),
            np.zeros_like(self.model_weights["hh"]),
            np.zeros_like(self.model_weights["hy"]),
        )
        mbh, mby = (
            np.zeros_like(self.model_biases["xh"]),
            np.zeros_like(self.model_biases["hy"]),
        )

        # loss at iteration 0
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length

        print(f"Starting Training. Save Model: {save_model}, ")
        try:
            for epoch in range(num_epochs):
                # prepare inputs (we're sweeping from left to right in steps seq_length long)
                if p + self.seq_length + 1 >= len(self.data) or epoch == 0:
                    hprev = np.zeros(
                        (self.n_hidden_neurons, 1)
                    )  # reset RNN memory
                    p = 0  # go from start of self.data
                inputs = [
                    self.char_to_ix[ch]
                    for ch in self.data[p : p + self.seq_length]
                ]
                targets = [
                    self.char_to_ix[ch]
                    for ch in self.data[p + 1 : p + self.seq_length + 1]
                ]

                # sample from the model now and then
                if epoch % checkpoint_range == 0 and verbose:
                    sample_ix = self.sample(hprev, inputs[0], 200)
                    txt = "".join(self.ix_to_char[ix] for ix in sample_ix)
                    print("-" * 30, f"\n {txt} \n", "-" * 30)

                # forward seq_length characters through the net and fetch gradient
                loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.calc_gradients(
                    inputs, targets, hprev
                )
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                if epoch % checkpoint_range == 0:
                    print(f"Iteration: {epoch}; Loss: {smooth_loss}")

                    # Save Model
                    if save_model:
                        self.save_model_params(model_dir, epoch)

                # perform parameter update with Adagrad
                for param, dparam, mem in zip(
                    [
                        self.model_weights["xh"],
                        self.model_weights["hh"],
                        self.model_weights["hy"],
                        self.model_biases["xh"],
                        self.model_biases["hy"],
                    ],
                    [dWxh, dWhh, dWhy, dbh, dby],
                    [mWxh, mWhh, mWhy, mbh, mby],
                ):
                    mem += dparam * dparam
                    param += (
                        -learning_rate * dparam / np.sqrt(mem + 1e-8)
                    )  # adagrad update

                p += self.seq_length  # move self.data pointer
            print("Training Complete.")

        except KeyboardInterrupt:
            print("Halting Training.")

        # Save final weights & biases
        if save_model:
            print("Saving last checkpoint...")
            self.save_model_params(model_dir, "final")

    def predict_sequence(self, input_char: str, generated_str_len=200):
        hprev = np.zeros((self.n_hidden_neurons, 1))  # reset RNN memory

        sample_ix = self.sample(
            hprev, self.char_to_ix[input_char], generated_str_len
        )
        txt = "".join(self.ix_to_char[ix] for ix in sample_ix)
        print("-" * 30, f"\n {txt} \n", "-" * 30)
