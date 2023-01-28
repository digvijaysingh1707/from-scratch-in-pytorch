# Karpathy = Chad
import os
import shutil
from datetime import datetime
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_DROPOUT_RATE = 0.4
MODEL_NAME = "nanoGPT"


class DataGenerator:
    def __init__(self, file_path: str, device: str, train_split=0.8) -> None:
        """Data Generator Class

        Args:
            file_path (str): File Path
            device (str): Device to Initialize tensors on
            train_split (float): Train vs Val Split
        """
        self.device = device
        # Data loading
        with open(file_path, "r") as f:
            self.data = f.read()

        # defining vocab
        vocab = sorted(list(set(self.data)))
        self.vocab_size = len(vocab)

        # lookup dicts
        self.ix_to_char = {i: ch for i, ch in enumerate(vocab)}
        self.char_to_ix = {ch: i for i, ch in enumerate(vocab)}

        # encode data into train & eval batches
        self.data_encoded = self.encode(self.data)
        split_idx = int(train_split * len(self.data_encoded))
        self.train_data = self.data_encoded[:split_idx]
        self.eval_data = self.data_encoded[split_idx:]

    def encode(self, input_str: str) -> torch.tensor:
        encoded = []
        for ch in input_str:
            encoded.append(self.char_to_ix[ch])
        return torch.tensor(encoded, device=self.device)

    def decode(self, input_ints: torch.tensor) -> str:
        decoded = ""
        for ix in input_ints:
            decoded += self.ix_to_char[ix.item()]
        return decoded

    def generate_batch(self, split_type, batch_size, block_size):
        if split_type == "train":
            batch_to_generate_from = self.train_data
        else:
            batch_to_generate_from = self.eval_data

        # generate random starting points
        ixes = torch.randint(
            0,
            len(batch_to_generate_from) - block_size,
            (batch_size,),
            device=self.device,
        )

        # extend the generated starting points
        X = torch.stack(
            [batch_to_generate_from[ix : ix + block_size] for ix in ixes],
        )  # B, T
        Y = torch.stack(
            [
                batch_to_generate_from[ix + 1 : ix + block_size + 1]
                for ix in ixes
            ]
        )  # B, T
        return X, Y


class FeedFowardLayer(nn.Module):
    def __init__(
        self, emb_size: int, dropout=DEFAULT_DROPOUT_RATE, scaling_factor=4
    ) -> None:
        """FeedForward Layer that comes after Multi-Head Attention.
            Scales the output of MHA from head_size to emb_size.

        Args:
            emb_size (int): Embedding Size
            dropout (float, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
            scaling_factor (int, optional): Scaling Factor among the two Linear Layers.
                Defaults to 4.
        """
        super().__init__()
        self.emb_size = emb_size
        self.feed_foward = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size * scaling_factor),
            nn.ReLU(),
            nn.Linear(self.emb_size * scaling_factor, self.emb_size),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.tensor) -> torch.tensor:
        """_summary_

        Args:
            X (torch.tensor): Should be the output of MHA. Output shape: B, T, head_size

        Returns:
            torch.tensor: Output shape: B, T, emb_size
        """
        return self.feed_foward(X)  # B, T, emb_size


class ScaledAttention(nn.Module):
    def __init__(
        self,
        head_size: int,
        emb_size: int,
        block_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """ScaledAttention Block.
            Generates K, Q, V vectors of size: head_size to calculate weighted attention (Q.K)
                and returns (weighted_attention).V

        Args:
            head_size (int): Single Head Size.
                Generally MHA head_size / n_heads
            emb_size (int): Embedding Size.
            block_size (int): Block Size.
            dropout (int, DEFAULT_DROPOUT_RATE): Dropout Rate.
                Defaults to DEFAULT_DROPOUT_RATE
        """
        super().__init__()
        self.head_size = head_size
        self.l_key = nn.Linear(emb_size, head_size, bias=False)
        self.l_query = nn.Linear(emb_size, head_size, bias=False)
        self.l_value = nn.Linear(emb_size, head_size, bias=False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X) -> torch.tensor:
        """Forward Function

        Args:
            X (torch.tensor): X should be the output of sem_emb + pos_emb of shape B, T, emb_size

        Returns:
            torch.tensor: Output of shape B, T, head_size
        """
        B, T, C = X.shape
        Q = self.l_query(X)  # B, T, head_size
        K = self.l_key(X)  # B, T, head_size
        V = self.l_value(X)  # B, T, head_size
        # Produce weights
        wei = Q @ K.transpose(-2, -1) * C**-0.5  # B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = wei.softmax(-1)  # B, T, T
        wei = self.dropout(wei)
        out = wei @ V  # B, T, head_size
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        head_size: int,
        n_heads: int,
        emb_size: int,
        block_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """Multi-Headed Attention Class

        Args:
            head_size (int): Output head size for MHA.
                Head size of individual SelfAttention block = head_size / n_heads
            n_heads (int): Number of Heads of Self Attention.
            emb_size (int): Embedding Size
            block_size (int): Block Size
            dropout (int, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
        """
        super().__init__()
        self.n_heads = n_heads
        self.attention_blocks = nn.ModuleList(
            [
                ScaledAttention(
                    head_size=head_size // self.n_heads,
                    emb_size=emb_size,
                    block_size=block_size,
                )
                for _ in range(n_heads)
            ]
        )
        self.proj_layer = nn.Linear(head_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward Function for Multi-Headed Self-Attention

        Args:
            X (torch.tensor): Input to MHA block. Of Shape B, T, emb_size

        Returns:
            torch.tensor: _description_
        """
        out = torch.cat([block(X) for block in self.attention_blocks], -1)
        # B, T, head_size -> B, T, emb_size
        out = self.dropout(self.proj_layer(out))
        return out


class AttentionBlock(nn.Module):
    def __init__(
        self,
        head_size: int,
        n_heads: int,
        emb_size: int,
        block_size: int,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """Attention Block consisting of MHA, FeedForward, Residual connections, and LayerNorm

        Args:
            head_size (int): Output head size for MHA.
                Head size of individual SelfAttention block = head_size / n_heads
            n_heads (int): Number of Heads of Self Attention.
            emb_size (int): Embedding Size
            block_size (int): Block Size
            dropout (int, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
        """
        super().__init__()
        self.mha = MultiHeadedAttention(
            head_size=head_size,
            n_heads=n_heads,
            emb_size=emb_size,
            block_size=block_size,
            dropout=dropout,
        )
        self.ff = FeedFowardLayer(emb_size=emb_size, dropout=dropout)
        # Layer Norm Layers
        self.ln1 = nn.LayerNorm(emb_size)
        self.ln2 = nn.LayerNorm(emb_size)

    def forward(self, X: torch.tensor) -> torch.tensor:
        """Forward Function for Attention Block

        Args:
            X (torch.tensor): Should be the output of (sem_emb + pos_emb).
                Of shape: B, T, emb_size

        Returns:
            torch.tensor: _description_
        """
        X = self.mha(self.ln1(X)) + X
        X = self.ff(self.ln2(X)) + X
        return X


class GPT(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        head_size: int,
        n_heads: int,
        emb_size: int,
        block_size: int,
        data_generator: DataGenerator,
        device: str,
        dropout=DEFAULT_DROPOUT_RATE,
    ) -> None:
        """GPT Module

        Args:
            num_blocks (int): Num blocks of the attention block.
            head_size (int): Output head size for MHA.
                Head size of individual SelfAttention block = head_size / n_heads
            n_heads (int): Number of Heads of Self Attention.
            emb_size (int): Embedding Size
            block_size (int): Block Size
            data_generator (DataGenerator): Data Generator
            device (str): Device to store params GPT params on.
            dropout (int, optional): Dropout Rate. Defaults to DEFAULT_DROPOUT_RATE.
        """
        super().__init__()
        self.device = device
        self.block_size = block_size
        self.data_generator = data_generator
        self.semantic_embedding_table = nn.Embedding(
            self.data_generator.vocab_size, emb_size
        )
        self.positional_emb_table = nn.Embedding(block_size, emb_size)
        self.attention_layers = nn.Sequential(
            *[
                AttentionBlock(
                    head_size=head_size,
                    n_heads=n_heads,
                    emb_size=emb_size,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_layer = nn.Linear(emb_size, self.data_generator.vocab_size)
        # final layer norm
        self.ln_f = nn.LayerNorm(emb_size)

    def forward(self, X: torch.tensor) -> torch.tensor:
        sem_emb = self.semantic_embedding_table(X)  # B, T, emb_size
        # TODO: Check if position start from 0 or 1
        pos_emb = self.positional_emb_table(
            torch.arange(X.shape[1], device=self.device)
        )  # T, emb_size
        # return sem_emb + pos_emb
        out = self.ln_f(
            self.attention_layers(sem_emb + pos_emb)
        )  # B, T, emb_size
        return self.linear_layer(out)  # B, T, vocab_size

    def train(
        self,
        batch_size: int,
        num_epochs: int,
        checkpoint_itvl: int,
        lr=3e-4,
        save_model=False,
    ):
        """Train nanoGPT

        Args:
            batch_size (int): Batch Size.
            num_epochs (int): Num Epochs to Train on
            checkpoint_itvl (int): Checkpoint Interval to calculate eval_loss
            lr (float): Learning Rate. Defaults to 3e-4.
            save_model (bool): Whether to save model or not. Defaults to True.
        """
        if save_model:
            # Ensure checkpoints directory exists
            curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
            model_name = f"{MODEL_NAME}_{curr_time}"
            model_dir = f"saved_models/{model_name}"

            if os.path.isdir(model_dir):
                # to clear the existing
                shutil.rmtree(model_dir)
            os.makedirs(model_dir)

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        # loss_func = nn.CrossEntropyLoss()

        print(
            f"Training Starting. Total Params: {sum(p.numel() for p in self.parameters()) / 1e6}M"
        )

        for epoch in range(1, num_epochs + 1):
            X, Y = self.data_generator.generate_batch(
                "train", batch_size, self.block_size
            )
            logits = self.forward(X)
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.view(B * T, C),
                Y.view(B * T),
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            print(f"Epoch: {epoch}; Loss: {loss.item()}")
            opt.step()

            if epoch % checkpoint_itvl == 0:
                if save_model:
                    # Save model
                    with open(f"{model_dir}/epoch_{epoch}.pt", "wb") as f:
                        torch.save(self.state_dict(), f)

    def load_model(self, model_path: str) -> None:
        """Load saved model

        Args:
            model_path (str): Model path
        """
        try:
            self.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            print("Saved model loaded successfully!")
        except Exception as err:
            print(err)

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = logits.softmax(dim=-1)  # (B, C)
            # probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using Device: {device}")

    # Hyper Params
    emb_size = 256
    block_size = 32
    head_size = 128
    batch_size = 512
    dropout = 0.4
    num_blocks = 6
    n_heads = 4
    n_epochs = 1000
    checkpoint_itvl = 100

    data_gen = DataGenerator(f"{dir_path}/shakespear.txt", device=device)
    gpt = GPT(
        num_blocks=num_blocks,
        head_size=head_size,
        n_heads=n_heads,
        emb_size=emb_size,
        block_size=block_size,
        data_generator=data_gen,
        device=device,
        dropout=dropout,
    )
    gpt.to(device)
    gpt.train(
        batch_size=batch_size,
        num_epochs=n_epochs,
        checkpoint_itvl=checkpoint_itvl,
    )
