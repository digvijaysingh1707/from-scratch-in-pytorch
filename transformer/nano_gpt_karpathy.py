# Karpathy = Chad
import os
import shutil
from datetime import datetime
from time import time

import torch
from torch import nn

emb_size = 256
block_size = 32
head_size = 128
batch_size = 512


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using Device: {device}")


class DataGenerator:
    def __init__(self, file_path: str, train_split=0.8) -> None:
        """_summary_

        Args:
            file_path (str): _description_
            train_split (float): _description_
        """
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
        return torch.tensor(encoded, device=device)

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
            device=device,
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
    def __init__(self, scaling_factor=4) -> None:
        super().__init__()
        self.feed_foward = nn.Sequential(
            nn.Linear(emb_size, emb_size * scaling_factor),
            nn.ReLU(),
            nn.Linear(emb_size * scaling_factor, emb_size),
        )

    def forward(self, X):
        """_summary_

        Args:
            X (_type_): Should be the output of MHA. Output shape: B, T, head_size

        Returns:
            torch.tensor: Output shape: B, T, emb_size
        """
        return self.feed_foward(X)  # B, T, emb_size


class ScaledAttention(nn.Module):
    def __init__(self, single_head_size) -> None:
        super().__init__()
        self.single_head_size = single_head_size
        self.l_key = nn.Linear(emb_size, single_head_size)
        self.l_query = nn.Linear(emb_size, single_head_size)
        self.l_value = nn.Linear(emb_size, single_head_size)

    def forward(self, X) -> torch.tensor:
        """Forward Function

        Args:
            X (torch.tensor): X should be the output of sem_emb + pos_emb of shape B, T, emb_size

        Returns:
            torch.tensor: _description_
        """
        Q = self.l_query(X)  # B, T, single_head_size
        K = self.l_key(X)  # B, T, single_head_size
        V = self.l_value(X)  # B, T, single_head_size
        # Produce weights
        wei = Q @ K.transpose(-1, -2)  # B, T, T
        tril = torch.tril(torch.ones(block_size, block_size))
        tril = tril.to(device)
        masked_wei = wei.masked_fill(tril == 0, float("-inf")) / (
            self.single_head_size**0.5
        )
        soft_wei = masked_wei.softmax(-1)  # B, T, T

        out = soft_wei @ V  # B, T, single_head_size
        return out
        # return self.ff(out) # B, T, emb_size


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attention_blocks = nn.ModuleList(
            [ScaledAttention(head_size // n_heads) for i in range(n_heads)]
        )
        self.proj_layer = nn.Linear(head_size, emb_size)

    def forward(self, X) -> torch.tensor:
        out = torch.cat(
            [self.attention_blocks[ix](X) for ix in range(self.n_heads)], -1
        )
        return self.proj_layer(out)  # 4, 8, 16 -> 4, 8, 32


class AttentionBlock(nn.Module):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.mha = MultiHeadedAttention(n_heads)
        self.ff = FeedFowardLayer()
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, X: torch.tensor):
        """_summary_

        Args:
            X (torch.tensor): Should be input emb (sem_emb + pos_emb)
        """
        X = self.mha(self.layer_norm(X)) + X
        out = self.ff(self.layer_norm(X)) + X
        return out


class GPT(nn.Module):
    def __init__(
        self, num_blocks, n_heads, data_generator: DataGenerator
    ) -> None:
        super().__init__()
        self.data_generator = data_generator
        self.semantic_embedding_table = nn.Embedding(
            self.data_generator.vocab_size, emb_size
        )
        self.positional_emb_table = nn.Embedding(block_size, emb_size)
        self.attention_layers = nn.Sequential(
            *[AttentionBlock(n_heads) for i in range(num_blocks)]
        )
        self.linear_layer = nn.Linear(emb_size, self.data_generator.vocab_size)

    def forward(self, X):
        sem_emb = self.semantic_embedding_table(X)  # B, T, emb_size
        # TODO: Check if position start from 0 or 1
        pos_emb = self.positional_emb_table(
            torch.arange(block_size, device=device)
        )  # T, emb_size
        # return sem_emb + pos_emb
        att_out = self.attention_layers(sem_emb + pos_emb)  # B, T, emb_size
        return self.linear_layer(att_out)  # B, T, vocab_size

    def train(self, num_epochs: int, checkpoint_itvl: int, save_model=True):
        if save_model:
            # Ensure checkpoints directory exists
            model_name = "nanoGPT"
            curr_time = datetime.now().strftime("%m_%d_%Y_T_%H_%M_%S")
            model_name += f"_{curr_time}"
            model_dir = f"saved_models/{model_name}"

            if os.path.isdir(model_dir):
                # to clear the existing
                shutil.rmtree(model_dir)
            os.makedirs(model_dir)

        opt = torch.optim.AdamW(self.parameters())
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(1, num_epochs + 1):
            start_time = time()
            X, Y = self.data_generator.generate_batch(
                "train", batch_size, block_size
            )
            out = self.forward(X)
            logits = out.softmax(-1)
            loss = loss_func(
                logits.view(
                    batch_size * block_size, self.data_generator.vocab_size
                ),
                Y.flatten(),
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % checkpoint_itvl == 0:
                total_time = time() - start_time
                print(
                    f"Epoch: {epoch}, Loss: {loss.item()}; Time taken: {total_time}"
                )

                # Save model
                with open(f"{model_dir}/epoch_{epoch}.net", "wb") as f:
                    torch.save(self.state_dict(), f)


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_gen = DataGenerator(f"{dir_path}/shakespear.txt")
    gpt = GPT(6, 4, data_gen)
    gpt.to(device)
    gpt.train(num_epochs=2, checkpoint_itvl=2)
