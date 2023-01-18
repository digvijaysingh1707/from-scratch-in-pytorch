head_size = 16

class SingleHead(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.l_key = nn.Linear(emb_size, head_size)
        self.l_query = nn.Linear(emb_size, head_size)
        self.l_value = nn.Linear(emb_size, head_size)
        self.ff = nn.Linear(head_size, emb_size)

    def forward(self, X) -> torch.tensor:
        """Forward Function

        Args:
            X (torch.tensor): X should be the output of sem_emb + pos_emb of shape B, T, emb_size

        Returns:
            torch.tensor: _description_
        """
        Q = self.l_query(X) # B, T, head_size
        K = self.l_key(X) # B, T, head_size
        V = self.l_value(X) # B, T, head_size
        # Produce weights
        wei = Q @ K.transpose(-1, -2) # B, T, T
        tril = torch.tril(torch.ones(block_size, block_size))
        masked_wei = wei.masked_fill(tril==0, float('-inf')) / (head_size ** 0.5)
        soft_wei = masked_wei.softmax(-1) # B, T, T

        out = soft_wei @ V # B, T, head_size
        return out
        # return self.ff(out) # B, T, emb_size
        
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.attention_blocks = nn.ModuleList([SingleHead(head_size//n_heads) for i in range(n_heads)])
        self.proj_layer = nn.Linear(head_size, emb_size)
    def forward(self, X) -> torch.tensor:
        out = torch.cat([self.attention_blocks[ix](X) for ix in range(self.n_heads)], -1)
        return self.proj_layer(out)  # 4, 8, 16 -> 4, 8, 32

class FeedFowardLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feed_foward = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.ReLU(),
            nn.Linear(emb_size * 4, emb_size)
        )

    def forward(self, X):
        """_summary_

        Args:
            X (_type_): Should be the output of MHA. Output shape: B, T, head_size

        Returns:
            torch.tensor: Output shape: B, T, emb_size
        """
        return self.feed_foward(X) # B, T, emb_size

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
     