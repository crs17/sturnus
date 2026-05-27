import torch

# In order to keep track of the dimensions of each array, 
# we use the following one-letter abbreviations:
# I self attention block input size
# O self attention block output size
# B batch size
# C context length

# In coments after each array, we write the dimensions of the array.

class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        head_count: int,
        dropout_rate: float,
        qkv_bias: bool = False
    ):
        super().__init__()

        assert (d_out % head_count) == 0, "d_out must be divisible by head_count"
        self.d_out = d_out
        self.head_count = head_count
        self.d_out_per_head = d_out // head_count

        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        
        self.out_projection = torch.nn.Linear(d_out, d_out)  # [O, O]
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1).bool()
        )  # [C, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, count_tokens, d_in = x.shape

        q = self.W_Q(x)  # [batch_size, count_tokens, d_out]
        k = self.W_K(x)  # [batch_size, count_tokens, d_out]
        v = self.W_V(x)  # [batch_size, count_tokens, d_out]

        head_view = (batch_size, count_tokens, self.head_count, self.d_out_per_head)
        q = q.view(head_view)
        k = k.view(head_view)
        v = v.view(head_view)

        q = q.transpose(1, 2)  # [batch_size, head_count, count_tokens, d_out_per_head]
        k = k.transpose(1, 2)  # [batch_size, head_count, count_tokens, d_out_per_head]
        v = v.transpose(1, 2)  # [batch_size, head_count, count_tokens, d_out_per_head]

        attention_scores = q @ k.transpose(2, 3)  # [batch_size, head_count, count_tokens, d_out_per_head] @ [batch_size, head_count, d_out_per_head, count_tokens] = [batch_size, head_count, count_tokens, count_tokens]
        # print('attention_scores', attention_scores.shape)

        mask = self.mask[:count_tokens, :count_tokens]  # [count_tokens, count_tokens]
        attention_scores.masked_fill_(mask, float('-inf'))
        # print('attention_scores', attention_scores.shape)

        attention_weights = torch.softmax(attention_scores / self.d_out_per_head ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # print('attention_weights', attention_weights.shape)

        context_vectors = (attention_weights @ v).transpose(1, 2)  # [batch_size, head_count, count_tokens, count_tokens] @ [batch_size, head_count, count_tokens, d_out_per_head] = [batch_size, head_count, count_tokens, d_out_per_head]
        # print('context_vectors', context_vectors.shape)

        context_vectors = context_vectors.contiguous().view(batch_size, count_tokens, self.d_out)  # [batch_size, head_count, count_tokens, d_out_per_head] = [batch_size, count_tokens, d_out]
        # print('context_vectors (after contiguous)', context_vectors.shape)

        context_vectors = self.out_projection(context_vectors)  # [batch_size, count_tokens, d_out]
        # print('context_vectors (after out_projection)', context_vectors.shape)

        return context_vectors
        


class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(config['embed_dim'], 4 * config['embed_dim']),
            torch.nn.GELU(approximate='tanh'),
            torch.nn.Linear(4 * config['embed_dim'], config['embed_dim']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ShortcutExampleNN(torch.nn.Module):
    def __init__(self, count_layers: int, use_shortcut: bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        1 if i == 0 else 5,
                        1 if i == count_layers -1 else 5
                    ),
                    torch.nn.GELU(approximate='tanh')
                )
                for i in range(count_layers)
            ]            
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            if self.use_shortcut:
                x = x + layer(x)
            else:
                x = layer(x)
        return x
    

class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=config['embed_dim'],
            d_out=config['embed_dim'],
            context_length=config['block_size'],
            head_count=config['count_heads'],   
            dropout_rate=config['dropout'],
            qkv_bias=config['qkv_bias']
        )
        self.ln1 = LayerNorm(config['embed_dim'])
        self.ffn = FeedForward(config)
        self.ln2 = LayerNorm(config['embed_dim'])
        self.dropout = torch.nn.Dropout(config['dropout'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        # Identifies and analyses relationships between tokens
        shortcut = x
        x = self.ln1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        # Feed-forward block
        # Modifies tokens individually
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x    
        