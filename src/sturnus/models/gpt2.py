import torch

# This file contains the implementation of the GPT model. I have tried to supply
# plenty of comments to make the code easier to understand.

# In order to keep track of the dimensions of each array, 
# we use the following one-letter abbreviations:
# I   self attention block input size
# O   self attention block output size
# B   batch size
# C   context length
# H   Head count
# P   output size per head


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention block. This is the main building block of the GPT model.
    It is used to compute the attention between the tokens in the context.
    """
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

        # This class implements <head_count> parallel attention heads. 
        # The output dimension <d_out> is divided equally among the heads.
        # We must therefore first ensure that the output size is divisible by
        # the number of heads.
        assert (d_out % head_count) == 0, "d_out must be divisible by head_count"
        self.d_out = d_out
        self.head_count = head_count
        self.d_out_per_head = d_out // head_count

        # Next we instatiate the trainable parameters for the query, key
        # and value matrices.
        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        
        # The final output projection layer and dropout layer are also instantiated.
        self.out_projection = torch.nn.Linear(d_out, d_out)  # [O, O]
        self.dropout = torch.nn.Dropout(dropout_rate)

        # We also register the mask buffer. This is a triangular matrix of ones
        # above the diagonal and zeros at and below the diagonal.
        # This is used to prevent the model from attending to tokens that are
        # later in the context.
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1).bool()
        )  # [C, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        batch_size, count_tokens, d_in = x.shape  # [B, C, I]

        # Here we project the input tokens into the query, key and value spaces.
        # Note that the resulting query, key and value matrices are shared across the
        # heads with each head using separate slices.
        # [B, C, I] @ [O, I].T = [B, C, O]
        q = self.W_Q(x)  # [B, C, O]
        k = self.W_K(x)  # [B, C, O]
        v = self.W_V(x)  # [B, C, O]

        # Next we reshape the query, key and value matrices to separate slices
        # for each head. Note that the last dimension which was of size <O> is 
        # now split into <H> splits with size <P>.
        # [B, C, H, P]
        head_view = (batch_size, count_tokens, self.head_count, self.d_out_per_head)
        q = q.view(head_view)  # [B, C, H, P]
        k = k.view(head_view)  # [B, C, H, P]
        v = v.view(head_view)  # [B, C, H, P]

        # Here we transpose the second and third dimensions, 
        # swapping the context length for the head count. This enables us 
        # to perform the matrix multiplication per head in parallel.
        q = q.transpose(1, 2)  # [B, H, C, P]
        k = k.transpose(1, 2)  # [B, H, C, P]
        v = v.transpose(1, 2)  # [B, H, C, P]

        # Now we can compute the attention scores for each head in parallel.
        # [B, H, C, P] @ [B, H, C, P].T(2,3) = [B, H, C, C]
        attention_scores = q @ k.transpose(2, 3) 

        # Next we apply the mask to the attention scores. This is used to prevent
        # the model from attending to tokens that are later in the context.
        # We fill the masked positions with negative infinity. The softmax function
        # will then set the weights of the masked positions to zero.
        mask = self.mask[:count_tokens, :count_tokens]  # [C, C]
        attention_scores.masked_fill_(mask, float('-inf'))

        # Next we apply the softmax function to the attention scores. This
        # normalizes the scores to a sum of 1. The division by the
        # square root of the output size per head helps with numerical stability.
        attention_weights = torch.softmax(
            attention_scores / self.d_out_per_head ** 0.5, dim=-1
        )
        # Finally we apply the dropout layer to the attention weights.
        attention_weights = self.dropout(attention_weights)

        # Here we compute the context vectors by coing a matrix multiplication between
        # th attention weights and the value vectors. The context vectors are then reshaped
        # to the "per head" shape.
        # ([B, H, C, C] @ [B, H, C, P]).T(1,2) = [B, C, H, P]
        context_vectors = (attention_weights @ v).transpose(1, 2) 

        # The "per head" dimensions are now concatenated back into the original shape.
        # [B, C, H, P].contiguous().view(B, C, O) = [B, C, O]
        context_vectors = context_vectors.contiguous().view(batch_size, count_tokens, self.d_out)

        # Finally we appy the additional output projection.
        # [B, C, O] @ [O, O] = [B, C, O]
        context_vectors = self.out_projection(context_vectors)

        return context_vectors
        


class LayerNorm(torch.nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = torch.nn.Parameter(torch.ones(emb_dim))
        self.shift = torch.nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
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
        # Modifies tokens individually - no information is shared between tokens
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x    


class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_emb = torch.nn.Embedding(config['block_size'], config['embed_dim'])

        self.drop_emb = torch.nn.Dropout(config['dropout'])

        self.trf_blocks = torch.nn.Sequential(
            *[TransformerBlock(config) for _ in range(config['count_blocks'])]
        )

        self.final_norm = LayerNorm(config['embed_dim'])
        self.out_head = torch.nn.Linear(
            config['embed_dim'], config['vocab_size'], bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)

        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits