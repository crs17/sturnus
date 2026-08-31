from typing import Optional

import torch

from sturnus.models.gpt2 import (FeedForward, LayerNorm)

# In order to keep track of the dimensions of each array, 
# we use the following one-letter abbreviations:
# I   self attention block input size
# O   self attention block output size
# B   batch size
# C   cached context length
# H   Head count
# P   output size per head
# Q   query context length


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

        # Finally, we will register buffers for the K and V caches. We do not want them to be part
        #  of the model's statedict so we set persistent=Fale:
        self.register_buffer('K_cache', None, persistent=False)
        self.register_buffer('V_cache', None, persistent=False)
        self.position_current = 0

    def reset_cache(self):
        self.K_cache = None
        self.V_cache = None
        self.position_current = 0

    def forward(self, x: torch.Tensor, use_cache: bool=False) -> torch.Tensor:
        
        batch_size, count_tokens, d_in = x.shape  # [B, Q, I]

        # Here we project the input tokens into the query, key and value spaces.
        # Note that the resulting query, key and value matrices are shared across the
        # heads with each head using separate slices.
        # [B, Q, I] @ [O, I].T = [B, Q, O]
        q = self.W_Q(x)  # [B, Q, O]
        k = self.W_K(x)  # [B, Q, O]
        v = self.W_V(x)  # [B, Q, O]

        # Next we reshape the query, key and value matrices to separate slices
        # for each head. Note that the last dimension which was of size <O> is 
        # now split into <H> splits with size <P>.
        # [B, Q, H, P]
        head_view = (batch_size, count_tokens, self.head_count, self.d_out_per_head)
        q = q.view(head_view)  # [B, Q, H, P]
        k = k.view(head_view)  # [B, Q, H, P]
        v = v.view(head_view)  # [B, Q, H, P]

        # If we are using caching we can here recyckle the previously calculated k and v vectors
        if use_cache:
            if self.K_cache is None:
                self.K_cache = k
                self.V_cache = v
            else:
                self.K_cache = torch.concat((self.K_cache, k), dim=1)
                self.V_cache = torch.concat((self.V_cache, v), dim=1)
            
            k = self.K_cache  # [B, C, H, P]
            v = self.V_cache  # [B, C, H, P]

        # Here we transpose the second and third dimensions, 
        # swapping the context length for the head count. This enables us 
        # to perform the matrix multiplication per head in parallel.
        q = q.transpose(1, 2)  # [B, H, Q, P]
        k = k.transpose(1, 2)  # [B, H, C, P]
        v = v.transpose(1, 2)  # [B, H, C, P]

        # Now we can compute the attention scores for each head in parallel.
        # [B, H, Q, P] @ [B, H, C, P].T(2,3) = [B, H, Q, C]
        attention_scores = q @ k.transpose(2, 3) 

        # Next we apply the mask to the attention scores. This is used to prevent
        # the model from attending to tokens that are later in the context.
        # We fill the masked positions with negative infinity. The softmax function
        # will then set the weights of the masked positions to zero.
        count_tokens_q = q.shape[-2]
        count_tokens_k = k.shape[-2]

        mask = self.mask[self.position_current: self.position_current + count_tokens_q, :count_tokens_k]

        if use_cache:
            self.position_current += count_tokens_q

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

    def forward(self, x: torch.Tensor, use_cache: bool=False) -> torch.Tensor:
        # Attention block
        # Identifies and analyses relationships between tokens
        shortcut = x
        x = self.ln1(x)
        x = self.att(x, use_cache=use_cache)
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


class GPTKVCache(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_emb = torch.nn.Embedding(config['block_size'], config['embed_dim'])

        self.drop_emb = torch.nn.Dropout(config['dropout'])

        self.trf_blocks = torch.nn.ModuleList(
            [TransformerBlock(config) for _ in range(config['count_blocks'])]
        )

        self.final_norm = LayerNorm(config['embed_dim'])
        self.out_head = torch.nn.Linear(
            config['embed_dim'], config['vocab_size'], bias=False
        )

        self.position_current = 0

    def reset_cache(self):
        for block in self.trf_blocks:
            block.att.reset_cache()

        self.position_current = 0
        
    def forward(self, x: torch.Tensor, use_cache:bool=False) -> torch.Tensor:
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)

        idx = torch.arange(
            self.position_current,
            self.position_current + seq_len,
            device=x.device
        )
        pos_embeds = self.pos_emb(idx)

        if use_cache:
            self.position_current += seq_len

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            x = block(x, use_cache=use_cache)

        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits