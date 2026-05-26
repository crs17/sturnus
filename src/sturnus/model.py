import torch


class SelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout_rate: float,
        qkv_bias: bool = False
    ):
        super().__init__()
        self.d_out = d_out

        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [d_out, d_in]
        self.W_K = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [d_out, d_in]
        self.W_V = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [d_out, d_in]

        # print('W_Q', self.W_Q.weight.shape)
        # print('W_K', self.W_K.weight.shape)
        # print('W_V', self.W_V.weight.shape)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
            diagonal=1).bool()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_K(x)  # [batch_size, num_tokens, d_out]
        queries = self.W_Q(x)  # [batch_size, num_tokens, d_out]
        values = self.W_V(x)  # [batch_size, num_tokens, d_out]        
        
        # print('keys', keys.shape)
        # print('queries', queries.shape)
        # print('values', values.shape)
        attention_scores = queries @ keys.transpose(1, 2)  # [batch_size, num_tokens, d_out] @ [batch_size, d_out, num_tokens] = [batch_size, num_tokens, num_tokens]
        # print('attention_scores', attention_scores.shape)
        # print('mask', self.mask.shape)
        attention_scores.masked_fill_(
            self.mask[:num_tokens, :num_tokens],
            float('-inf')
        )  # [batch_size, num_tokens, num_tokens]
        # print('attention_scores', attention_scores.shape)
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )  # [batch_size, num_tokens, num_tokens]
        attention_weights = self.dropout(attention_weights)
        # print('attention_weights', attention_weights.shape)
        context_vectors = attention_weights @ values  # [batch_size, num_tokens, num_tokens] @ [batch_size, num_tokens, d_out] = [batch_size, num_tokens, d_out]
        # print('context_vectors', context_vectors.shape)
        return context_vectors