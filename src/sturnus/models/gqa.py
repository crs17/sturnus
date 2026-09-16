import torch

# In order to keep track of the dimensions of each array, 
# we use the following one-letter abbreviations:
# I   self attention block input size
# O   self attention block output size
# B   batch size
# C   context length
# Q   Query head count
# K   KV head count
# P   output size per head
# R   Total KV output size



class GroupQueryAttention(torch.nn.Module):
    """
    Group-Query attention block.
    """
    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        query_head_count: int,
        kv_head_count: int,
        dropout_rate: float,
        qkv_bias: bool = False
    ):
        super().__init__()

        # This class implements <query_head_count> parallel attention heads. 
        # The output dimension <d_out> is divided equally among the heads.
        # We must therefore first ensure that the output size is divisible by
        # the number of heads.
        assert (d_out % query_head_count) == 0, "d_out must be divisible by query_head_count"

        # We must also ensure that <query_head_count> is divisible by <kv_head_count>
        assert (query_head_count % kv_head_count) == 0, 'query_head_count must be divisible by kv_head_count'


        self.d_out = d_out
        self.query_head_count = query_head_count
        self.kv_head_count = kv_head_count
        self.d_out_per_head = d_out // query_head_count

        self.q_head_per_kv_head = query_head_count // kv_head_count

        d_out_kv = self.d_out_per_head * kv_head_count

        # Next we instantiate the trainable parameters for the query, key
        # and value matrices.
        self.W_Q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)  # [O, I]
        self.W_K = torch.nn.Linear(d_in, d_out_kv, bias=qkv_bias)  # [R, I]
        self.W_V = torch.nn.Linear(d_in, d_out_kv, bias=qkv_bias)  # [R, I]
        
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

        assert count_tokens <= self.mask.size(0), 'Context size is too big'

        # Here we project the input tokens into the query, key and value spaces.
        # Note that the resulting query, key and value matrices contain values for all
        # heads with each head using separate slices.
        # [B, C, I] @ [O, I].T = [B, C, O]
        q = self.W_Q(x)  # [B, C, O]

        # [B, C, I] @ [R, I].T = [B, C, R]
        k = self.W_K(x)  # [B, C, R]
        v = self.W_V(x)  # [B, C, R]

        # Next we reshape the query, key and value matrices to separate slices
        # for each head. Note that the last dimension which was of size <O> is 
        # now split into <H> splits with size <P>.
        # [B, C, H, P]
        head_view_q = (batch_size, count_tokens, self.query_head_count, self.d_out_per_head)
        q = q.view(head_view_q)  # [B, C, Q, P]
        head_view_kv = (batch_size, count_tokens, self.kv_head_count, self.d_out_per_head)
        k = k.view(head_view_kv)  # [B, C, K, P]
        v = v.view(head_view_kv)  # [B, C, K, P]

        # Here we transpose the second and third dimensions, 
        # swapping the context length for the head count. This enables us 
        # to perform the matrix multiplication per head in parallel.
        q = q.transpose(1, 2)  # [B, Q, C, P]
        k = k.transpose(1, 2)  # [B, K, C, P]
        v = v.transpose(1, 2)  # [B, K, C, P]

        # We need to repeat the k and v tensor
        k = torch.repeat_interleave(k, self.q_head_per_kv_head, dim=1)  # [B, Q, C, P]
        v = torch.repeat_interleave(v, self.q_head_per_kv_head, dim=1)  # [B, Q, C, P]

        # Now we can compute the attention scores for each head in parallel.
        # [B, Q, C, P] @ [B, Q, C, P].T(2,3) = [B, Q, C, C]
        attention_scores = q @ k.transpose(2, 3) 

        # Next we apply the mask to the attention scores. This is used to prevent
        # the model from attending to tokens that are later in the context.
        # We fill the masked positions with negative infinity. The softmax function
        # will then set the weights of the masked positions to zero.
        mask = self.mask[:count_tokens, :count_tokens]  # [C, C]
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Next we apply the softmax function to the attention scores. This
        # normalizes the scores to a sum of 1. The division by the
        # square root of the output size per head helps with numerical stability.
        attention_weights = torch.softmax(
            attention_scores / self.d_out_per_head ** 0.5, dim=-1
        )
        # Finally we apply the dropout layer to the attention weights.
        attention_weights = self.dropout(attention_weights)

        # Here we compute the context vectors by doing a matrix multiplication between
        # the attention weights and the value vectors. The context vectors are then reshaped
        # to the "per head" shape.
        # ([B, Q, C, C] @ [B, Q, C, P]).T(1,2) = [B, C, Q, P]
        context_vectors = (attention_weights @ v).transpose(1, 2) 

        # The "per head" dimensions are now concatenated back into the original shape.
        # [B, C, Q, P].contiguous().view(B, C, O) = [B, C, O]
        context_vectors = context_vectors.contiguous().view(batch_size, count_tokens, self.d_out)

        # Finally we apply the additional output projection.
        # [B, C, O] @ [O, O] = [B, C, O]
        context_vectors = self.out_projection(context_vectors)

        return context_vectors