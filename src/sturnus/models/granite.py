import torch

from sturnus.models.rope import calculate_thetas, apply_rope_half


# In order to keep track of the dimensions of each array, 
# we use the following one-letter abbreviations:
# I   self attention block input size
# O   self attention block output size
# B   batch size
# C   context length
# Q   Query head count
# K   KV head count
# R   Total KV output size
# P   output size per head
# E   Number of total experts in layer
# K   Number of expers to use for each token
# M   Hidden size of each expert MLP



class ExpertSwiGLU(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.gate_W = torch.nn.Linear(config['embed_dim'], config['mlp_hidden_size'], bias=False)  # [I, M]
        self.up_W = torch.nn.Linear(config['embed_dim'], config['mlp_hidden_size'], bias=False)  # [I, M]
        self.down_W = torch.nn.Linear(config['mlp_hidden_size'], config['embed_dim'], bias=False)  # [M, I]

        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_W(x)  # [B, C, M]
        up = self.up_W(x)  # [B, C, M]

        return self.down_W(self.silu(gate) * up)   # [B, C, I]


class RMSNorm(torch.nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = torch.nn.Parameter(torch.ones(emb_dim).float())

    def forward(self, x):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(means + self.eps)
        return (x_normed * self.weight)




class GroupQueryAttentionWithRoPE(torch.nn.Module):
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
        attention_multiplier: float,
        theta_zero: int,
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

        self.attention_multiplier = attention_multiplier

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
        self.out_projection = torch.nn.Linear(d_out, d_out, bias=False)  # [O, O]
        self.dropout = torch.nn.Dropout(dropout_rate)

        # Pre-calculate thetas for Rotational Position Embeddings
        c_values, _ = calculate_thetas(
            embedding_dimension=self.d_out_per_head,
            context_size=context_length,
            theta_zero=theta_zero
        )
        self.register_buffer('c_values', c_values)

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
        # for each head. Note that the last dimension which was of size O or R is 
        # now split into Q or K splits with size P.
        head_view_q = (batch_size, count_tokens, self.query_head_count, self.d_out_per_head)
        q = q.view(head_view_q)  # [B, C, Q, P]
        head_view_kv = (batch_size, count_tokens, self.kv_head_count, self.d_out_per_head)
        k = k.view(head_view_kv)  # [B, C, K, P]
        v = v.view(head_view_kv)  # [B, C, K, P]

        # Apply RoPE
        q = apply_rope_half(q, self.c_values[:count_tokens, :])  # [B, Q, C, P]
        k = apply_rope_half(k, self.c_values[:count_tokens, :])  # [B, Q, C, P]

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

        
        attention_scores = q @ k.transpose(2, 3)  * self.attention_multiplier

        # Next we apply the mask to the attention scores. This is used to prevent
        # the model from attending to tokens that are later in the context.
        # We fill the masked positions with negative infinity. The softmax function
        # will then set the weights of the masked positions to zero.
        mask = self.mask[:count_tokens, :count_tokens]  # [C, C]
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))

        # Next we apply the softmax function to the attention scores. This
        # normalizes the scores to a sum of 1.
        attention_weights = torch.softmax(attention_scores, dim=-1)
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
        # [B, C, O] @ [O, O].T = [B, C, O]
        context_vectors = self.out_projection(context_vectors)

        return context_vectors


class NaiveMixtureOfExpertsLayer(torch.nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        count_experts = config['count_experts']
        self.count_experts_used = config['count_experts_used']

        self.routing_weights = torch.nn.Linear(config['embed_dim'], count_experts, bias=False)

        self.experts = torch.nn.ModuleList(
            [ExpertSwiGLU(config) for _ in range(count_experts)]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project the inputs into the dimension of the number of experts. This way we will get 
        # one value for each token and each expert
        logits = self.routing_weights.forward(x)  # [B, C, E]

        # Convert the logits to probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)  # [B, C, E]

        # We now select the top <count_experts_used> experts for each token
        expert_weights, expert_indices = torch.topk(
            probabilities, self.count_experts_used
        )  # [B, C, K]

        # Normalize the weights of the selected experts
        expert_weights /= expert_weights.sum(keepdim=True, dim=-1)  # [B, C, K]

        # Instantiate the y tensor with zeros - we will sum the outputs from each expert 
        # in this tensor
        y = torch.zeros_like(x)  # B, C, I

        for i, expert in enumerate(self.experts):
            # Make a mask for the current expert over the tokens
            mask = (expert_indices == i)  # [B, C, K]

            # Use the mask to filter out the tokens in the current batch that are to 
            # be processed by the current expert
            tokens_for_expert = x[mask.any(dim=-1)]  # [?, I]
          
            # Pass the selected token through the current expert
            output = expert.forward(tokens_for_expert)  # [?, I]

            # Find the indices of the processed tokens in the original input tensor 
            # so that we can fill them in correctly in the output tensor
            indices_of_tokens_for_expert = torch.nonzero(mask)  # [?, 3]

            # Go through each of the processed tokens, look up the corresponding expert weight
            # for that token and add the scaled contribution to the output tensor  
            for index, vector in zip(indices_of_tokens_for_expert, output):
                weight = expert_weights[index[0]][index[1]][index[2]]
                y[index[0]][index[1]] += weight * vector  # [I]
            
        return y



class GraniteTransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = GroupQueryAttentionWithRoPE(
            d_in=config['embed_dim'],
            d_out=config['embed_dim'],
            context_length=config['block_size'],
            query_head_count=config['count_query_heads'],
            kv_head_count=config['count_kv_heads'],
            dropout_rate=config['dropout'],
            attention_multiplier=config['attention_multiplier'],
            theta_zero=config['rope_theta'],
            qkv_bias=config['qkv_bias']
        )
        self.residual_multiplier = config['residual_multiplier']
        self.rmsn1 = RMSNorm(config['embed_dim'], eps=config['rms_norm_eps'])
        self.moe = NaiveMixtureOfExpertsLayer(config)

        self.rmsn2 = RMSNorm(config['embed_dim'], eps=config['rms_norm_eps'])
        self.dropout = torch.nn.Dropout(config['dropout'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        # Identifies and analyses relationships between tokens
        shortcut = x
        x = self.rmsn1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x * self.residual_multiplier + shortcut

        # Feed-forward block
        # Modifies tokens individually - no information is shared between tokens
        shortcut = x
        x = self.rmsn2(x)
        # x = self.ffn(x)
        x = self.moe(x)
    
        x = self.dropout(x)
        x = x * self.residual_multiplier + shortcut

        return x


class GraniteModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.embedding_multiplier = config['embedding_multiplier']
        self.logits_scaling = config['logits_scaling']

        self.drop_emb = torch.nn.Dropout(config['dropout'])

        self.trf_blocks = torch.nn.Sequential(
            *[GraniteTransformerBlock(config) for _ in range(config['count_blocks'])]
        )

        self.final_norm = RMSNorm(config['embed_dim'], eps=config['rms_norm_eps'])
        self.out_head = torch.nn.Linear(
            config['embed_dim'], config['vocab_size'], bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        x = self.tok_emb(x) * self.embedding_multiplier
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        logits = logits / self.logits_scaling

        return logits