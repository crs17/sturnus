# %% [markdown]
# # Mixture of Experts
#
# In this notebook we set out to add [sparsely-gated mixture of expert layers](https://arxiv.org/abs/1701.06538) (MoE) to our implementation of the GPT2 model.
#
# In the standard GPT2 model each transformer block has a single feed-forward neural network. The idea in MoE is to instead have multiple feed-forward neural networks in each transformer block with the added condition that only a subset of these networks are active for each forward pass. This allows the neural networks to specialize and gives the model more parameters without significantly increased compute costs.
#
# A gating network $G(x)$ controls which experts are used given the input data and the output of the MoE module then becomes:
#
# $$y = \sum_{i}^{n} G(x)_i E_i(x)$$
#
# where $G(x)_i$ is the i'th output of the gating network, $E_i(x)$ is the output of the i'th expert and $n$ is the number of experts.
#
# The gating network can be a single linear layer without bias with a softmax activation:
#
# $$G(x) = \text{softmax}(\text{topK}(x \cdot W_g))$$
#
# In the Shazeer paper linked above, a noise terms was also added in the gating function to introduce some randomness in the selection projects. This is done to enable better load-balancing between the experts. However, later implementations such as the [Switch Transformers](https://arxiv.org/abs/2101.03961) and [GLaM](https://arxiv.org/abs/2112.06905) have abandonded the noise term again. Instead, an auxillary loss term is added to ensure load-balancing.
#
#
# Expert capacity can be defined as:
#
# $$ E = \frac{k \cdot B \cdot C}{ N} \alpha $$
#
# where $k$ is the number of active experts, $B$ is the batch size, $C$ is the context length, $N$ is the total number of experts and $\alpha$ is the capacity factor. A capacity factor of 1 enforces equal distribution of the tokens but if the router decides to send a larger proportion of the tokens in a batch to a specific expert, those additional tokens are dropped by the epxert (however they are still passed on via the shortcut connection of the transformer block). Setting a larger capacity factor reduces the amount of droppped tokens at a efficiency loss due to less uniform distribution among the experts.
#
#

# %% [markdown]
# We intend re-implement [`OLMoE-1B-7B`](https://huggingface.co/allenai/OLMoE-1B-7B-0125). With quantization this model is small enough that we can download the open weight and run it locally to verify that the implementation works.

# %% [markdown]
#

# %%
import torch
from sturnus.models.gpt2 import FeedForward, MultiHeadAttention, LayerNorm, GPTModel


# In order to keep track of the dimensions of each array, 
# we use the following one-letter abbreviations:
# I   self attention block input size
# O   self attention block output size
# B   batch size
# C   context length
# H   Head count
# P   output size per head
# E   Number of total experts in layer
# K   Number of expers to use for each token


class NaiveMixtureOfExpertsLayer(torch.nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        count_experts = config['count_experts']
        self.count_experts_used = config['count_experts_used']

        self.routing_weights = torch.nn.Linear(config['embed_dim'], count_experts, bias=False)

        self.experts = torch.nn.ModuleList(
            [FeedForward(config) for _ in range(count_experts)]
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


class NaiveMoETransformerBlock(torch.nn.Module):
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
        # self.ffn = FeedForward(config)
        self.moe = NaiveMixtureOfExpertsLayer(config)

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
        # x = self.ffn(x)
        x = self.moe(x)
    
        x = self.dropout(x)
        x = x + shortcut

        return x

class NaiveMOEGPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(config['vocab_size'], config['embed_dim'])
        self.pos_emb = torch.nn.Embedding(config['block_size'], config['embed_dim'])

        self.drop_emb = torch.nn.Dropout(config['dropout'])

        self.trf_blocks = torch.nn.Sequential(
            *[NaiveMoETransformerBlock(config) for _ in range(config['count_blocks'])]
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



# %%
GPT_MOE_CONFIG = {
    'vocab_size': 50257,
    'block_size': 1024,
    'count_heads': 12,
    'count_blocks': 12,
    # 'embed_dim': 768,
    'embed_dim': 24,
    'dropout': 0.1,
    'qkv_bias': True, # Used in GPT2 but typically not in modern LLMs as the biases do not improve performance
    'count_experts': 4,
    'count_experts_used': 2,
}


torch.manual_seed(42)

MoEGPT = NaiveMOEGPTModel(GPT_MOE_CONFIG)

import tiktoken
from sturnus.util import generate_and_print_sample

tokenizer = tiktoken.get_encoding("gpt2")
device='cpu'
start_context = 'Hello, world'

generate_and_print_sample(MoEGPT, tokenizer, device, start_context)

# %% [markdown]
# The samllest open-weight MoE model I could find is IBM's `Granite-3.0-1B-A400M`. So the goal is to re-implement this model so that we can download the open weights for testing the MoE implementation. However, the `Granite-3.0-1B-A400M` model has a few additional changes relative to out GPT2 implementation besides the Mixture of Experts - so we need to implement these as well:
# - RoPE position embeddings instead of learned absolute position embeddings
# - RMSNorm instead of LayerNorm
# - SwiGLU instead of GELU activation
# - Grouped-query Attention instead of Multi-head Attention
# - Some scalar multipliers
