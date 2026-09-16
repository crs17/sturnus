# %% [markdown]
# ---
# title: "Grouped-query Attention"
# author: "Chresten R. Søndergaard"
# ---

# %% [markdown]
# [Group-query Attention (GQA)](https://arxiv.org/abs/2305.13245) divides query heads into groups where each group share a single key and value head. It can be seen as a compromise between standard multi-head attention where keys and values are calculated for query each head and [Multi-query attention (MQA)](https://arxiv.org/abs/1911.02150) where single key and value heads are shared for all query heads. Hence it seeks to gain some of the efficiency gain of MQA while still preserving most of the performance of full MHA.
#
# My implementation GQA with some hopefully helpful comments can be found [here](https://github.com/crs17/sturnus/blob/main/src/sturnus/models/gqa.py).
#
# In this short demo, we will not go into evaluating performance differences between MHA and GQA. Instead we will just do a quick comparison of the number of parameters in a GPT2-style attention block with either MHA or GQA grouping the 12 heads into 3 groups. However, we will use GQA in later demos.

# %%
from sturnus.models.gpt2 import MultiHeadAttention
from sturnus.models.gqa import GroupQueryAttention


mha_config = {
    'vocab_size': 50257,
    'block_size': 1024,
    'count_heads': 12,
    'count_blocks': 12,
    'embed_dim': 768,
    'dropout': 0.1,
    'qkv_bias': True,
}


mha = MultiHeadAttention(
    d_in=mha_config['embed_dim'],
    d_out=mha_config['embed_dim'],
    context_length=mha_config['block_size'],
    head_count=mha_config['count_heads'],   
    dropout_rate=mha_config['dropout'],
    qkv_bias=mha_config['qkv_bias']
)

gqa_config = {
    'vocab_size': 50257,
    'block_size': 1024,
    'count_query_heads': 12,
    'count_kv_heads': 3,
    'count_blocks': 12,
    'embed_dim': 768,
    'dropout': 0.1,
    'qkv_bias': True,
}


gqa = GroupQueryAttention(
    d_in=gqa_config['embed_dim'],
    d_out=gqa_config['embed_dim'],
    context_length=gqa_config['block_size'],
    query_head_count=gqa_config['count_query_heads'],
    kv_head_count=gqa_config['count_kv_heads'],
    dropout_rate=gqa_config['dropout'],
    qkv_bias=gqa_config['qkv_bias']
)


# %%
def count_parameters(module):
    total = 0
    for name, param in module.named_parameters():
        n = param.numel()
        print(f'{name:25} {n:7}')
        total += n
    print('---------------------------------')
    print(f'Total:                    {total:7}')
    return total


print('Multi-head attention block:')
N_mha = count_parameters(mha)
print('\n\n')
print('Group-query attention block:')
N_gqa = count_parameters(gqa)
print('\n\n')
print(f'Difference: {N_mha - N_gqa} ({(N_mha - N_gqa)/N_mha * 100} %)')


# %% [markdown]
# We see that for a single GPT2-style attention block we save 885,888 parameters or 37.5 % by substituting MHA for GQA.
