# %% [markdown]
# ---
# title: "HCA-GPT: How to train a GPT from scratch"
# author: "Chresten R. Søndergaard"
# ---
# %%
# %load_ext autoreload
# %autoreload 2

# %% [markdown]
# # HCA-GPT: How to train a GPT from scratch
#
# In this notebook we will go through the steps needed to implement a GPT from scratch and train it - to the extent possible - on a consumer laptop. We will start by parsimg a piece of text, translating it into tokens and implementing a method splitting the tokens into batches to be used during training.
#
# Next, we will implement the building blocks of the GPT model. This includes of course the multi-head attention blocks and also feed-forward and layer normalization modules. We will combine all of these into the final GPT model.
#  
# The fairytale "The Nightingale" by H. C. Andersen will be used as training text. The artificial, mechanical nightingale seems a fitting analogy of the pattern recognition employed by GPTs. After training, our GPT should be able to generate text resembling mid-1800 Danish or at least fractions of the fairytale.
#
# This notebook is roughly based on chapters 1-5 of [Build A Larger Language Model (from scratch)](https://sebastianraschka.com/llms-from-scratch/) by Sebastian Raschka and I highly recommend the book.

# %% [markdown]
# ## 1. Translate the text into tokens
#
# GPTs do not work directly on raw text, instead the text must first be split into small chunks called tokens. The allowed tokens are predefined and given in a vocabolary of a fixed size. Since the vocabulary is fixed in size, each token is associated with an integer token id. Hence, the text is encoded into a string of intergers and can be losslessly be decoded back into the original text.
#
# We will use the GPT2 tokenizer which is based on [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding).
#
#

# %% [markdown]
# We read in the fairytale and see that it contains 18,136 characters:

# %%
with open('../text/nattergalen.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(text[:88])
print('Number of characters:', len(text))

# %% [markdown]
# Next we instantiate the tokenizer and we see that it works on vocabulary of 50,257 unique tokens:

# %%
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print('Vocabulary size:', tokenizer.n_vocab)


# %%
def show_token(i):
    try:
        u = tokenizer.token_byte_values()[i].decode('utf-8')
    except:
        u = '?'
    print(u)

for i in range(10_000, 10_005):
    show_token(i)

print('-'*100)

for i in range(50_000, 50_005):
    show_token(i)


# %% [markdown]
# We can see that the tokenizer is focused on English words, however, it will be able to encode any language as it will simply break down unknown words into smaller chunks untill it finds a matching token in the vocabulary.

# %% [markdown]
# Now let's encode our text giving us 7,621 tokens:

# %%
tokens = tokenizer.encode(text)
print(len(tokens))

# %% [markdown]
# We can decode the tokens back to the original text:

# %%
print('Token Id | Token')
print('---------+---------')
for i in range(10):
    token_id = tokens[i]
    token = tokenizer.decode([token_id])
    print(f'{token_id:<8} | "{token}"')

print()
print(tokenizer.decode(tokens)[0:100])


# %% [markdown]
# ## 2. Preparing the text for training
# The [make_data_loader](github.com) function takes some text and organises it into batches of input and target tokens. Let's give it a run to see how it works:

# %%
from sturnus.util import make_data_loader

data_loader = make_data_loader(text, batch_size=2, length_max=4, stride=4, shuffle=False)

print('All tokens: ', tokens)

for i, (input_tokens, target_tokens) in enumerate(data_loader):
    if i > 1:
        break

    print(f'Batch {i}:')
    print('input_tokens:\n', input_tokens)
    print('target_tokens:\n', target_tokens)
    print()




# %% [markdown]
# We asked for a batch size of 2 with a maximum length of 4 tokens per example and that is what we got. The first batch contain an input and a target tensor both of shape (2,4). We see that the target tokens are merely shifted one place to the right compared to the input tokens. This allows us to train the model to predict the second token when only given the first and to predict the third token when given the first and second tokens as input etc.
#
# The stride of 4 ensures that there are 4 tokens between each example. This corresponds to the maximum lenght and there is therefore no overlap between examples.

# %% [markdown]
# ## 3. Building blocks for our GPT implementation
# ### 3.1 Embedding vectors

# %% [markdown]
# Next we will add embedding vectors. Embedding vectors are vectors with learnable values representing each token value. So for each token in our vocabulary we will have a learnable embedding vector containing continuos values. We saw above that the BPE tokenizer has a vocabulry of 50,257 possible tokens. We will set up each embedding vector to hold 256 parameters so total we will need 50257 times 256 parameters:

# %%
import torch

vocab_size = 50257
output_dim = 256

torch.manual_seed(42)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer)

# %%
max_length = 4
data_loader = make_data_loader(
    text,
    batch_size=8,
    length_max=max_length,
    stride=max_length,
    shuffle=False
)

# inputs dim: [8, 4]
# output dim: [8, 4]

data_iter = iter(data_loader)
input_tokens, target_tokens = next(data_iter)

print(input_tokens.shape)
print(target_tokens.shape)



# %%
token_embeddings = embedding_layer(input_tokens)
print(token_embeddings.shape)

# %% [markdown]
# So the token embeddings have the shape [batch_size, max_length, embedding_dim] = [8, 4, 256].
#
# The `token_embeddings` do not contain any information on the positions of the individual tokens in the text. However, context is of course important when training a language model. So in order to include information about position, we will add positional embedding vector that is also learnable:

# %%
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
print(pos_embeddings)

# %%
input_embeddings = token_embeddings + pos_embeddings # [8, 4, 256] + [4, 256] = [8, 4, 256]
print(input_embeddings.shape)

# %% [markdown]
# The `pos_embeddings` tensor is broadcastet along the batch dimennsion of the `token_embeddings` tensor.

# %% [markdown]
# ### 3.2 Self-attention
#
# Self-attention or scaled dot-product attention is at the heart of the attention block and thereby the GPT models. This is what is described in Equation 1 of the famous "[Attention is all you need](https://arxiv.org/abs/1706.03762)" paper. 
# For simplicity, we start out with a simple example with an input containing six tokens each represented by a three dimensional embedding vector:

# %%
torch.manual_seed(42)
d_embed = 3 
inputs = torch.randn(6, d_embed)
print(inputs)


# %% [markdown]
# The first step of the attention mechsnism is to calculate *query*, *key* and *value* vectors for each input token. These are calculated using parameters that are optimizied during the training the GPT. As an example we can instantiate parameter matrices that project from the three-dimensional input embeddings to two-dimensional vectors:

# %%
d_out = 2

torch.manual_seed(42)
W_Q = torch.nn.Parameter(torch.randn(d_embed, d_out), requires_grad=False)
W_K = torch.nn.Parameter(torch.randn(d_embed, d_out), requires_grad=False)
W_V = torch.nn.Parameter(torch.randn(d_embed, d_out), requires_grad=False)

# %% [markdown]
# With these paramters instantiated, we can now calculate the query, key and value vectors. We do it at onces for all six input tokens using matrix multiplication, giving us matrices of shape 6x2:

# %%
keys = inputs @ W_K  # [6, 3] @ [3, 2] = [6, 2]
values = inputs @ W_V  # [6, 3] @ [3, 2] = [6, 2]
queries = inputs @ W_Q  # [6, 3] @ [3, 2] = [6, 2]

# %% [markdown]
# From the queries and the keys, we then calculate the attention scores:

# %%
attention_scores = queries @ keys.T  # [6, 2] @ [6, 2].T = [6, 6]

# %% [markdown]
# We then use softmax to normalize the attention scores. The factor $\sqrt{d_{key}}$ is the reason for the name *scaled* dot-product attention. When the embedding dimension is increased in size, the gradients of the attention weights can become quite small leading to inefficient training. The scaling factor is a remedy for this problem.

# %%
d_key = keys.shape[-1]
attention_weights = torch.softmax(attention_scores / d_key ** 0.5, dim=-1)
print(attention_weights)

# %% [markdown]
# Each row sums up to 1.0:

# %%
attention_weights.sum(dim=-1)

# %% [markdown]
# Finally, we can calculate the context vectors which are the result of the attention calculation:

# %%
context_vectors = attention_weights @ values  # [6, 6] @ [6, 2] = [6, 2]
print(context_vectors)

# %% [markdown]
# The final implementation of the attention calculation can be seen the model.py file. A number of things have been added:
# - The algorithm has now been implemented as a PyTorch Module which makes it easy to integrate it into our final GPT model.
# - We have extended it to multi-head attention meaning that instead of just one attention head, we now have multiple heads working in parallel allowing them focus on different aspects of the input string. For efficiency all attention heads share the same parameter matrices allowing matrix multiplication to be done for all heads in one go. Different slices of the matrices are then used by each head.
# - A [dropout](https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) layer has been added for regularization and to prevent over-reliance on specific paramter values.

# %%
from sturnus.models.gpt2 import MultiHeadAttention

batch = torch.stack((inputs, inputs), dim=0)  # [B, C, I] = [2, 6, 3]
print('Bath shape:', batch.shape)

mha = MultiHeadAttention(
    d_in=3,
    d_out=20,
    context_length=6,
    head_count=5,
    dropout_rate=0.4
)

context_vectors = mha(batch)  # [batch_size, token_count, d_out] = [2, 6, 20]
print('Context vectors shape:', context_vectors.shape)  

# %% [markdown]
# ## GPT Implementation

# %%
GPT_CONFIG_124 = {
    'vocab_size': 50257,
    'block_size': 1024,
    'count_heads': 12,
    'count_blocks': 12,
    'embed_dim': 768,
    'dropout': 0.1,
    'qkv_bias': False,
}

# %% [markdown]
# ### Layer Normalization
#
# [Layer normalisation](https://arxiv.org/abs/1607.06450) normalizes the inputs across the embedding dimension. This is often used in deep neural netowrks to improve training efficiency and reduce numerical instabilities. Below we will instantiate the LayerNorm class from the [model.py](https://github.com/crs17/sturnus/blob/main/src/sturnus/model.py) file and pass the inputs through it. Since the parameters `scale` and `shift` are initiialized to 1 and 0, respectively, we see that the normalized inputs have (close to) mean 0 anv variance 1 for each example:

# %%
from sturnus.models.gpt2 import LayerNorm

print('Input mean:\n', torch.mean(inputs, dim=-1))
print('Input variance:\n', torch.var(inputs, dim=-1))

ln = LayerNorm(3)
inputs_normed = ln.forward(inputs)

print('Normalized mean:\n', torch.mean(inputs_normed, dim=-1))
print('Normalized variance:\n', torch.var(inputs_normed, dim=-1))    







# %% [markdown]
# ### Activation function
# The GELU, Guassian error linear unit, activation function is often used in LLMs. It has the advantage over the traditional ReLU, Rectified Linear Unit, activation function, that it has non-zero values for negative values of x and therefore yields non-zero gradients.
# The GELU activation function is given as $GELU(x) = \phi(x) \cdot x$ where $\phi(x)$ is the cumulative distribution function of the normal distribution. It turns out that it can be approximated to lessen the computational burden using: $\frac{x}{2} \cdot (1 + \tanh(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715 x^{3})$ and that is the approximation we will use.

# %%
import matplotlib.pyplot as plt

x = torch.linspace(-3, 3, 100)
gelu = torch.nn.GELU()
gelu_approx = torch.nn.GELU(approximate='tanh')

plt.plot(x, torch.relu(x), label='ReLU')
plt.plot(x, gelu(x), label='GELU')
plt.plot(x, gelu_approx(x), '--', label='GELU (approx)')
plt.grid()
plt.xlabel('x')
plt.ylabel('Activation function')
plt.legend()
plt.show()




# %%



# %%
from sturnus.models.gpt2 import FeedForward

ffn = FeedForward(GPT_CONFIG_124)
x = torch.randn(2, 1024, 768)
y = ffn(x)
print(y.shape)


# %% [markdown]
# ### Shortcut Connections
# Shortcut connections are frequently used in deep neural networks as they gradients to travel more quickly through the layers. Below is a small example with made up training data and a neural network with 10 layers each holding 10 neurons. In the plot below we see that the training of the network with shortcut connections is typically more efficient (the loss goes down quicker) than for the network without.

# %%
class ShortcutExampleNN(torch.nn.Module):
    def __init__(self, count_layers: int, use_shortcut: bool):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(10, 10),
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
    


# %%
def train_shortcut_example(model, x, target):
    losses = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for i in range(50):
        y = model(x)
        loss = torch.nn.MSELoss()(y, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses

torch.manual_seed(42)
x = torch.randn(10)
target = torch.randn(10)

losses_shortcut = train_shortcut_example(
    ShortcutExampleNN(count_layers=10, use_shortcut=True), x, target
)
losses_no_shortcut = train_shortcut_example(
    ShortcutExampleNN(count_layers=10, use_shortcut=False), x, target
)

plt.plot(losses_shortcut, label='Shortcut')
plt.plot(losses_no_shortcut, label='No shortcut')
plt.legend()
plt.yscale('log')
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# %% [markdown]
# ### Transformer Block

# %%

# %%
from sturnus.models.gpt2 import TransformerBlock

torch.manual_seed(123)

x = torch.randn(2, 1024, 768)

transformer_block = TransformerBlock(GPT_CONFIG_124)
y = transformer_block(x)
print(x.shape)
print(y.shape)


# %% [markdown]
# ### GPT model

# %%
from sturnus.models.gpt2 import GPTModel


torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124)

batch = torch.tensor(
    [[6109,  3626,  6100,   345],
     [6109,  1110,  6622,   257]]
)

y = model(batch)
print(batch.shape)
print(y.shape)
print(y)



# %%
total_parameters = sum([p.numel() for p in model.parameters()])
print(f'Total parameters: {total_parameters}')






# %%
from sturnus.util import text_to_tokens, tokens_to_text, generate_text_simple


start_text = 'Hello, I am'
encoded = tokenizer.encode(start_text)
print(encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(encoded_tensor)

model.eval()
out = generate_text_simple(
    model,
    encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124['block_size']
)
print(out)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

# %% [markdown]
# ## LLM Pre-training

# %%
GPT_CONFIG_124_local = {
    'vocab_size': 50257,
    'block_size': 256,  # 1024, This makes it possible to train on a normal laptop
    'count_heads': 12,
    'count_blocks': 12,
    'embed_dim': 768,
    'dropout': 0.1,
    'qkv_bias': False,
}

# %%
import torch

torch.manual_seed(42)

model = GPTModel(GPT_CONFIG_124_local)
# # model.eval?


# %%
sum([p.numel() for p in model.parameters()])

# %%


start_text = 'Every effort moves you'
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_tokens(start_text, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124_local['block_size']
)
print('Ouput text:', tokens_to_text(token_ids, tokenizer))




# %% [markdown]
# ### Small example

# %%
# inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
#                        [40,    1107, 588]])   #  "I really like"]

# targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
#                         [1107, 588, 11311]])  #  " really like chocolate"]

# %%
# with torch.no_grad():     #1
#     logits = model(inputs)
# probas = torch.softmax(logits, dim=-1)     #2
# print(probas.shape)

# %%
# token_ids = torch.argmax(probas, dim=-1, keepdim=True)
# print("Token IDs:\n", token_ids)  


# %%
# text_idx = 0
# target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 1:", target_probas_1)

# text_idx = 1
# target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
# print("Text 2:", target_probas_2)

# %%
# log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
# print(log_probas)

# %%
# avg_log_probas = torch.mean(log_probas)
# print(avg_log_probas)

# %%
# targets.shape

# %%
# logits_flat = logits.flatten(0, 1)
# targets_flat = targets.flatten()

# loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
# print(loss)






# %%
# # loss.item?

# %%
# perplexity = torch.exp(loss)
# print(perplexity)

# %%
total_characters = len(text)
total_tokens = len(tokenizer.encode(text))
print(f'Total characters: {total_characters}')
print(f'Total tokens: {total_tokens}')

# %%
### text = text_the_verdict

train_ratio = 0.9
split_idx = int(train_ratio * total_characters)
train_text = text[:split_idx]
val_text = text[split_idx:]
print(f'Train text length: {len(train_text)}')
print(f'Validation text length: {len(val_text)}')




# %%
torch.manual_seed(123)

train_loader = make_data_loader(
    train_text,
    batch_size=2,
    length_max=GPT_CONFIG_124_local['block_size'],
    stride=GPT_CONFIG_124_local['block_size'],
    shuffle=True,
    drop_last=True,
    worker_count=0
)

val_loader = make_data_loader(
    val_text,
    batch_size=2,
    length_max=GPT_CONFIG_124_local['block_size'],
    stride=GPT_CONFIG_124_local['block_size'],
    shuffle=True,
    drop_last=True,
    worker_count=0
)

# %%
print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# %%
from sturnus.util import calc_loss_loader, calc_loss_loader, train_model_simple



# %%
torch.backends.mps.is_available()

# %%
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print(f'Train loss: {train_loss}')
print(f'Validation loss: {val_loss}')






# %%
device

# %%
torch.cuda.is_available()

# %%

# %%

# %%



# %%
GPT_CONFIG_124_local

# %%
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124_local)
model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004,
    weight_decay=0.1
)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq=10,
    eval_iter=5,
    start_context='Keiserens største ønske er',
    tokenizer=tokenizer
)

# %%
print(tokens_seen)
train_loader.dataset.input_tokens.shape
25*256

# %%
fig, ax = plt.subplots()
epochs_seen = torch.linspace(0, num_epochs, len(train_losses))
ax.plot(epochs_seen, train_losses, label='Train loss')
ax.plot(epochs_seen, val_losses, label='Validation loss')
ax2 =  ax.twiny()
ax2.plot(tokens_seen, train_losses, alpha=0)
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax2.set_xlabel('Tokens seen')
# ax.set_yscale('log')
ax.grid()
plt.show()




# %% [markdown]
# As expexted we see that the training error continues to go down for each epoch whereas the validation error only decreases to begin with and begins to increase as the model overtrains on the training set.

# %%
train_text[train_text.find('Saa begyndte'):]

# %%
val_text[val_text.find('Saa begyndte'):]

# %%
model.to('cpu')
model.eval();

# %% [markdown]
# After having trained the model, we can now experiment with generating some text:

# %%
from sturnus.util import generate

start_context = 'Saa begyndte'
device = torch.device('cpu')

torch.manual_seed(42)

new_tokens = generate(
    model,
    idx=text_to_tokens(start_context, tokenizer),
    max_new_tokens=100,
    context_size=GPT_CONFIG_124_local["block_size"],
    top_k=30,
    temperature=1.5
)
print(tokens_to_text(new_tokens, tokenizer))

# %% [markdown]
# Just as the mechanical nightingale could not save the emperior, we see that while the output of the model has similarities with old-fashioned Danish, it is non-sensical. To train a proper model, we need a lot more text for trainning and a lot more compute.
#
# However, we can still experiment with the temperature and top_k settings in the generate function. The temperature setting is increased it will increase the variance or "creativity" of the output. As a safeguard the top_k setting ensures that only the *k* most likely tokens are selected which should ensure that the output still make gramatical sense (at least on a well-trained model).

# %%
torch.manual_seed(42)

token_ids = {}
model.eval()
for temperature in [0, 0.1, 1.0, 5.0]:
    token_ids[temperature] = []

    for i in range(100):
        tid = generate(
            model=model,
            idx=text_to_tokens(start_context, tokenizer),
            max_new_tokens=1,
            context_size=GPT_CONFIG_124_local["block_size"],
            top_k=10,
            temperature=temperature
        )[-1][-1]
        token_ids[temperature].append(int(tid))


# %%
import itertools

tokens_unique = sorted(
    list(
        set[int](
            itertools.chain.from_iterable(token_ids.values())
        )
    )
)

N = len(token_ids)

fig, ax = plt.subplots()
for k, temperature in enumerate(token_ids.keys()):
    ax.bar(
        [i + (-N/2 + k+1)/10 for i in range(len(tokens_unique))],
        [
            len([t for t in token_ids[temperature] if t == tu])
            for tu in tokens_unique
        ],
        width=0.08,
        label=f'Temperature {temperature}'
    )
ax.set_xticks(range(len(tokens_unique)))
ax.set_xticklabels([f'{t:03d}\n {tokenizer.decode([t])}' for t in tokens_unique])
ax.legend()
ax.set_xlabel('Next token')
ax.set_ylabel('Frequency')
ax.set_title(f'Next token frequency ({start_context}...)')
plt.show()


# %% [markdown]
# At temperature 0 the most likely next token is always selected. As the temperature is increased more and more randomness is introduced.

# %% [markdown]
#
