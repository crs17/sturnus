import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, encoded_text, length_max, stride):
        self.input_tokens = []
        self.target_tokens = []

        for i in range(0, len(encoded_text) - length_max, stride):
            input_tokens = encoded_text[i: i + length_max]
            target_tokens = encoded_text[i + 1: i + length_max + 1]
            self.input_tokens.append(input_tokens)
            self.target_tokens.append(target_tokens)

        self.input_tokens = torch.tensor(self.input_tokens)
        self.target_tokens = torch.tensor(self.target_tokens)
    
    def __len__(self):
        return len(self.input_tokens)
    
    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]


def make_data_loader(text, batch_size, length_max, stride, shuffle=True, drop_last=True, worker_count=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded_text = tokenizer.encode(text)
    dataset = TextDataset(encoded_text, length_max, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=worker_count
    )


def generate(
    model,
    idx,
    max_new_tokens,
    context_size,
    temperature=0.0,
    top_k=None,
    eos_id=None
    ):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # In order to prevent the model from generating gibberish,
        # we can use the top_k parameter to ensure that only the most
        # likely tokens are considered even when temperature is increased.
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_value = top_logits[:, -1]
            logits = torch.where(
                logits < min_value,
                torch.tensor(-float('inf')),
                logits
            )
        # If temperature is greater than 0, we will scale the logits by the temperature.
        # Temperatures less than 1 lead to more deterministic output, while temperatures
        # greater than 1 lead to more variable or "creative" output.
        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probabilities, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
