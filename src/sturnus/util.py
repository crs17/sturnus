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
       