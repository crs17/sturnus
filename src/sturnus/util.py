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

def text_to_tokens(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def tokens_to_text(tokens, tokenizer):
    return tokenizer.decode(tokens.squeeze(0).tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    logits_flat = logits.flatten(0, 1)
    targets_flat = target_batch.flatten()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss


def calc_loss_loader(dataloader, model, device, batch_count=None):
    total_loss = 0
    if len(dataloader) == 0:
        return float('nan')

    if batch_count is None:
        batch_count = len(dataloader)
    else:
        batch_count = min(batch_count, len(dataloader))

    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= batch_count:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, device).item()

    return total_loss / batch_count


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    
    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_tokens(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = tokens_to_text(token_ids, tokenizer)
    print(decoded_text.replace('\n', ' '))
    model.train()


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, batch_count=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, batch_count=eval_iter)
    model.train()
    return train_loss, val_loss

