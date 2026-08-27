import pandas as pd
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data.iloc[idx]['Tokens'], dtype=torch.long),
            torch.tensor(self.data.iloc[idx]['Class'], dtype=torch.long)
        )


def calc_accuracy_loader(model, dataloader, device, num_batches=None):
    model.eval()
    model.to(device)
    examples_count, correct_count = 0, 0

    if num_batches is not None:
        num_batches = min(num_batches, len(dataloader))
    else:
        num_batches = len(dataloader)

    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break

        inputs, targets = batch

        inputs = inputs.to(device)
        targets = targets.to(device)
     
        logits = model(inputs)[:, -1, :]
        predicted_classes = torch.argmax(logits, dim=-1)
        
        correct_count += (predicted_classes == targets).sum().item()
        examples_count += targets.shape[0]

    return correct_count / examples_count


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(model, data_loader, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        total_loss += calc_loss_batch(
            input_batch, target_batch, model, device
        ).item()

    return total_loss / num_batches


def fine_tune_classification(
    model, optimizer, device, train_loader, validation_loader, num_epochs,
    eval_batch_count=10, eval_freq=10,
    ):

    train_losses, val_losses, eval_steps, train_accuracies, validation_accuracies = [], [], [], [], []
    global_step = -1
    examples_seen = 0

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch} has started')
        model.train()
        for (input_batch, target_batch) in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            global_step += 1
            examples_seen += target_batch.shape[0]

            if global_step % eval_freq == 0:
                model.eval()
                train_loss = calc_loss_loader(model, train_loader, device, eval_batch_count)
                val_loss = calc_loss_loader(model, train_loader, device, eval_batch_count)

                print(f'Step: {global_step:5d} Train loss: {train_loss:8.4f} Validation loss: {val_loss:8.4f}')
                eval_steps.append(global_step)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                model.train()
        
        train_acc = calc_accuracy_loader(model, train_loader, device, eval_batch_count)
        val_acc = calc_accuracy_loader(model, validation_loader, device, eval_batch_count)

        print(f'Epoch {epoch} done. Train accuracy: {train_acc * 100:5.2f} %. Validation accuracy: {val_acc*100:5.2f} %.')
        train_accuracies.append(train_acc)
        validation_accuracies.append(val_acc)
    
    return train_losses, val_losses, eval_steps, train_accuracies, validation_accuracies, examples_seen

