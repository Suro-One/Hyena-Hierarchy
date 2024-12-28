import uuid
import os
import random
import copy
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm

print(torch.cuda.is_available())
print("CUDA: ", torch.version.cuda)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================
# Hyena Hierarchy with EWC
# ==========================
class HyenaWithEWC(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dim_feedforward=2048, dropout=0.1):
        super(HyenaWithEWC, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Create Hyena-like layers (simplified: conv + gating + feedforward)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'conv': nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=4,
                    stride=1,
                    padding='same',
                    dilation=1,
                    groups=1,
                    bias=True,
                    padding_mode='reflect'
                ),
                'gate': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                ),
                'dropout': nn.Dropout(dropout)
            }) 
            for _ in range(n_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

        # EWC-specific attributes
        self.old_params = None
        self.fisher_diagonal = None

    def forward(self, src):
        """
        src shape: (batch, seq_len)
        Embedding -> (batch, seq_len, d_model)
        Conv expects (batch, d_model, seq_len) so we transpose
        Then apply gating, feed-forward, etc.
        """
        src = self.embedding(src)  # (B, S, d_model)
        for layer in self.layers:
            # Conv
            x = layer['conv'](src.transpose(1, 2))  # (B, d_model, S)
            x = x.transpose(1, 2)                   # (B, S, d_model)

            # Gating
            gate = torch.sigmoid(layer['gate'](x))
            x = x * gate

            # FFN + Dropout
            x = layer['ffn'](x)
            x = layer['dropout'](x)

            src = x

        return self.output(src)

    def calculate_fisher(self, dataset, device, samples=2000):
        """
        Approximates the diagonal of the Fisher Information Matrix by randomly
        sampling 'samples' number of sequences from dataset.
        """
        # Ensure the model is on the correct device
        self.to(device)
        self.eval()

        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}

        count = 0
        for seq in dataset:
            seq = seq.unsqueeze(0).to(device)  # move input to the same device
            self.zero_grad()

            output = self(seq[:, :-1])
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), 
                seq[:, 1:].contiguous().view(-1)
            )
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / samples

            count += 1
            if count >= samples:
                break

        self.fisher_diagonal = fisher
        self.old_params = copy.deepcopy(self.state_dict())

    def ewc_loss(self, lamda=15):
        """
        Computes the EWC loss term to keep the model close to old_params.
        """
        if self.fisher_diagonal is None or self.old_params is None:
            return 0.0

        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_diagonal:
                loss += (self.fisher_diagonal[n] * (p - self.old_params[n]) ** 2).sum()
        return lamda * loss


# ==========================
# Streaming Dataset
# ==========================
class StreamTextDataset(IterableDataset):
    """
    Streams data from 'file_path' line by line. 
    Splits into train or val on-the-fly using a random threshold 
    each line. This means each epoch has a different split.

    - file_path (str): path to text file
    - vocab (dict): char->token mapping
    - seq_len (int): chunk size
    - split (str): 'train' or 'val'
    - split_ratio (float): fraction of lines that go to 'train'
    - seed (int): for reproducibility
    """
    def __init__(self, file_path, vocab, seq_len=128,
                 split='train', split_ratio=0.9, seed=42):
        super().__init__()
        self.file_path = file_path
        self.vocab = vocab
        self.seq_len = seq_len
        self.split = split
        self.split_ratio = split_ratio
        self.rng = random.Random(seed)

    def __iter__(self):
        buffer = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                prob = self.rng.random()
                if self.split == 'train' and prob < self.split_ratio:
                    for ch in line:
                        buffer.append(self.vocab.get(ch, 1))  # 1 is <UNK>
                        if len(buffer) == self.seq_len:
                            yield torch.tensor(buffer, dtype=torch.long)
                            buffer = []
                elif self.split == 'val' and prob >= self.split_ratio:
                    for ch in line:
                        buffer.append(self.vocab.get(ch, 1))
                        if len(buffer) == self.seq_len:
                            yield torch.tensor(buffer, dtype=torch.long)
                            buffer = []
        # leftover tokens in buffer are discarded if < seq_len


# ==========================
# Utilities
# ==========================
def merge_vocab(old_vocab, new_vocab):
    merged_vocab = dict(old_vocab)  # shallow copy
    for token, idx in new_vocab.items():
        if token not in merged_vocab:
            merged_vocab[token] = len(merged_vocab)
    return merged_vocab


def expand_model_vocab(model, old_vocab_size, new_vocab_size):
    if new_vocab_size == old_vocab_size:
        return model

    # Expand/truncate embedding
    new_embedding = nn.Embedding(new_vocab_size, model.d_model)
    copy_size = min(old_vocab_size, new_vocab_size)
    new_embedding.weight.data[:copy_size] = model.embedding.weight.data[:copy_size]
    model.embedding = new_embedding

    # Expand/truncate output layer
    new_output = nn.Linear(model.d_model, new_vocab_size)
    new_output.weight.data[:copy_size] = model.output.weight.data[:copy_size]
    new_output.bias.data[:copy_size] = model.output.bias.data[:copy_size]
    model.output = new_output

    # Update model vocab_size
    model.vocab_size = new_vocab_size
    return model


def build_vocab(text, vocab_size):
    from collections import Counter
    counter = Counter(text)
    vocab = {char: i + 2 for i, (char, _) in enumerate(counter.most_common(vocab_size - 2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<STOP>'] = len(vocab)
    return vocab


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================
# Saving / Loading Models
# ==========================
def save_model(model, vocab, model_name="hyena_model"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'hyperparameters': {
            'd_model': model.d_model,
            'n_layers': model.n_layers,
            'vocab_size': len(vocab)
        }
    }, f"{model_name}.pth")
    print(f"Model saved as {model_name}.pth")


def load_model(model_name="hyena_model"):
    ckpt_path = f"{model_name}.pth"
    if not os.path.exists(ckpt_path):
        print(f"No saved model found with name {ckpt_path}")
        return None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device)
    vocab = checkpoint.get('vocab')
    hyperparams = checkpoint.get('hyperparameters', {})

    d_model = hyperparams.get('d_model', 64)
    n_layers = hyperparams.get('n_layers', 10)
    vocab_size = hyperparams.get('vocab_size', len(vocab) if vocab else 1000)

    model = HyenaWithEWC(vocab_size, d_model, n_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {ckpt_path}")
    return model, vocab, hyperparams


# ==========================
# Training with EWC
# ==========================
def train_model_ewc(model, vocab, train_loader, val_loader, 
                    epochs=10, learning_rate=0.001, 
                    early_stopping_patience=3, ewc_lambda=15,
                    steps_per_epoch=1000, val_steps=200):
    """
    Train model using EWC on streaming data.
    We wrap the training loop in a try/except so that if the user
    presses Ctrl+C, we gracefully finish the epoch and return.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0

    if model.old_params is None:
        model.calculate_fisher(train_loader.dataset, device, samples=2000)

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0

        try:
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for step, batch in enumerate(train_loader):
                    if step >= steps_per_epoch:
                        break
                    batch = batch.to(device)
                    optimizer.zero_grad()

                    output = model(batch[:, :-1])
                    loss = criterion(output.view(-1, output.size(-1)),
                                     batch[:, 1:].contiguous().view(-1))

                    reg = model.ewc_loss(lamda=ewc_lambda)
                    total_loss = loss + reg
                    total_loss.backward()
                    optimizer.step()

                    train_loss_sum += total_loss.item()
                    pbar.set_postfix({'loss': total_loss.item()})
                    pbar.update(1)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C). Stopping early...")
            break

        avg_train_loss = train_loss_sum / max(1, steps_per_epoch)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            step_count = 0
            for step, batch in enumerate(val_loader):
                if step >= val_steps:
                    break
                step_count += 1
                batch = batch.to(device)
                output = model(batch[:, :-1])
                val_loss = criterion(output.view(-1, output.size(-1)),
                                     batch[:, 1:].contiguous().view(-1))
                val_loss_sum += val_loss.item()
        avg_val_loss = val_loss_sum / max(1, step_count)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_model(model, vocab, "best_model_ewc")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


def update_model_ewc(model, old_vocab, new_text_file, seq_len=128, 
                     batch_size=32, epochs=5, learning_rate=0.0001, 
                     ewc_lambda=15, steps_per_epoch=1000, val_steps=200):
    """
    Continues training the existing model on text found in 'new_text_file' 
    while retaining old knowledge via EWC.

    Steps:
    1. Save the model's current (original) state
    2. Revert model to that original state (so shapes match the old checkpoint)
    3. Build partial vocab from new_text_file
    4. Expand model to new vocab size
    5. Calculate fisher with old_params
    6. Train
    """
    import copy

    # -----------------------------
    # 1) Save the original state
    # -----------------------------
    original_state = copy.deepcopy(model.state_dict())

    # -----------------------------
    # 2) Revert to original state immediately
    #    so shape = old checkpoint
    # -----------------------------
    model.load_state_dict(original_state)
    model.old_params = original_state  # keep reference for EWC

    # -----------------------------
    # 3) Read partial lines from new_text_file to build updated vocab
    # -----------------------------
    N = 50000  # partial line limit
    temp_lines = []
    with open(new_text_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            temp_lines.append(line)
    sample_for_vocab = "".join(temp_lines)

    # Build new sub-vocab from partial lines
    new_vocab_size_est = len(old_vocab) + 100
    vocab_update = build_vocab(sample_for_vocab, new_vocab_size_est)

    # Merge with old vocab
    merged_vocab = merge_vocab(old_vocab, vocab_update)
    final_vocab_size = len(merged_vocab)

    # -----------------------------
    # 4) Expand model if needed
    # -----------------------------
    model = expand_model_vocab(model, len(old_vocab), final_vocab_size)

    # -----------------------------
    # 5) Calculate fisher
    #    - Build streaming dataset from new_text_file
    #    - We already reverted to original weights
    # -----------------------------
    from torch.utils.data import DataLoader
    new_dataset = StreamTextDataset(new_text_file, merged_vocab,
                                    seq_len=seq_len, split='train', split_ratio=1.0)
    new_loader = DataLoader(new_dataset, batch_size=batch_size)

    model.calculate_fisher(new_loader.dataset, device, samples=2000)

    # -----------------------------
    # 6) Train with EWC
    # -----------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        try:
            with tqdm(total=steps_per_epoch, desc=f"Update Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
                for step, batch in enumerate(new_loader):
                    if step >= steps_per_epoch:
                        break

                    batch = batch.to(device)
                    optimizer.zero_grad()

                    output = model(batch[:, :-1])
                    loss = criterion(output.view(-1, output.size(-1)),
                                     batch[:, 1:].contiguous().view(-1))

                    reg = model.ewc_loss(lamda=ewc_lambda)
                    total_loss = loss + reg
                    total_loss.backward()
                    optimizer.step()

                    train_loss_sum += total_loss.item()
                    pbar.set_postfix({'loss': total_loss.item()})
                    pbar.update(1)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (Ctrl+C). Stopping early...")
            break

        avg_loss = train_loss_sum / max(1, steps_per_epoch)
        print(f"[Update] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, merged_vocab



# ==========================
# Inference
# ==========================
def inference(model, vocab, prompt, seq_len, max_generated=100, temperature=1.0, k=50):
    """
    Generate text from the model, given a prompt.
    - temperature: float controlling randomness
    - k: top-k sampling
    """
    model.to(device)
    model.eval()

    inv_vocab = {v: k for k, v in vocab.items()}
    input_seq = [vocab.get(ch, 1) for ch in prompt[-seq_len:]]
    input_seq = torch.tensor([input_seq], dtype=torch.long, device=device)

    generated_text = prompt

    for _ in range(max_generated):
        with torch.no_grad():
            logits = model(input_seq)
            # Apply temperature + top-k
            probs = F.softmax(logits[0, -1] / temperature, dim=0)
            topk_vals, topk_inds = torch.topk(probs, k)
            topk_probs = topk_vals / torch.sum(topk_vals)
            next_token = topk_inds[torch.multinomial(topk_probs, 1)].item()

        if '<STOP>' in vocab and next_token == vocab['<STOP>']:
            print("Stop token generated. Terminating sequence.")
            break

        next_char = inv_vocab.get(next_token, '<UNK>')
        generated_text += next_char

        # Slide input_seq left by 1
        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_token]], device=device)], dim=1)

    return generated_text


# ==========================
# Interactive Console
# ==========================
def train_new_model():
    """
    Train a new model from scratch, prompting for required inputs.
    """
    model_name = input("Enter model name to save (e.g. my_model) [default: hyena_model]: ") or "hyena_model"

    data_file = input("Enter the path to the text file (default: random_text.txt): ") or "random_text.txt"
    # build vocab from partial data
    N = 100000
    lines_read = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= N:
                break
            lines_read.append(line)
    text_for_vocab = "".join(lines_read)

    vocab_size = int(input("Enter vocabulary size (default: 1000): ") or "1000")
    d_model = int(input("Enter d_model size (default: 64): ") or "64")
    n_layers = int(input("Enter number of layers (default: 10): ") or "10")
    seq_len = int(input("Enter sequence length (default: 128): ") or "128")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    learning_rate = float(input("Enter learning rate (default: 0.001): ") or "0.001")
    epochs = int(input("Enter number of epochs (default: 10): ") or "10")
    ewc_lambda = float(input("Enter EWC lambda value (default: 15): ") or "15")

    steps_per_epoch = int(input("Enter steps per epoch (default: 1000): ") or "1000")
    val_steps = int(input("Enter val steps per epoch (default: 200): ") or "200")
    early_stopping_patience = int(input("Enter early stopping patience (default: 3): ") or "3")

    # build vocab
    vocab = build_vocab(text_for_vocab, vocab_size)

    # create streaming datasets
    train_dataset = StreamTextDataset(data_file, vocab, seq_len=seq_len, split='train', split_ratio=0.9, seed=0)
    val_dataset = StreamTextDataset(data_file, vocab, seq_len=seq_len, split='val', split_ratio=0.9, seed=0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = HyenaWithEWC(len(vocab), d_model, n_layers)

    train_model_ewc(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        ewc_lambda=ewc_lambda,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps
    )

    # After training, save to disk
    save_model(model, vocab, model_name)
    print("Training new model complete!")
    return model, vocab, seq_len


def continue_training_existing():
    """
    Continue training an existing model on new text.
    NOTE: Now uses a streaming approach to avoid loading entire new file into memory.
    """
    model_path = input("Enter the path (without .pth) to the existing model: ")
    if not model_path:
        print("No model path provided.")
        return None, None

    model, vocab, _ = load_model(model_path)
    if model is None:
        print("Failed to load model.")
        return None, None

    new_text_file = input("Enter the path to the new text file to continue training on: ")
    if not os.path.exists(new_text_file):
        print("New text file not found.")
        return None, None

    # IMPORTANT FIX: we pass the file path directly to update_model_ewc
    # so it never tries to read the entire file as a single string
    seq_len = int(input("Enter sequence length (default: 128): ") or "128")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    learning_rate = float(input("Enter learning rate (default: 0.0001): ") or "0.0001")
    epochs = int(input("Enter number of epochs (default: 5): ") or "5")
    ewc_lambda = float(input("Enter EWC lambda value (default: 15): ") or "15")
    steps_per_epoch = int(input("Enter steps per epoch (default: 1000): ") or "1000")
    val_steps = int(input("Enter val steps per epoch (default: 200, but not used here): ") or "200")

    # Instead of reading entire new_text_file, we just pass the path
    # because update_model_ewc now does partial read for vocab & streaming training
    model, vocab = update_model_ewc(
        model, 
        vocab, 
        new_text_file,   # pass the file path, not the content
        seq_len=seq_len,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        ewc_lambda=ewc_lambda,
        steps_per_epoch=steps_per_epoch,
        val_steps=val_steps
    )

    # save updated model
    updated_model_name = input("Enter name for the updated model (default: updated_model): ") or "updated_model"
    save_model(model, vocab, updated_model_name)
    print("Continue training complete!")
    return model, vocab


def load_and_inference():
    """
    Load an existing model and do interactive inference.
    """
    model_path = input("Enter the path (without .pth) to the model for inference: ")
    if not model_path:
        print("No model path provided.")
        return

    model, vocab, _ = load_model(model_path)
    if model is None:
        print("Failed to load model.")
        return

    seq_len = model.d_model  # or you can prompt for an actual seq_len
    prompt = input("Enter a prompt for inference: ")
    max_generated = int(input("Enter max characters to generate (default: 100): ") or "100")
    temperature = float(input("Enter temperature (default: 1.0): ") or "1.0")
    top_k = int(input("Enter top-k (default: 50): ") or "50")

    generated_text = inference(
        model,
        vocab,
        prompt,
        seq_len=seq_len,
        max_generated=max_generated,
        temperature=temperature,
        k=top_k
    )
    print("Generated text:", generated_text)


def main():
    """
    Interactive console menu for:
      1) Train new model
      2) Continue training
      3) Load model & inference
      4) Exit
    """
    while True:
        print("\n==== Hyena Model Console ====")
        print("1) Train a new model")
        print("2) Continue training an existing model")
        print("3) Load a model and do inference")
        print("4) Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            train_new_model()
        elif choice == "2":
            continue_training_existing()
        elif choice == "3":
            load_and_inference()
        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
