import uuid
import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
from collections import Counter
import copy
from sklearn.model_selection import train_test_split

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

        # Create Hyena-like layers (simplified conv + gating + FFN)
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
            x = layer['conv'](src.transpose(1, 2))      # (B, d_model, S)
            x = x.transpose(1, 2)                       # (B, S, d_model)

            # Gating
            gate = torch.sigmoid(layer['gate'](x))      # (B, S, d_model)
            x = x * gate

            # FFN + Dropout
            x = layer['ffn'](x)
            x = layer['dropout'](x)

            src = x

        return self.output(src)

    def calculate_fisher(self, dataset, device, samples=2000):
        """
        Calculates the diagonal of the Fisher Information Matrix.
        This helps EWC track which parameters are most important
        so they won't be overwritten during continued training.

        Parameters:
        - dataset: A TextDataset or any dataset that returns a single tensor.
        - samples: How many random samples to draw to approximate Fisher.
                   For large datasets, reduce this number to avoid huge overhead.
        """
        self.eval()

        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}

        # We randomly sample from the dataset to compute the Fisher
        # for memory constraints/performance.
        for _ in range(samples):
            idx = torch.randint(0, len(dataset), (1,))
            data = dataset[idx[0]].unsqueeze(0).to(device)  # shape: (1, seq_len)

            self.zero_grad()
            output = self(data[:, :-1])  # ignore the last token
            loss = F.cross_entropy(
                output.view(-1, output.size(-1)), 
                data[:, 1:].contiguous().view(-1)
            )
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += (p.grad.data ** 2) / samples

        self.fisher_diagonal = fisher
        self.old_params = copy.deepcopy(self.state_dict())

    def ewc_loss(self, lamda=15):
        """
        Computes the EWC loss term that keeps the model from deviating
        too much from previously learned weights.
        """
        if self.fisher_diagonal is None or self.old_params is None:
            return 0.0

        loss = 0.0
        for n, p in self.named_parameters():
            if n in self.fisher_diagonal:
                loss += (self.fisher_diagonal[n] * (p - self.old_params[n]) ** 2).sum()
        return lamda * loss


# ==========================
# Dataset / Data Prep
# ==========================
class TextDataset(Dataset):
    def __init__(self, text, seq_len, vocab):
        """
        text: raw text string
        seq_len: how many tokens in each sample
        vocab: dictionary mapping chars -> token IDs
        """
        self.text = text
        self.seq_len = seq_len
        self.vocab = vocab
        self.data = self.prepare_data()

    def prepare_data(self):
        tokens = [self.vocab.get(char, 1) for char in self.text]  # 1 is <UNK>
        samples = []
        # Step in increments of seq_len so each sample is exactly seq_len tokens
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            samples.append(tokens[i:i+self.seq_len])
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sequence of token IDs
        return torch.tensor(self.data[idx], dtype=torch.long)


def prepare_data_splits(text, seq_len, vocab, validation_split=0.1):
    dataset = TextDataset(text, seq_len, vocab)
    train_size = int((1 - validation_split) * len(dataset))
    train_data, val_data = torch.utils.data.random_split(
        dataset, 
        [train_size, len(dataset) - train_size]
    )
    return train_data, val_data


def build_vocab(text, vocab_size):
    """
    Builds a vocab from the most common characters in 'text',
    plus special tokens <PAD>, <UNK>, and <STOP>.
    """
    counter = Counter(text)
    # commonest chars => index from 2 up to vocab_size
    vocab = {char: i + 2 for i, (char, _) in enumerate(counter.most_common(vocab_size - 2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<STOP>'] = len(vocab)
    return vocab


# ==========================
# Model Save/Load
# ==========================
def save_model(model, vocab, model_name):
    """
    Save the model state and vocab. 
    """
    if not model_name:
        model_name = "hyena_model"
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
    """
    Load a saved model checkpoint from 'model_name.pth'.
    """
    ckpt_path = f"{model_name}.pth"
    if not os.path.exists(ckpt_path):
        print(f"No saved model found with name {ckpt_path}")
        return None, None, None

    checkpoint = torch.load(ckpt_path, map_location=device)

    vocab = checkpoint.get('vocab')
    hyperparams = checkpoint.get('hyperparameters', {})
    d_model = hyperparams.get('d_model', 64)
    n_layers = hyperparams.get('n_layers', 2)
    vocab_size = hyperparams.get('vocab_size', len(vocab) if vocab else 1000)

    model = HyenaWithEWC(vocab_size, d_model, n_layers)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {ckpt_path}")
    return model, vocab, hyperparams


# ==========================
# EWC Training
# ==========================
def train_model_ewc(model, vocab, train_loader, val_loader, 
                    epochs=10, learning_rate=0.001, 
                    device=device, 
                    early_stopping_patience=3, 
                    ewc_lambda=15):
    """
    Train model with EWC on a single dataset.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0

    # Calculate Fisher before training (this is optional if the model is new and has no old_params).
    # But if continuing training after a prior task, you'd typically do it after finishing previous training.
    if model.old_params is None:
        model.calculate_fisher(train_loader.dataset, device)

    for epoch in range(epochs):
        model.train()
        cumulative_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = model(batch[:, :-1])
                loss = criterion(
                    output.view(-1, output.size(-1)), 
                    batch[:, 1:].contiguous().view(-1)
                )

                # EWC regularization
                ewc_regularization = model.ewc_loss(lamda=ewc_lambda)
                total_loss = loss + ewc_regularization

                total_loss.backward()
                optimizer.step()

                cumulative_train_loss += total_loss.item()

                pbar.set_postfix({'loss': total_loss.item()})
                pbar.update(1)

        avg_train_loss = cumulative_train_loss / len(train_loader)

        # Validate the model
        model.eval()
        cumulative_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch[:, :-1])
                val_loss = criterion(
                    output.view(-1, output.size(-1)), 
                    batch[:, 1:].contiguous().view(-1)
                )
                cumulative_val_loss += val_loss.item()

        avg_val_loss = cumulative_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            save_model(model, vocab, "best_model_ewc")  # Save best so far
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


def update_model_ewc(model, old_vocab, new_text, seq_len, device=device, 
                     batch_size=32, epochs=5, learning_rate=0.0001, ewc_lambda=15):
    """
    Continue training the existing model on new_text while retaining old knowledge via EWC.
    1) Build new vocab from new text.
    2) Merge with old vocab.
    3) Expand model embedding and output if vocab size changed.
    4) Calculate fisher on old model parameters.
    5) EWC train on new_text.
    """
    # 1) Original state before expanding
    original_state = copy.deepcopy(model.state_dict())

    # 2) Build new vocab from new text
    new_vocab_size = len(old_vocab) + 100
    new_vocab = build_vocab(new_text, new_vocab_size)
    
    # 3) Merge old and new vocab
    merged_vocab = merge_vocab(old_vocab, new_vocab)
    new_vocab_size = len(merged_vocab)

    # 4) Expand model if needed
    model = expand_model_vocab(model, len(old_vocab), new_vocab_size)

    # Prepare new dataset
    dataset = TextDataset(new_text, seq_len, merged_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5) Calculate Fisher w.r.t. the original model params
    model.load_state_dict(original_state)  # ensure same old_params
    model.calculate_fisher(dataset, device)
    model.old_params = original_state      # store old model weights to compare to new updates

    # 6) Continue training
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        cumulative_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch[:, :-1])
            loss = criterion(
                output.view(-1, output.size(-1)), 
                batch[:, 1:].contiguous().view(-1)
            )

            ewc_reg = model.ewc_loss(lamda=ewc_lambda)
            total_loss = loss + ewc_reg

            total_loss.backward()
            optimizer.step()

            cumulative_loss += total_loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {cumulative_loss / len(dataloader):.4f}")

    return model, merged_vocab


# ==========================
# Generating Text
# ==========================
def inference(model, vocab, text, seq_len, device, max_generated=100):
    """
    Generate text from the model, given a prompt 'text'.
    """
    model.to(device)
    model.eval()

    inv_vocab = {v: k for k, v in vocab.items()}
    # Take the last seq_len tokens from the prompt
    input_seq = [vocab.get(char, 1) for char in text[-seq_len:]]
    input_seq = torch.tensor([input_seq], dtype=torch.long, device=device)

    generated_text = text

    for _ in range(max_generated):
        with torch.no_grad():
            output = model(input_seq)
            probabilities = F.softmax(output[0, -1], dim=0)
            predicted_token = torch.argmax(probabilities).item()

        # If model outputs <STOP>, break
        if '<STOP>' in vocab and predicted_token == vocab['<STOP>']:
            print("Stop token generated. Terminating sequence generation.")
            break

        predicted_char = inv_vocab.get(predicted_token, '<UNK>')
        generated_text += predicted_char

        # Shift input_seq left and append the predicted token
        input_seq = torch.cat([
            input_seq[:, 1:], 
            torch.tensor([[predicted_token]], device=device)
        ], dim=1)

    return generated_text


# ==========================
# Vocab Merging & Expansion
# ==========================
def merge_vocab(old_vocab, new_vocab):
    """
    Combine old_vocab with new_vocab, ensuring unique tokens are added.
    """
    merged_vocab = dict(old_vocab)  # make a shallow copy
    for token, idx in new_vocab.items():
        if token not in merged_vocab:
            merged_vocab[token] = len(merged_vocab)
    return merged_vocab


def expand_model_vocab(model, old_vocab_size, new_vocab_size):
    """
    If the new_vocab_size > old_vocab_size, expand the embedding
    and output layers to accommodate new tokens.
    """
    if new_vocab_size == old_vocab_size:
        return model

    # Expand or truncate embedding layer
    new_embedding = nn.Embedding(new_vocab_size, model.d_model)
    # Copy over existing weights
    copy_size = min(old_vocab_size, new_vocab_size)
    new_embedding.weight.data[:copy_size] = model.embedding.weight.data[:copy_size]
    model.embedding = new_embedding

    # Expand or truncate output layer
    new_output = nn.Linear(model.d_model, new_vocab_size)
    new_output.weight.data[:copy_size] = model.output.weight.data[:copy_size]
    new_output.bias.data[:copy_size] = model.output.bias.data[:copy_size]
    model.output = new_output

    # Update stored vocab size
    model.vocab_size = new_vocab_size

    return model


def count_parameters(model):
    """
    Utility to count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==========================
# Experiment Saving/Loading
# ==========================
def save_experiment(experiment_name, model, optimizer, loss, epoch, vocab,
                    input_data=None, hyperparameters=None):
    """
    Saves an experiment checkpoint in 'experiments/experiment_name/' dir.
    """
    base_dir = "experiments"
    os.makedirs(base_dir, exist_ok=True)

    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    experiment_data = {
        'experiment_name': experiment_name,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'epoch': epoch,
        'vocab': vocab,
        'd_model': model.d_model,
        'n_layers': model.n_layers,
        'vocab_size': model.vocab_size,
        'input_data': input_data,
        'hyperparameters': hyperparameters
    }

    save_path = os.path.join(experiment_dir, f"best_model_epoch_{epoch}.pt")
    torch.save(experiment_data, save_path)
    print(f"Experiment saved at {save_path}")


def load_experiment(filepath):
    """
    Loads an experiment checkpoint from 'filepath'.
    Returns: (model, optimizer, loss, epoch, vocab, hyperparams)
    """
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return None, None, None, None, None, None

    checkpoint = torch.load(filepath, map_location=device)

    vocab_size = checkpoint['vocab_size']
    d_model = checkpoint['d_model']
    n_layers = checkpoint['n_layers']

    model = HyenaWithEWC(vocab_size, d_model, n_layers)
    model.load_state_dict(checkpoint['model_state_dict'])

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loss = checkpoint['loss']
    epoch = checkpoint['epoch']
    vocab = checkpoint['vocab']
    hyperparameters = checkpoint.get('hyperparameters', None)

    print(f"Experiment '{checkpoint['experiment_name']}' loaded (Epoch {epoch}, Loss: {loss})")
    
    return model, optimizer, loss, epoch, vocab, hyperparameters


# ==========================
# Higher-level training loop
# ==========================
def train_model_with_ewc(model, vocab, train_loader, val_loader,
                         epochs=10, learning_rate=0.001, 
                         device=device, experiment_folder="default_exp",
                         ewc_lambda=15, stop_loss=0.1, patience=5):
    """
    Train model using EWC and save experiment checkpoints.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0

    # If there's no old_params, compute fisher once
    if model.old_params is None:
        model.calculate_fisher(train_loader.dataset, device)

    for epoch in range(epochs):
        model.train()
        cumulative_train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                output = model(batch[:, :-1])
                loss = criterion(
                    output.view(-1, output.size(-1)), 
                    batch[:, 1:].contiguous().view(-1)
                )
                reg = model.ewc_loss(lamda=ewc_lambda)
                total_loss = loss + reg
                total_loss.backward()
                optimizer.step()

                cumulative_train_loss += total_loss.item()
                pbar.set_postfix({'loss': total_loss.item()})
                pbar.update(1)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch[:, :-1])
                val_loss = criterion(
                    output.view(-1, output.size(-1)), 
                    batch[:, 1:].contiguous().view(-1)
                )
                val_loss_sum += val_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}")

        # Checkpointing (simple early stopping)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            save_experiment(
                experiment_name=experiment_folder,
                model=model,
                optimizer=optimizer,
                loss=avg_val_loss,
                epoch=epoch,
                vocab=vocab,
                hyperparameters={
                    'epochs': epochs, 
                    'learning_rate': learning_rate, 
                    'ewc_lambda': ewc_lambda
                }
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        if avg_val_loss < stop_loss:
            print("Training stopped as the loss threshold is met.")
            break

    return model, vocab


# ==========================
# Interactive Functions
# ==========================
def train_new_model(device):
    """
    Interactive function that asks for user input,
    trains a new model from scratch, and saves it.
    """
    experiment_name = input("Enter experiment name (leave blank for GUID): ") or str(uuid.uuid4())

    data_file = input("Enter the path to the text file (default: random_text.txt): ") or "random_text.txt"
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    vocab_size = int(input("Enter vocabulary size (default: 1000): ") or "1000")
    d_model = int(input("Enter d_model size (default: 64): ") or "64")
    n_layers = int(input("Enter number of layers (default: 2): ") or "2")
    seq_len = int(input("Enter sequence length (default: 128): ") or "128")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    learning_rate = float(input("Enter learning rate (default: 0.001): ") or "0.001")
    epochs = int(input("Enter number of epochs (default: 10): ") or "10")
    stop_loss = float(input("Enter loss threshold to stop training (default: 0.1): ") or "0.1")
    ewc_lambda = float(input("Enter EWC lambda value (default: 15): ") or "15")

    vocab = build_vocab(text, vocab_size)
    model = HyenaWithEWC(len(vocab), d_model, n_layers)

    dataset = TextDataset(text, seq_len, vocab)
    train_data, val_data = train_test_split(dataset, test_size=0.1)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    hyperparameters = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'n_layers': n_layers,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'stop_loss': stop_loss,
        'ewc_lambda': ewc_lambda
    }

    model, vocab = train_model_with_ewc(
        model=model,
        vocab=vocab,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        experiment_folder=experiment_name,
        ewc_lambda=ewc_lambda,
        stop_loss=stop_loss,
        patience=5
    )
    return model, vocab


def load_existing_model():
    """
    Interactive: prompts user for the path of a previously saved experiment.
    Returns (model, vocab, hyperparameters)
    """
    model_path = input("Enter the path to the saved model (e.g. experiments/xyz/best_model_epoch_4.pt): ")
    model, optimizer, loss, epoch, vocab, hyperparams = load_experiment(model_path)
    return model, vocab, hyperparams


def continue_training(device):
    """
    Interactive: loads an existing model, expands vocab if needed, and 
    continues training on a new text file (EWC).
    """
    model, vocab, hyperparameters = load_existing_model()
    if model is None:
        print("Failed to load the model. Returning to main menu.")
        return None, None

    data_file = input("Enter the path to the new text file for continued training: ")
    with open(data_file, "r", encoding="utf-8") as f:
        new_text = f.read()

    seq_len = int(input("Enter sequence length (default: 128): ") or "128")
    batch_size = int(input("Enter batch size (default: 32): ") or "32")
    learning_rate = float(input("Enter learning rate (default: 0.0001): ") or "0.0001")
    epochs = int(input("Enter number of epochs (default: 5): ") or "5")
    ewc_lambda = float(input("Enter EWC lambda value (default: 15): ") or "15")

    model, vocab = update_model_ewc(model, vocab, new_text, seq_len, device, 
                                    batch_size=batch_size, 
                                    epochs=epochs, 
                                    learning_rate=learning_rate, 
                                    ewc_lambda=ewc_lambda)
    return model, vocab


def test_inference(model, vocab, device):
    """
    Interactive: prompt the user for input text, run inference, and print result.
    """
    input_text = input("Enter the input text for inference: ") + " "
    seq_len = model.d_model  # Using the model's d_model as the sequence length (arbitrary choice)
    max_generated = int(input("Enter maximum number of characters to generate: ") or "100")
    result = inference(model, vocab, input_text, seq_len, device, max_generated)
    print(f"Generated text: {result}")


def main():
    model, vocab = None, None

    while True:
        print("\nHyena Model Training and Inference")
        print("1. Train a new model")
        print("2. Load an existing model")
        print("3. Continue training an existing model")
        print("4. Test inference")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            model, vocab = train_new_model(device)
        elif choice == '2':
            model, vocab, _ = load_existing_model()
        elif choice == '3':
            model, vocab = continue_training(device)
        elif choice == '4':
            if model is None or vocab is None:
                print("Please load or train a model first.")
            else:
                test_inference(model, vocab, device)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
