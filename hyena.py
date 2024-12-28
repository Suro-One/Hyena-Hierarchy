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
print("CUDA: ",torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyena Hierarchy with Elastic Weight Consolidation to prevent catastrophic forgetting
# Multi GPU
class HyenaWithEWC(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, dim_feedforward=20480, dropout=0.1):
        super(HyenaWithEWC, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                # 'conv': nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                'conv': nn.Conv1d(
                        in_channels=d_model,        # Adjust based on input data (e.g., number of features in signal)
                        out_channels=d_model,      # Start with higher output channels for better learning capacity
                        kernel_size=4,         # Optimal for quick learning with a balance of detail/context
                        stride=1,              # Small stride for detailed feature extraction
                        padding='same',        # Maintains input size without altering data too much
                        dilation=1,            # Standard dilation; no need for expansion in early stages
                        groups=1,              # No channel grouping; standard convolution
                        bias=True,             # Enable bias for better flexibility in learning
                        padding_mode='reflect' # Keeps border information intact for smoother learning
                    ),
                'gate': nn.Linear(d_model, d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.ReLU(),
                    nn.Linear(dim_feedforward, d_model),
                ),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(n_layers)
        ])
        
        self.output = nn.Linear(d_model, vocab_size)
        
        # EWC-specific attributes
        self.old_params = None
        self.fisher_diagonal = None

    def forward(self, src):
        src = self.embedding(src)
        for layer in self.layers:
            src = layer['conv'](src.transpose(1, 2)).transpose(1, 2)
            gate = torch.sigmoid(layer['gate'](src))
            src = src * gate
            src = layer['ffn'](src)
            src = layer['dropout'](src)
        return self.output(src)

    # def calculate_fisher(self, dataset, device, samples=200):
    def calculate_fisher(self, dataset, device, samples=20000):
        self.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        
        for _ in range(samples):
            idx = torch.randint(0, len(dataset), (1,))
            data = dataset[idx[0]].unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            self.zero_grad()
            output = self(data[:, :-1])
            loss = F.cross_entropy(output.view(-1, output.size(-1)), data[:, 1:].contiguous().view(-1))
            loss.backward()
            
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / samples
        
        self.fisher_diagonal = fisher
        self.old_params = copy.deepcopy(self.state_dict())

# lambda: memory importance
# So in simple terms, ewc_loss is like a "memory-preservation spell" that the NN uses to avoid overwriting important parts of its brain while learning something new.
    def ewc_loss(self, lamda=15):
        loss = 0
        for n, p in self.named_parameters():
            if n in self.fisher_diagonal:
                loss += (self.fisher_diagonal[n] * (p - self.old_params[n]) ** 2).sum()
        return lamda * loss

class   TextDataset(Dataset):
    def __init__(self, text, seq_len, vocab):
        self.text = text
        self.seq_len = seq_len
        self.vocab = vocab
        self.data = self.prepare_data()

    def prepare_data(self):
        tokens = [self.vocab.get(char, 1) for char in self.text]  # 1 is <UNK>
        return [tokens[i:i+self.seq_len] for i in range(0, len(tokens)-self.seq_len, self.seq_len)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long, device=device)

def prepare_data_splits(text, seq_len, vocab, validation_split=0.1):
    dataset = TextDataset(text, seq_len, vocab)
    train_size = int((1 - validation_split) * len(dataset))
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    return train_data, val_data

def build_vocab(text, vocab_size):
    counter = Counter(text)
    vocab = {char: i+2 for i, (char, _) in enumerate(counter.most_common(vocab_size-2))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    vocab['<STOP>'] = len(vocab)  # Add a stop token
    return vocab

def save_model(model, vocab, model_name):
    if not model_name:
        model_name = "hyena_model"
    torch.save({
        'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
        'vocab': vocab,
        'hyperparameters': {
            'd_model': model.module.d_model if isinstance(model, nn.DataParallel) else model.d_model,
            'n_layers': model.module.n_layers if isinstance(model, nn.DataParallel) else model.n_layers,
            'vocab_size': len(vocab)
        }
    }, f"{model_name}.pth")
    print(f"Model saved as {model_name}")

def load_model(model_name=None):
    if model_name is None or model_name == '':
        model_name = "hyena_model"
    
    if not os.path.exists(f"{model_name}"):
        print(f"No saved model found with name {model_name}")
        return None, None, None, None

    checkpoint = torch.load(f"{model_name}")
    
    vocab = checkpoint.get('vocab')
    hyperparameters = checkpoint.get('hyperparameters', {})
    
    d_model = hyperparameters.get('d_model', 64)
    n_layers = hyperparameters.get('n_layers', 2)
    vocab_size = hyperparameters.get('vocab_size', len(vocab) if vocab else 1000)

    model = HyenaWithEWC(vocab_size, d_model, n_layers)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_name}")
    return model, vocab, hyperparameters, d_model

def train_model_ewc(model, vocab, train_loader, val_loader, epochs, learning_rate, device, early_stopping_patience=3, ewc_lambda=15):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Calculate Fisher Information Matrix before training
    model.calculate_fisher(train_loader.dataset, device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), batch[:, 1:].contiguous().view(-1))
                
                # Add EWC loss
                ewc_loss = model.ewc_loss(lamda=ewc_lambda)
                total_loss = loss + ewc_loss
                
                total_loss.backward()
                optimizer.step()
                total_loss += total_loss.item()

                # Update progress bar
                pbar.set_postfix({'loss': total_loss.item()})
                pbar.update(1)
        avg_train_loss = total_loss / len(train_loader)

        # Validate the model
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), batch[:, 1:].contiguous().view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            save_model(model, vocab, "best_model_ewc")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

def update_model_ewc(model, old_vocab, new_text, seq_len, device, batch_size=32, epochs=10, learning_rate=0.0001, ewc_lambda=15):
    # Store the original model state
    original_state = copy.deepcopy(model.state_dict())

    # Build new vocab from new text
    new_vocab = build_vocab(new_text, len(old_vocab) + 100)
    
    # Merge old and new vocab
    merged_vocab = merge_vocab(old_vocab, new_vocab)
    
    # Expand the model's embedding and output layers to the new vocabulary size
    model = expand_model_vocab(model, len(old_vocab), len(merged_vocab))
    
    # Prepare the new dataset and dataloader
    dataset = TextDataset(new_text, seq_len, merged_vocab)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate Fisher Information Matrix for the original model
    model.calculate_fisher(dataset, device)
    
    # Set the old_params to the original model state
    model.old_params = original_state
    
    # Continue training with EWC
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = model(batch[:, :-1])
            loss = criterion(output.view(-1, output.size(-1)), batch[:, 1:].contiguous().view(-1))
            
            # Add EWC loss
            ewc_loss = model.ewc_loss(lamda=ewc_lambda)
            total_loss = loss + ewc_loss
            
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")
    
    return model, merged_vocab

# def update_model_ewc(model, old_vocab, new_text, seq_len, device, early_stopping_patience, batch_size=32, epochs=10, learning_rate=0.0001, ewc_lambda=15, stop_los=0.1):
#     # Build new vocab from new text
#     new_vocab_size = len(old_vocab) + 100 if old_vocab else 1000
#     new_vocab = build_vocab(new_text, new_vocab_size)
    
#     # Merge old and new vocab
#     merged_vocab = merge_vocab(old_vocab, new_vocab)
    
#     # Expand the model's embedding and output layers to the new vocabulary size
#     model = expand_model_vocab(model, len(old_vocab) if old_vocab else 0, len(merged_vocab))
    
#     # Prepare the new dataset and dataloader
#     dataset = TextDataset(new_text, seq_len, merged_vocab)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
#     # Continue training with EWC
#     train_model_ewc(model, merged_vocab, dataloader, dataloader, epochs, learning_rate, device, early_stopping_patience= early_stopping_patience, ewc_lambda=ewc_lambda)
    
#     return model, merged_vocab

def inference(model, vocab, text, seq_len, device, max_generated=100):
    model.to(device)  # Ensure model is on the right device
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    input_seq = torch.tensor([[vocab.get(char, 1) for char in text[-seq_len:]]]).to(device)  # Move input_seq to the device
    
    generated_text = text

    for _ in range(max_generated):
        with torch.no_grad():
            output = model(input_seq)
            probabilities = F.softmax(output[0, -1], dim=0)
            predicted_token = torch.argmax(probabilities).item()
            # TODO:

        if predicted_token == vocab['<STOP>']:
            print("Stop token generated. Terminating sequence generation.")
            break

        predicted_char = inv_vocab.get(predicted_token, '<UNK>')
        generated_text += predicted_char

        input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_token]]).to(device)], dim=1)  # Ensure new token is on the device

    return generated_text


def merge_vocab(old_vocab, new_vocab):
    merged_vocab = old_vocab.copy() if old_vocab else {}
    for token, idx in new_vocab.items():
        if token not in merged_vocab:
            merged_vocab[token] = len(merged_vocab)
    return merged_vocab

def expand_model_vocab(model, old_vocab_size, new_vocab_size):
    if new_vocab_size == old_vocab_size:
        return model
    
    # Expand or truncate embedding layer
    new_embedding = nn.Embedding(new_vocab_size, model.d_model)
    new_embedding.weight.data[:min(old_vocab_size, new_vocab_size)] = model.embedding.weight.data[:min(old_vocab_size, new_vocab_size)]
    model.embedding = new_embedding
    
    # Expand or truncate output layer
    new_output = nn.Linear(model.d_model, new_vocab_size)
    new_output.weight.data[:min(old_vocab_size, new_vocab_size)] = model.output.weight.data[:min(old_vocab_size, new_vocab_size)]
    new_output.bias.data[:min(old_vocab_size, new_vocab_size)] = model.output.bias.data[:min(old_vocab_size, new_vocab_size)]
    model.output = new_output
    
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train_model_with_ewc(model, vocab, train_loader, val_loader, epochs, learning_rate, device, experiment_folder, ewc_lambda=15, stop_loss=0.1, patience=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    model.calculate_fisher(train_loader.dataset, device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), batch[:, 1:].contiguous().view(-1))
                
                ewc_loss = model.ewc_loss(lamda=ewc_lambda)
                total_loss = loss + ewc_loss
                
                total_loss.backward()
                optimizer.step()
                
                pbar.set_postfix({'loss': total_loss.item()})
                pbar.update(1)

        # Validate the model
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch[:, :-1])
                loss = criterion(output.view(-1, output.size(-1)), batch[:, 1:].contiguous().view(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {avg_val_loss:.4f}")

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
                hyperparameters={'epochs': epochs, 'learning_rate': learning_rate, 'ewc_lambda': ewc_lambda}
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

# def save_experiment(experiment_name, model, optimizer, loss, epoch):
#     # Define what to save and how
#     experiment_data = {
#         'experiment_name': experiment_name,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#         'epoch': epoch
#     }
#     torch.save(experiment_data, f"{experiment_name}.pt")
def save_experiment(experiment_name, model, optimizer, loss, epoch, vocab, input_data=None, hyperparameters=None):
    # Create the base experiments directory if it doesn't exist
    base_dir = "experiments"
    os.makedirs(base_dir, exist_ok=True)

    # Create the experiment-specific directory
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
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return None, None, None, None, None, None

    checkpoint = torch.load(filepath)

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

def train_new_model(device):
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

    model, vocab = train_model_with_ewc(model, vocab, train_loader, val_loader, epochs, learning_rate, device, experiment_name, ewc_lambda, stop_loss, hyperparameters)
    return model, vocab

def load_existing_model():
    model_path = input("Enter the path to the saved model: ")
    model, optimizer, loss, epoch, vocab, hyperparameters = load_experiment(model_path)
    return model, vocab, hyperparameters

def continue_training(device):
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
    stop_loss = float(input("Enter loss threshold to stop training (default: 0.1): ") or "0.1")
    ewc_lambda = float(input("Enter EWC lambda value (default: 15): ") or "15")

    model, vocab = update_model_ewc(model, vocab, new_text, seq_len, device, batch_size, epochs, learning_rate, ewc_lambda, stop_loss)
    return model, vocab

def test_inference(model, d_model,  vocab, device):
    input_text = input("Enter the input text for inference: ")  + " "
    seq_len =  d_model # Use the model's sequence length
    max_generated = int(input("Enter maximum number of characters to generate: ") or "100")
    result = inference(model, vocab, input_text, seq_len, device, max_generated)
    print(f"Generated text: {result}")
def load_existing_model():
    model_path = input("Enter the path to the saved model: ")
    model, optimizer, loss, epoch, vocab, hyperparameters = load_experiment(model_path)
    return model, vocab, hyperparameters

def continue_training(device):
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
    stop_loss = float(input("Enter loss threshold to stop training (default: 0.1): ") or "0.1")
    ewc_lambda = float(input("Enter EWC lambda value (default: 15): ") or "15")

    model, vocab = update_model_ewc(model, vocab, new_text, seq_len, device, batch_size, epochs, learning_rate, ewc_lambda)
    return model, vocab

def test_inference(model, vocab, device):
    input_text = input("Enter the input text for inference: ") + " "
    seq_len = model.d_model  # Use the model's d_model as sequence length
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