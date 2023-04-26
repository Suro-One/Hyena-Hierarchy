import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import string

class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, index):
        return torch.tensor(self.text[index:index+self.seq_len]), torch.tensor(self.text[index+self.seq_len])

class Hyena(nn.Module):
    def __init__(self, input_dim, output_dim, filter_size, depth, positional_dim):
        super(Hyena, self).__init__()
        self.depth = depth
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.positional_dim = positional_dim
        self.filter_size = filter_size

        self.linear1 = nn.Linear(input_dim, (output_dim + 1) * depth)
        self.conv1d = nn.Conv1d(depth, depth, filter_size, padding=filter_size // 2)
        self.linear2 = nn.Linear(depth * positional_dim, output_dim)

    def forward(self, x):
        x = x.float()  # Convert input tensor to float
        x = self.linear1(x)
        x = x.view(x.size(0), -1, self.depth).transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.depth * self.positional_dim)
        x = self.linear2(x)
        return x

def train_hyena_model(text_file, input_dim, output_dim, filter_size, depth, positional_dim, lr, num_epochs):
    text = open(text_file, 'r').read()
    text = text.lower()
    chars = list(set(text))
    vocab = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    dataset = TextDataset([char_to_idx[ch] for ch in text], input_dim)
    dataloader = DataLoader(dataset, batch_size=input_dim)

    model = Hyena(input_dim, len(chars), filter_size, depth, positional_dim)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        for seqs, targets in dataloader:
            model.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training completed.')

    return model, chars, char_to_idx


def generate_text(model, seed_text, length, char_to_idx, idx_to_char, vocab):
    model.eval()
    with torch.no_grad():
        seed_indices = torch.LongTensor([
            char_to_idx.get(c, random.randint(0, vocab - 1)) for c in seed_text.lower()
        ])
        out = []
        for i in range(length):
            outputs = model(seed_indices.float())
            probs = nn.functional.softmax(outputs[-1], dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            out.append(idx_to_char[next_idx])

    return out

def main():
    random_text = ''.join(
        random.choice(string.ascii_lowercase + string.digits + string.punctuation + ' ')
        for _ in range(1000)
    )
    with open('random_text.txt', 'w') as f:
        f.write(random_text)

    input_dim = 70
    output_dim = 64
    filter_size = 3
    depth = 3
    positional_dim = (input_dim - filter_size + 2 * (filter_size // 2)) // 1 + 1
    lr = 0.001
    num_epochs = 100

    model, vocab, char_to_idx = train_hyena_model(
        'random_text.txt', input_dim, output_dim, filter_size, depth, positional_dim, lr, num_epochs
    )

    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    seed_text = 'The quick brown fox'
    num_chars = 70
    generated_text = generate_text(model, seed_text, num_chars, char_to_idx, idx_to_char, len(vocab))
    
    print('Generated text: ' + ''.join(generated_text))

if __name__ == '__main__':
    main()