# Hyena Hierarchy

This code trains a character-level language model using a variant of a convolutional neural network architecture called Hyena. The trained model can be used to generate random text given a seed string.

### Requirements

To run this code, you need to have Python 3 and the following packages installed:

- torch

You can install these packages by running `pip install -r requirements.txt`.

### Usage

To train the model, run `train_hyena_model()` with the following parameters:

- `text_file`: Path to a text file used for training the model.
- `input_dim`: The number of characters to feed into the model at once.
- `output_dim`: The number of output classes (characters) of the model.
- `filter_size`: The size of the convolutional filters used in the model.
- `depth`: The number of convolutional filters in the model.
- `positional_dim`: The dimensionality of the output of the convolutional layer.
- `lr`: The learning rate used during optimization.
- `num_epochs`: The number of epochs to train the model.

The function returns the trained model, a list of characters in the vocabulary, and a dictionary that maps characters to their indices in the vocabulary.

To generate random text using the trained model, run `generate_text()` with the following parameters:

- `model`: The trained model.
- `seed_text`: The seed string used to start text generation.
- `length`: The length of the text to generate.
- `char_to_idx`: The dictionary that maps characters to their indices in the vocabulary.
- `idx_to_char`: The dictionary that maps indices in the vocabulary to characters.
- `vocab`: The number of characters in the vocabulary.

### Example

You can run the example code in `main()` to train the model on a randomly generated text and generate random text given a seed string. Note: The training code works, however the inference code needs work. Help would be appreciated.


## Hyena Model code overview

### Text Dataset

- $\text{TextDataset}(text, seq\_len)$: Initializes the text dataset with the given `text` and `seq_len`.
    - `text` (string): input text.
    - `seq_len` (int): sequence length.

- `__len__()`: Returns the length of the dataset.

- `__getitem__(index)`: Returns the tensor of the sequence and target at the given `index`.
    - `index` (int): index of the sequence.

### Hyena Model

- $\text{Hyena}(input\_dim, output\_dim, filter\_size, depth, positional\_dim)$: Initializes the Hyena model with the given parameters.
    - `input_dim` (int): input dimension.
    - `output_dim` (int): output dimension.
    - `filter_size` (int): filter size for convolution.
    - `depth` (int): depth of the model.
    - `positional_dim` (int): positional dimension of the model.

- `forward(x)`: Computes the forward pass of the Hyena model with the given input tensor `x`.
    - `x` (tensor): input tensor.

### Training Hyena Model

- `train_hyena_model(text_file, input_dim, output_dim, filter_size, depth, positional_dim, lr, num_epochs)`: Trains the Hyena model with the given parameters and returns the trained model, character list, and character-to-index dictionary.
    - `text_file` (string): input text file path.
    - `input_dim` (int): input dimension.
    - `output_dim` (int): output dimension.
    - `filter_size` (int): filter size for convolution.
    - `depth` (int): depth of the model.
    - `positional_dim` (int): positional dimension of the model.
    - `lr` (float): learning rate.
    - `num_epochs` (int): number of epochs.

### Text Generation

- `generate_text(model, seed_text, length, char_to_idx, idx_to_char, vocab)`: Generates text using the trained Hyena model with the given parameters.
    - `model` (Hyena): trained Hyena model.
    - `seed_text` (string): seed text.
    - `length` (int): length of generated text.
    - `char_to_idx` (dict): character-to-index dictionary.
    - `idx_to_char` (dict): index-to-character dictionary.
    - `vocab` (int): vocabulary size.
    - `input_dim` (int): input dimension

### Main Function

- `main()`: Runs the main function which generates random text, trains the Hyena model, and generates text using the trained model.


# Credits

This code is inspired by the papers 
## [A Convolutional Neural Network for Modelling Sentences"](https://arxiv.org/abs/1404.2188)
 by Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom.


## [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866.pdf)
by: Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Affiliations:
1. Department of Computer Science, Stanford University, Stanford, CA, USA
2. Mila - Quebec AI Institute and DIRO, Université de Montréal, Montréal, QC, Canada

