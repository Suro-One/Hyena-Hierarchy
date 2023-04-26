## README.md

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

# Credits

This code is inspired by the papers 
## [A Convolutional Neural Network for Modelling Sentences"](https://arxiv.org/abs/1404.2188)
 by Nal Kalchbrenner, Edward Grefenstette, and Phil Blunsom.


## [Hyena Hierarchy: Towards Larger Convolutional Language Models](https://arxiv.org/pdf/2302.10866.pdf)
by: Michael Poli, Stefano Massaroli, Eric Nguyen, Daniel Y. Fu, Tri Dao, Stephen Baccus, Yoshua Bengio, Stefano Ermon, Christopher Ré

Affiliations:
1. Department of Computer Science, Stanford University, Stanford, CA, USA
2. Mila - Quebec AI Institute and DIRO, Université de Montréal, Montréal, QC, Canada

