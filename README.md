# Named-Entity-Recognition-NER-with-BiLSTM-and-BERT
This project implements and compares two different deep learning models for Named Entity Recognition (NER) on the CoNLL-2003 dataset.
NER is a fundamental task in Natural Language Processing (NLP) that involves identifying and classifying named entities in text into pre-defined categories such as persons, organizations, locations, and miscellaneous entities.\
This notebook demonstrates two popular approaches:\
A Bidirectional LSTM (BiLSTM) Model: A classic recurrent neural network (RNN) architecture built from scratch using PyTorch. This model learns to understand the context of words in a sequence to predict entity tags.
A Fine-Tuned BERT Model: A modern, transformer-based model using the Hugging Face transformers library. This approach leverages a large pre-trained language model (bert-base-uncased) and fine-tunes it on the specific NER task, often achieving state-of-the-art performance.\
Dataset\
The project uses the CoNLL-2003 dataset, a standard benchmark for NER.\
Below is a detailed breakdown of each script in the Jupyter Notebook, explaining its purpose and functionality.
Part 1: Initial Setup and Data Exploration\
Script 1: Environment Setup\
What it does: This is a standard Kaggle setup cell. It imports essential libraries like numpy for numerical operations and pandas for data manipulation. It also includes code to list the files in the input directory, which is common in Kaggle environments.
How it works: It loads the necessary packages for the project to begin.\
Script 2: Install Dependencies\
What it does: Installs the datasets library from Hugging Face.
Why it's needed: This library provides a very convenient way to download, load, and process many publicly available datasets, including CoNLL-2003, without needing to handle manual downloads and parsing.\
Script 3: Load the CoNLL-2003 Dataset\
What it does: Uses the load_dataset function from the datasets library to fetch the CoNLL-2003 dataset. It then prints the first example from the training set to show the data's structure.
How it works: The dataset is loaded as a DatasetDict containing train, validation, and test splits. Each item contains tokens (words) and their corresponding ner_tags, pos_tags, and chunk_tags as numerical IDs.\
Script 4: Visualize Data with Pandas\
What it does: Converts the training dataset into a pandas DataFrame for a clean, tabular view.
Why it's useful: This step is great for initial data exploration. It allows you to quickly see the format of the data, the number of entries, and examples of sentences and their corresponding tags.\
Part 2: Model 1 - Bidirectional LSTM (BiLSTM) from Scratch\
This section covers building, training, and evaluating a BiLSTM model for NER using PyTorch.\
Script 5: Data Preprocessing for BiLSTM\
What it does: Prepares the data for the BiLSTM model. This involves two key steps:
Extracting sentences (tokens) and their corresponding labels (ner_tags).
Creating vocabulary mappings: word2idx (maps each unique word to an integer) and label2idx (maps each NER tag to an integer).
How it works: A special <PAD> token (for padding) and <UNK> token (for unknown words) are added to the vocabulary. Neural networks work with numbers, so these mappings are essential to convert text data into a format the model can process.\
Script 6: BiLSTM Model Definition\
What it does: Defines the architecture of the BiLSTMNERModel using PyTorch's nn.Module.
Model Layers:
nn.Embedding: Converts the integer indices of words into dense vector representations (embeddings).
nn.LSTM: The core of the model. It's set to bidirectional=True, meaning it processes the sequence from left-to-right and right-to-left, capturing context from both directions.
nn.Linear: A fully connected layer that maps the LSTM's output to the number of NER tags, producing a score for each tag.
log_softmax: Converts the final scores into a log probability distribution.\
Script 7: PyTorch Dataset and DataLoader\
What it does: Sets up the data pipeline for training.
NERDataset: A custom PyTorch Dataset class that takes sentences and labels and returns them as tensors of indices.
collate_fn: A function to pad sequences within a batch to the same length. This is crucial because sentences have varying lengths. Labels for padded tokens are set to -100, a value that the loss function will ignore.
DataLoader: Creates iterable data loaders for the training and validation sets, which handle batching, shuffling, and applying the collate_fn.\
Script 8: Training the BiLSTM Model\
What it does: This is a comprehensive script that ties everything together for the BiLSTM model. It defines the complete training and evaluation workflow.
Key Components:
Device Setup: Checks if a CUDA-enabled GPU is available and sets the device accordingly to accelerate training.
EarlyStopping Class: A utility class to monitor the validation loss and stop training if it doesn't improve for a set number of epochs (patience). This helps prevent overfitting.
evaluate_model function: Calculates the model's performance (loss and F1-score) on the validation set.
train_model function: Contains the main training loop. For each epoch, it iterates through the training data, computes the loss (CrossEntropyLoss), performs backpropagation, and updates the model's weights (Adam optimizer). After each epoch, it evaluates the model and checks for early stopping.
Initialization and Training: The script initializes the model, data loaders, and kicks off the training process.\
Script 9: Saving and Loading the BiLSTM Model\
What it does: Provides helper functions to save the trained model's state (state_dict) to a file and to load it back later for inference.
Why it's important: This allows you to reuse a trained model without having to retrain it every time.\
Script 10: Prediction with the BiLSTM Model\
What it does: Defines a predict_with_model function to perform NER on a new sentence.
How it works: The function takes a sentence, tokenizes it, converts it to tensor indices using the saved word2idx mapping, feeds it through the loaded model, and returns the words paired with their predicted NER tags.\
Part 3: Model 2 - Fine-Tuning BERT\
This section uses the powerful Hugging Face transformers library to fine-tune a pre-trained BERT model for the NER task.\
Script 11: BERT Model Fine-Tuning\
What it does: This is a complete script for setting up, fine-tuning, and saving a BERT model for token classification.
Step-by-Step Process:
Load Tokenizer: Loads BertTokenizerFast from the bert-base-uncased model. This tokenizer uses a subword tokenization strategy (WordPiece).
Align Labels: Defines a function tokenize_and_align_labels. Since BERT breaks words into subwords (e.g., "Hossain" -> "ho", "##ssa", "##in"), this function ensures that the NER label is assigned only to the first subword of a token. Other subwords get a label of -100 so they are ignored by the loss function.
Load Model: Loads BertForTokenClassification, which is the BERT base model with a token classification head on top.
Training Arguments: Defines TrainingArguments to control the fine-tuning process (e.g., learning rate, batch size, number of epochs, evaluation strategy).
Metrics: The compute_metrics function calculates precision, recall, and F1-score for evaluation.
Trainer: Sets up the Hugging Face Trainer, a high-level API that handles the entire training and evaluation loop. It includes an EarlyStoppingCallback to prevent overfitting.
Train and Save: Calls trainer.train() to start fine-tuning. After training, it saves the fine-tuned model and tokenizer for later use.\
Script 12: Prediction with the Fine-Tuned BERT Model\
What it does: Defines functions for making predictions with the fine-tuned BERT model and providing clear explanations for the predicted tags.\
Key Functions:\
explain_ner_tag: A simple helper function that maps an NER tag like B-PER to a human-readable string like "Beginning of a Person entity".
predict_with_model: Takes a sentence, tokenizes it using the BERT tokenizer, runs it through the fine-tuned model, and returns each token with its predicted tag and a clear explanation.\
Script 13: Redundant BiLSTM Training Function\
What it does: This cell contains a simplified training function for the BiLSTM model.\
Note: This script is a more basic version of the training loop found in Script 8. It lacks a validation loop and early stopping. For a complete and robust training process, Script 8 should be considered the primary training script for the BiLSTM model.
