# Next_Word_Predictor
This repository contains a next word prediction model implemented using a Recurrent Neural Network (RNN) architecture, specifically Long Short-Term Memory (LSTM). The model is trained on text data to predict the next word or sequence of words in a sentence with high accuracy.
## Project Overview
The Next Word Predictor uses LSTM, a type of RNN, to predict the next word or series of words given a sequence of text. The model was trained on a large text corpus and achieved an accuracy of 97%.

## Model Architecture
The model utilizes LSTM layers to handle the sequential nature of text data. LSTM is well-suited for this task due to its ability to maintain information over long sequences, which is crucial for predicting the next word in a sentence.

Embedding Layer: Converts words into dense vectors of fixed size.
LSTM Layers: Captures dependencies between the words in the input sequence.
Dense Layer: Outputs the probability distribution over the vocabulary for the next word prediction.

## Data
The model was trained on a dataset of text sequences, preprocessed to remove noise and tokenized into a format suitable for training. The dataset was split into training and validation sets to evaluate the model's performance.

## Training
The model was trained using categorical cross-entropy loss and Adam optimizer. The training process involved tuning hyperparameters such as the number of LSTM units, batch size, and learning rate to achieve optimal performance.

Optimizer: Adam
Loss Function: Categorical Cross-Entropy
Accuracy Achieved: 97%

## Results
The model achieved a 97% accuracy in predicting the next word, demonstrating its effectiveness in understanding and generating natural language sequences.
