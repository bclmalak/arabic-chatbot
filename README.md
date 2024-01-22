# Deep Learning - NLP Project: Chatbot design and implementation in the arabic language with a web-based interface

## Overview
This project involves the conceptualization and implementation of a chatbot using deep learning techniques and natural language processing (NLP). The primary components of the project include text classification using a neural network model and the development of an intelligent virtual assistant capable of understanding and responding in the Arabic language, through a web-based interface.

## Key Features
- **Data preprocessing:** Using NLP techniques involving data cleaning, tokenization, vectorization, padding and encoding.
- **Text Classification with Neural Networks:** Utilizes deep learning techniques, specifically neural networks, for text classification. This allows the chatbot to categorize and understand different types of user inputs.
- **Intelligent Virtual Assistant in Arabic:** Implements a virtual assistant capable of understanding and responding in the Arabic language. This involves leveraging NLP techniques for language understanding and generation.

## Model Architecture
**Layers:**
- Input Layer: Takes input data with a shape corresponding to the number of features in X.
- Embedding Layer: Maps input sequences into dense vectors of fixed size (output_dim=100).
- LSTM Layers: Three stacked LSTM layers (64 units each) to capture sequential patterns.
- LayerNormalization: Applied after each LSTM layer to normalize the activations.
- Dense Layers: Two fully connected layers with ReLU activation and layer normalization.
- Dropout Layers: Applied to prevent overfitting during training.
- Output Layer: Dense layer with softmax activation for multiclass classification.
**Compilation:** The model is compiled using the Adam optimizer and sparse categorical crossentropy loss function, with accuracy as the metric.

## Technologies Used
**Python:**
- **Tensorflow - Keras:** Frameworks for building and training neural network models. Tensorflow is a deep learning library, and Keras is a high-level neural networks API.
- **scikit-learn (sklearn):** Utilized for additional machine learning tasks and preprocessing.
- **Flask:** Used to develop a web-based interface for the chatbot, allowing users to interact with the virtual assistant.

## Dataset
The ARCD dataset (Arabic Reading Comprehension Dataset) is available in Hugging Face's Datasets library, which is an open-source library containing numerous datasets for research in automatic language processing.
