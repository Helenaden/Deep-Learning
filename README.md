# Deep-Learning Projects

This repository contains a series of hands-on deep learning projects, starting from foundational concepts like image classification with simple neural networks to advanced techniques such as transfer learning and natural language processing with BERT. Each project builds upon the previous one, offering a comprehensive learning path in applied deep learning.

# Table of Contents

1.  [Image Classification with the MNIST Dataset](#1-image-classification-with-the-mnist-dataset)
2.  [Image Classification of an American Sign Language Dataset](#2-image-classification-of-an-american-sign-language-dataset)
3.  [Convolutional Neural Networks](#3-convolutional-neural-networks)
4a.  [Data Augmentation](#4-data-augmentation)
4b.  [Deploying Your Model](#5-deploying-your-model)
5a.  [Pre-Trained Models](#6-pre-trained-models)
5b.  [Transfer Learning](#7-transfer-learning)
6.  [Natural Language Processing (NLP)](#8-natural-language-processing-nlp)
7.  [Fresh and Rotten Fruit Recognition](#9-fresh-and-rotten-fruit-recognition)

# 1. Image Classification with the MNIST Dataset

This section introduces the "Hello World" of deep learning: training a deep learning model to correctly classify hand-written digits. It's a foundational step to understand the basic workflow of a deep learning project.

# Objectives
* Understand how deep learning can solve problems that traditional programming methods cannot.
* Learn about the MNIST handwritten digits dataset.
* Use `torchvision` to load the MNIST dataset and prepare it for training.
* Create a simple neural network to perform image classification.
* Train the neural network using the prepped MNIST dataset.
* Observe the performance of the trained neural network.

# Key Concepts
* Basic Neural Networks (Dense/Fully Connected Layers)
* Image Classification Fundamentals
* Dataset Loading and Preparation (PyTorch `Dataset` and `DataLoader`)

# 2. Image Classification of an American Sign Language Dataset

Building on the MNIST experience, this project applies the data preparation, model creation, and model training steps to a new dataset: images of hands making letters in American Sign Language. This helps reinforce the general deep learning workflow.

# Objectives
* Prepare image data for training.
* Create and compile a simple model for image classification.
* Train an image classification model and observe the results.

# Key Concepts
* Applying a deep learning workflow to a new dataset.
* Basic Model Training Loop.

# 3. Convolutional Neural Networks

In the previous section, the simple model likely overfit the ASL dataset, performing well on training data but poorly on validation data. This section introduces Convolutional Neural Networks (CNNs), a powerful type of model specifically designed for image data, to address overfitting and improve generalization.

# Objectives
* Prepare image data specifically for a CNN.
* Create a more sophisticated CNN model, understanding a greater variety of model layers (e.g., `Conv2d`, `MaxPool2d`).
* Train a CNN model and observe its performance.

# Key Concepts
* Convolutional Neural Networks (CNNs)
* Overfitting and Generalization
* Convolutional Layers, Pooling Layers

# 4a. Data Augmentation

Even with a CNN, overfitting can still occur. This section tackles this by programmatically increasing the size and variance of the dataset through data augmentation. This technique helps the model become more robust and generalize better to unseen data.

# Objectives
* Augment the ASL dataset using various transformations.
* Use the augmented data to train an improved model.
* Save the well-trained model to disk for use in deployment.

# Key Concepts
* Data Augmentation (e.g., rotations, cropping, flipping, color jitter)
* Improving Model Robustness
* Saving and Loading Model Weights (`state_dict`)

# 4b. Deploying Your Model

With a well-trained model in hand, this exercise focuses on practical deployment. It demonstrates how to load a saved model and use it to perform inference on new, unseen images, evaluating its real-world performance.

# Objectives
* Load an already-trained model from disk.
* Reformat new images for a model trained on images of a different format.
* Perform inference with new images never seen by the trained model and evaluate its performance.

# Key Concepts
* Model Inference
* Deployment Considerations (Pre-processing for new data)

# 5a. Pre-Trained Models

This section explores the power of pre-trained models. These are models that have already been trained on vast datasets (like ImageNet) and can be used "out of the box" for various tasks, often achieving high accuracy without any further training.

# Objectives
* Use `TorchVision` to load a very well-trained pretrained model.
* Preprocess our own images to work with the pretrained model's expected input format.
* Use the pretrained model to perform accurate inference on your own images.

# Key Concepts
* Pre-trained Models
* Feature Extraction (using the backbone of a pre-trained model)
* `torchvision.models` and `torchvision.transforms` for pre-trained models

# 5b. Transfer Learning

When a suitable pre-trained model for your exact task isn't available, but you have a small dataset, transfer learning comes to the rescue. This technique involves taking a pre-trained model and retraining a portion of it on your specific task, leveraging the knowledge gained from the original training.

# Objectives
* Prepare a pretrained model for transfer learning (e.g., freezing layers).
* Perform transfer learning with your own small dataset on a pretrained model.
* Further fine-tune the model for even better performance (unfreezing more layers).

# Key Concepts
* Transfer Learning (Freezing/Unfreezing layers)
* Fine-tuning
* Training with Small Datasets

# 6. Natural Language Processing (NLP)

This tutorial takes a detour from image data to sequence data, specifically text. It explores how neural networks handle language, introducing concepts like tokenization, embeddings, and the groundbreaking BERT model.

# Objectives
* Use a tokenizer to prepare text for a neural network.
* See how embeddings are used to identify numerical features for text data.
* Understand the goals and see BERT in action (Masked Language Model and Next Sentence Prediction).

# Key Concepts
* Sequence Data
* Natural Language Processing (NLP)
* Tokenization
* Word Embeddings
* BERT (Bidirectional Encoder Representations from Transformers)

# 7. Fresh and Rotten Fruit Recognition

This is the culminating project where all the skills learned throughout the previous projects are applied. The goal is to train a model capable of recognizing fresh and rotten fruit, achieving a high validation accuracy.

# Objectives
* Train a new model that is able to recognize fresh and rotten fruit.
* Achieve a validation accuracy of 92% or higher.
* Utilize a combination of transfer learning, data augmentation, and fine-tuning.
* Save the well-trained model and assess its final accuracy.

# The Dataset
* **Source:** A Kaggle dataset of fresh and rotten fruits.
* **Structure:** Located in the `data/fruits` folder.
* **Categories:** 6 distinct categories: fresh apples, fresh oranges, fresh bananas, rotten apples, rotten oranges, and rotten bananas.
* **Model Output:** Your model's output layer will require 6 neurons for successful categorization.
* **Loss Function:** For multi-class classification, `nn.CrossEntropyLoss` (or `categorical_crossentropy` in Keras/TensorFlow contexts) is required.
