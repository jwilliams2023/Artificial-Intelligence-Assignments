# Artificial-Intelligence-Assignments

## Intel Image Classification Project

### Objective
To classify images into six classes: **buildings**, **forest**, **glacier**, **mountain**, **sea**, and **street** using a Convolutional Neural Network (CNN). The dataset comprises 14,000 training images and 3,000 validation images.

### Background
This project was initially assigned as a school assignment to deepen my understanding of deep learning concepts. To gain more hands-on experience with Jupyter, Anaconda, Git, and leveraging my NVIDIA GeForce RTX 4080 GPU, I transferred the project to this GitHub repository. This transition allows for better version control, collaboration, and efficient management of dependencies using Conda.

### Project Overview 
This repository contains implementations of two deep learning tasks:

1. **Part 1: Movie Genre Prediction**
    - **Objective**: Predict the genres of a movie from its short description.
    - **Models**:
        - Recurrent Neural Network (RNN)
        - Long Short-Term Memory (LSTM)
    - **Dataset**: IMDB movie data from 2006 to 2016.
  
2. **Part 2: Intel Image Classification**
    - **Objective**: Classify natural images into six classes using CNNs.
    - **Models**:
        - CNN with 3 Convolutional and 3 Max Pooling layers
        - CNN with 6 Convolutional and 3 Max Pooling layers
    - **Dataset**: Intel Image Classification dataset.

---

## Part 1: Movie Genre Prediction

### Objective
Implement two deep learning algorithms to predict the genres of a movie from its short description. This is a multi-class classification problem since a movie can belong to multiple genres simultaneously.

### Models
1. **RNN Model**
    - Predict genre classes using an RNN.
    - Use sigmoid activation for multi-label classification.
  
2. **LSTM Model**
    - Predict genre classes using an LSTM.
    - Compare performance metrics with the RNN model.

3. **Enhanced Model with Titles**
    - Incorporate movie titles along with descriptions to improve performance.

### Dataset
**IMDB Data (2006-2016)**  
[Download Dataset](https://www.kaggle.com/datasets/PromptCloudHQ/imdb-data)  
Features: Title, Genre, Description, Director, Actors, Year, Runtime, Rating, Votes, Revenue, Metascore  
**Used Columns**: Title, Genre, Description

### Data Preprocessing
- **Cleaning**: Remove stopwords, numbers, and perform lemmatization.
- **Tokenization**: Tokenize words and convert them to numerical vectors using GloVe embeddings.
- **Padding**: Pad sequences to ensure uniform input size.
- **Label Encoding**: Multi-label binarization for genres.

### Implementation
The implementation involves downloading the dataset using the Kaggle API, preprocessing the textual data, encoding the labels, and setting up the models for training and evaluation.

### Training and Evaluation
- Split data into training, validation, and test sets.
- Train both RNN and LSTM models.
- Evaluate using accuracy, precision, recall, and F1-Score.
- Compare performance metrics between the two models.

---

## Part 2: Intel Image Classification

### Objective
To classify natural images into six classes: **streets**, **buildings**, **woods**, **mountains**, **seas**, and **glaciers** using Convolutional Neural Networks (CNNs). The dataset includes 14,000 training images and 3,000 validation images.

### Dataset
**Intel Image Classification**  
[Download Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification?resource=download)  
Features: Images organized into subfolders by class.

### Data Preprocessing
- **Loading Images**: Use OpenCV to load images.
- **Labeling**: Label images based on folder names.
- **Augmentation**: Apply transformations as needed.
- **Normalization**: Normalize image data for training.

### Model
1. **CNN Model**
    - 3 Convolutional layers
    - 3 Max Pooling layers
    - Dropout layer
    - Dense layer before output
  
3. **Hyperparameter Tuning**
    - Experiment with different learning rates, batch sizes, and optimizers.

### Implementation
The implementation includes downloading the dataset using the Kaggle API, preprocessing the image data, defining the CNN architectures, training the models, and evaluating their performance.

### Training and Evaluation
- Train the CNN models on the training set.
- Validate the models on the validation set.
- Report accuracy, precision, recall, and F1-Score.
- Plot training and validation loss and accuracy graphs.

---

## Setup

### Prerequisites
- **Git**: Ensure Git is installed on your system. [Download Git](https://git-scm.com/downloads)
- **Anaconda**: Install Anaconda for environment management. [Download Anaconda](https://www.anaconda.com/products/distribution)
- **NVIDIA CUDA Toolkit**: Required for GPU acceleration. [Download CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
