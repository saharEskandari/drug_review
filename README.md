# RoBERTa and CNN for Drug Review Classification

## Overview
This project uses a combination of **RoBERTa** (a transformer-based model) and **CNN (Convolutional Neural Network)** to classify drug reviews into three sentiment categories:
- **Negative (0):** Ratings between 1-4
- **Neutral (1):** Ratings between 5-6
- **Positive (2):** Ratings between 7-10

## Dataset
The dataset is obtained from **Drugs.com** and consists of two files:
- `drugsComTrain_raw.csv` (Training Data)
- `drugsComTest_raw.csv` (Test Data)

After loading both datasets, they are merged into a single dataframe for preprocessing and model training.

## Steps in the Project
### 1. Install and Import Libraries
We use **pandas**, **nltk**, **torch**, **transformers**, **tensorflow**, and **keras** for data processing, training, and evaluation.

### 2. Load the Dataset
- The dataset is loaded from Google Drive.
- It contains columns such as `drugName`, `condition`, `review`, `rating`, and `date`.
- Exploratory Data Analysis (EDA) is performed to check missing values and class distribution.

### 3. Label Conversion
- The `rating` column is converted into categorical labels:
  - Ratings **1-4** → **Negative (0)**
  - Ratings **5-6** → **Neutral (1)**
  - Ratings **7-10** → **Positive (2)**

### 4. Data Preprocessing
- HTML tags and URLs are removed from reviews.
- Text is converted to lowercase.
- Negations (e.g., *isn't*, *wasn't*) are handled by merging words (e.g., *isn't happy* → *isn_t_happy*).

### 5. Feature Extraction using RoBERTa
- **RoBERTa-Base** tokenizer and model are used to generate text embeddings.
- Reviews are tokenized and converted into dense vector representations.
- Embeddings are saved as `all_embeddings.npy` for future use.

### 6. CNN Model for Classification
- A CNN model with **two convolutional layers**, **batch normalization**, **max pooling**, and **dropout layers** is built using Keras.
- The model input shape is `(768,)`, corresponding to the RoBERTa embeddings.
- The final output layer has 3 neurons (one for each class) with a softmax activation function.
- **Adam optimizer** and **categorical cross-entropy loss** are used for training.

### 7. Model Training and Evaluation
- The dataset is split into **train (65%)**, **validation (10%)**, and **test (25%)**.
- The model is trained using **early stopping** and **learning rate reduction** strategies.
- Performance is evaluated using **accuracy, precision, recall, F1-score**, and a **confusion matrix**.

### 8. Results
- The model achieves a competitive classification performance.
- A confusion matrix is plotted to visualize classification performance.

## Requirements
Ensure the following libraries are installed:
```bash
pip install torch transformers nltk pandas keras tensorflow matplotlib seaborn
```

## How to Run
1. Load the dataset into Google Drive.
2. Run the script in Google Colab or a local Jupyter Notebook.
3. Train the RoBERTa model and extract embeddings.
4. Train the CNN model on the extracted embeddings.
5. Evaluate the model on the test set.

## Author
Developed by [sahar Eskandari].

## License
This project is open-source and can be modified and distributed under the **MIT License**.

