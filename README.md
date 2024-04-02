# Sentiment Analysis and Review Classification

This project involves sentiment analysis and review classification using various natural language processing (NLP) techniques, like traditional lexicon-based methods and state-of-the-art deep learning models. The dataset used contains Amazon Fine Food reviews.

## Methods:

1. **VADER** (Valence Aware Dictionary and sEntiment Reasoner)
2. **RoBERTa** (A Robustly Optimized BERT Pretraining Approach)
3. **Huggingface Pipeline**

## Project Overview:

1. **Exploratory Data Analysis (EDA):** Initial exploration of the dataset to understand its structure, distribution, and key features.

2. **VADER Sentiment Scoring:** Utilizes the VADER (Valence Aware Dictionary and sEntiment Reasoner) tool for sentiment analysis. Calculates sentiment scores for each review and visualizes the distribution of positive, neutral, and negative sentiments.

3. **RoBERTa Pretrained Model:** Implements a pre-trained RoBERTa model for sentiment analysis. Uses the Cardiff University's Twitter RoBERTa base model to classify the sentiment of reviews into negative, neutral, and positive categories. The model's predictions are evaluated and compared with VADER scores.

4. **Review Examples Analysis:** Investigates review examples where the model's predictions significantly differ from the review scores. Specifically examines cases where positive reviews are predicted as 1-star reviews and negative reviews as 5-star reviews.

5. **HuggingFace Transformer Pipeline:** Utilizes the HuggingFace Transformer pipeline for sentiment analysis on random sentences to demonstrate the model's capabilities beyond the dataset.

## Libraries and Tools Used:

- `pandas`: For data manipulation and analysis
- `NLTK`: For natural language processing tasks, including tokenization, part-of-speech tagging, and sentiment analysis
- `seaborn` and `matplotlib`: For data visualization
- `transformers` from HuggingFace: For accessing pre-trained language models like RoBERTa
- `torch`: For tensor computations and GPU support

## Dataset:

The dataset used in this project is sourced from [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). It contains a collection of reviews, including textual reviews, ratings, and other metadata.

## Usage:

1. Clone the repository:

```
git clone https://github.com/your-username/sentiment-analysis-project.git
```

2. Navigate to project directory
```
cd sentiment-analysis
```
3. Open the Jupyter notebook Sentiment_Analysis.ipynb using Google Colab.
