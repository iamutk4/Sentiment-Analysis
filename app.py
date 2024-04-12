import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

def vader_sentiment_analysis(sentence):
    # VADER sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(sentence)
    return sentiment

def roberta_sentiment_analysis(sentence):
    # RoBERTa sentiment analysis
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    encoded_text = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = output[0][0].detach().cpu().numpy()
    scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
    }
    return scores_dict

def huggingface_transformer_sentiment_analysis(sentence):
    # HuggingFace Transformer sentiment analysis
    classifier = pipeline('sentiment-analysis')
    result = classifier(sentence)
    return result

if __name__ == "__main__":
    # Prompt user for input sentence
    sentence = input("Enter a sentence: ")

    # Prompt user to choose technique
    technique = input("Choose a technique (VADER, Roberta, HuggingFace): ")

    # Perform sentiment analysis based on chosen technique
    if technique.lower() == "vader":
        sentiment = vader_sentiment_analysis(sentence)
    elif technique.lower() == "roberta":
        sentiment = roberta_sentiment_analysis(sentence)
    elif technique.lower() == "huggingface":
        sentiment = huggingface_transformer_sentiment_analysis(sentence)
    else:
        print("Invalid technique chosen.")
        exit()

    print("Sentiment Analysis Result:", sentiment)
