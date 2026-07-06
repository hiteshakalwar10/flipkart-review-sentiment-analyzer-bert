# Flipkart Review Sentiment Analyzer

A sentiment analysis model that classifies Flipkart product reviews as Positive or Negative, fine-tuned on BERT (bert-base-uncased) using Hugging Face Transformers.

## Problem

Flipkart gets thousands of customer reviews everyday, and manually reading through them to understand overall sentiment isn't practical. This project fine-tunes a BERT model to automatically classify a review as positive or negative, so it can be used to quickly gauge customer sentiment at scale.

## Dataset

The dataset (`Flipkart_Reviews.csv`) contains customer reviews with a 1-5 star rating for each.

- Ratings of 3 (neutral) were dropped since they don't clearly indicate positive or negative sentiment
- Ratings >= 4 were labeled Positive (1), ratings <= 2 were labeled Negative (0)
- Duplicates and missing values were removed before training

## Approach

- Tokenized reviews using `bert-base-uncased` tokenizer (max_length=128)
- Fine-tuned `BertForSequenceClassification` for binary classification using Hugging Face `Trainer`
- 80/20 train-test split, stratified on sentiment to keep class balance consistent
- Trained for 3 epochs, tracking accuracy and F1 score on the test set after each epoch

## Results

The model was evaluated on the held-out test set using accuracy, F1 score, and a confusion matrix (see notebook for exact numbers and plots).

## Demo

A simple Gradio interface is included at the end of the notebook, where you can type in a review and get an instant Positive/Negative prediction.

## Tech Stack

- Python, Pandas
- PyTorch
- Hugging Face Transformers (BERT)
- scikit-learn (metrics, train-test split)
- Gradio (demo interface)

## What I'd improve next

- Try a lighter model like DistilBERT to compare speed vs accuracy
- Add a neutral class instead of dropping 3-star reviews
- Test on reviews from other categories (right now it's trained on one product category's language patterns)

## How to run

1. Open the notebook in Google Colab
2. Upload `Flipkart_Reviews.csv` when prompted
3. Run all cells in order
4. Use the Gradio interface at the end to test your own reviews
