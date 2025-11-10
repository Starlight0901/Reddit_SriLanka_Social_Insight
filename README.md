# Reddit_SriLanka_Social_Insight
An NLP and LLM-driven system for understanding public discourse in Sri Lankan social media. An advanced NLP and LLM-based framework for analyzing Sri Lankan Reddit discussions: combining data collection, machine learning, deep learning, and transformer models to extract thematic, sentiment, and policy insights.

## Project Overview

Lanka Discourse AI is an advanced analytical system designed to explore and interpret large-scale user-generated content from the r/SriLanka subreddit.
The project integrates the full NLP pipeline — from data collection and preprocessing to topic classification, sentiment analysis, and insight generation — using a range of traditional ML, deep learning, and transformer-based models.

By combining modern Natural Language Processing (NLP) techniques with Large Language Models (LLMs), the system aims to uncover complex social and policy-related narratives in Sri Lankan online discussions and transform them into actionable insights for decision-makers and researchers.

## Objectives
This project aims to:
- Collect and preprocess real-world Reddit data from r/SriLanka using the Reddit API.
- Explore and analyze linguistic and structural features of the dataset through exploratory data analysis (EDA).
- Develop and evaluate multiple tokenization and feature engineering strategies.
- Classify and cluster posts by topics using machine learning and deep learning approaches.
- Fine-tune and compare transformer models for post categorization and sentiment classification.
- Leverage LLMs for zero-shot and few-shot analysis to extract deeper contextual insights.
- Perform sentiment and entity analysis on posts related to education and generate policy-relevant summaries.

## Methodological Highlights
- Data Source: Reddit API (r/SriLanka) — 50,000+ user-generated posts
- Text Preprocessing: Cleaning, normalization, and subword tokenization (BPE, WordPiece, SentencePiece)
- Feature Representations: TF-IDF, Bag-of-Words, Word2Vec, GloVe, transformer embeddings

## Models Explored:
- Traditional ML: Logistic Regression, SVM, Naive Bayes
- Deep Learning: CNN, LSTM, BiLSTM, GRU
- Transformer Models: BERT, RoBERTa (encoder-based); GPT, LLaMA (decoder-based)
- LLM Tasks: Zero-shot and few-shot classification, sentiment evaluation, and entity-relation extraction
- Evaluation Metrics: Accuracy, F1-score, Precision, Recall, Perplexity

## Insights and Applications
This project demonstrates how AI-driven text analysis can be used to:
Monitor public sentiment toward education reforms and other policy topics.
Identify emerging social issues and public discourse trends.
Support data-informed policy-making and public communication strategies in Sri Lanka.

## Technologies Used
- Python (pandas, numpy, nltk, spacy, sklearn, matplotlib, seaborn)
- Deep Learning Frameworks: TensorFlow, PyTorch
- Transformer Libraries: Hugging Face Transformers, OpenAI API (optional)
- APIs: Reddit API (via PRAW)
- Visualization: Matplotlib, WordCloud, Plotly

## Expected Outcomes
- A fully preprocessed and analyzed Reddit corpus
- Comparative performance evaluation of ML, DL, and LLM models
- A fine-tuned transformer model for Sri Lankan discourse classification
- Sentiment and entity-based insights on education policy debates
