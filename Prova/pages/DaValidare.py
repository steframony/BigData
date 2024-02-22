import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from Backend import SparkBuilder
from Utility import *
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd

@st.cache_resource
def getSpark():
    return SparkBuilder("appName")

with st.spinner("Loading...."):

    # Estrazione dei dati

    #spark = getSpark()
    #dataset = spark.query.getDatasetForClassification()
    #CON STOP WORD
    dataset = pd.read_csv("C:\\Users\\ste\\Desktop\\dataset_classification2.csv")

    X = dataset['Review']  # Features
    y = dataset['Sentiment']  # Variabile Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)


    random_forest = joblib.load("C:\\Users\\ste\\Desktop\\random_forest_model_2.pkl")

st.title("Sentimental Anlaysis")

st.markdown(
    "In this section we will use the Hotel's Dataset to perform a sentiment analysis on new reviews never seen before")

review = st.text_input("Enter your review:")
if review != '':
    with st.spinner("Loading the model..."):
        newreview = [review]
        process = preprocess(newreview[0])
        new_data_tfidf = vectorizer.transform(process)

        # Usiamo il RANDOMFOREST già addestrato
        predictions = random_forest.predict(new_data_tfidf)

        #Analisi con Bert e RoBERTa
        #Carichiamo il modello di tokenizzazione per entrambi i modelli
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        roberta_model = RobertaForSequenceClassification.from_pretrained('roberta-base')

        # Tokenizzazione del testo
        inputs_bert = bert_tokenizer(review, return_tensors='pt', padding=True, truncation=True)
        inputs_roberta = roberta_tokenizer(review, return_tensors='pt', padding=True, truncation=True)

        # Otteniamo la percentuale relativa ad entrami i sentimenti
        outputs_bert = bert_model(**inputs_bert)
        predictions_bert = torch.softmax(outputs_bert.logits, dim=1).tolist()[0]

        outputs_roberta = roberta_model(**inputs_roberta)
        predictions_roberta = torch.softmax(outputs_roberta.logits, dim=1).tolist()[0]


    st.subheader("Random Forest")
    #NOSTRE PREVISIONI
    for review, prediction in zip(newreview, predictions):
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        if prediction == 0:
            st.write(f'The review is: {review}')
            st.error(f'The sentiment is: {sentiment.upper()}')
        else:
            st.write(f'The review is: {review}')
            st.success(f'The sentiment is: {sentiment.upper()}')
        print(f"Recensione: {review}")
        print(f"Sentimento predetto: {sentiment}")

    st.divider()

    # Interpretazione delle predizioni modelli Bert e RoBERTa
    labels = ['negativo', 'positivo']
    predizioneBert = {labels[i]: round(predictions_bert[i], 3) for i in range(len(predictions_bert))}
    predizioneRoberta = {labels[i]: round(predictions_roberta[i], 3) for i in range(len(predictions_roberta))}

    print("Predizioni con BERT:", predizioneBert)
    print("Predizioni con RoBERTa:", predizioneRoberta)

    st.subheader("Prediction with Bert")
    if predizioneBert['negativo'] > predizioneBert['positivo']:
        st.error("Sentiment:  NEGATIVE")
    else:
        st.success("Sentiment: POSITIVE")

    st.divider()

    st.subheader("Prediction with RoBERTa")
    if predizioneRoberta['negativo'] > predizioneRoberta['positivo']:
        st.error("Sentiment: NEGATIVE")
    else:
        st.success("Sentiment: POSITIVE")
