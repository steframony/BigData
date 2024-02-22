import joblib
import streamlit as st
from Backend import SparkBuilder
from Utility import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


@st.cache_resource
def getSpark():
    return SparkBuilder("appName")


analysis = False

if analysis:

    # Inizializzazione di Spark
    spark = getSpark()

    # Estrazione dei dati
    dataset = spark.query.getDatasetForClassification()

    X = dataset['Review']  # Features
    y = dataset['Sentiment']  # Variabile di Target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    vectorizer = TfidfVectorizer()  # Usiamo la rappresentazione TF-IDF per le parole
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train_tfidf, y_train)

    y_pred = random_forest.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    model = joblib.dump(random_forest, "C:\\Users\\ste\\Desktop\\random_forest_model.pkl")

    importances = random_forest.feature_importances_

    # lista di tuple contenente il nome della feature e la sua importanza
    feature_importances = [(feature, importance) for feature, importance in
                           zip(vectorizer.get_feature_names_out(), importances)]

    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Stampa le prime N feature più importanti
    top_n = 10
    print(f"Top {top_n} feature più importanti:")
    for feature, importance in feature_importances[:top_n]:
        print(f"{feature}: {importance}")

else:

    with st.spinner("Loading...."):
        # Estrazione dei dati
        spark = getSpark()
        dataset = spark.query.getDatasetForClassification()

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

            # Usiamo il modello addestrato
            predictions = random_forest.predict(new_data_tfidf)

        # Stampa le previsioni
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
