import nltk

from Backend import *
import joblib
import streamlit as st
from Backend import SparkBuilder
from Utility import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


PATH_DS = "C:\\Users\\ste\\Desktop\\Hotel_Reviews.csv"

if __name__ == "__main__":
    spark = SparkBuilder("AppName")


