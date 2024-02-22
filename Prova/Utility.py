import string
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyspark.sql import DataFrame

PATH_DS = "C:\\Users\\ste\\Desktop\\Hotel_Reviews.csv"

# Solo gli indirizzi inglesi sono costituiti in modo alternativo
def estraiCitta(indirizzo: str, country: str) -> str:
    if country == "United Kingdom":
        indirizzo_split = indirizzo.split(' ')
        country_index = indirizzo_split.index("Kingdom") #Troviamo la posizione
        # La città si trova generalmente prima dello stato nel formato dell'indirizzo
        city_index = country_index - 6 if "6BD" in indirizzo_split else country_index - 4
        return indirizzo_split[city_index]
    else:
        indirizzo_split = indirizzo.split(' ')
        country_index = indirizzo_split.index(country)
        city_index = country_index - 1
        return indirizzo_split[city_index]


def get_word_frequencies_dict(word_counts_df: DataFrame):
    word_frequencies_dict = {}
    for row in word_counts_df.collect():
        word_frequencies_dict[row['word']] = row['Frequency']
    return word_frequencies_dict


def get_tags_frequencies_dict(tags_df : DataFrame):
    word_frequencies_dict = {}
    for row in tags_df.collect():
        word_frequencies_dict[row['word']] = row['count']
    return word_frequencies_dict

def haversine(lat1: float, lon1: float, lat2: float, lon2: float):
    R = 6371.0  # Raggio medio della Terra in chilometri

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # Formula di Haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return distance

def preprocess(text: str) -> list[str]:
    # Rimozione della punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenizzazione
    tokens = word_tokenize(text)
    # Rimozione dellle stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens

def pulisci_recensione(recensione: str) -> str:
    parole = word_tokenize(recensione)
    stop_words = set(stopwords.words('english'))
    parole_pulite = [parola for parola in parole if parola.lower() not in stop_words]
    recensione_pulita = ' '.join(parole_pulite)
    return recensione_pulita
