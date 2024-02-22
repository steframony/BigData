import matplotlib.pyplot as plt
import streamlit as st
import wordcloud as wc

from Backend import SparkBuilder
from Utility import get_word_frequencies_dict, get_tags_frequencies_dict

st.set_page_config(
    page_title="BigDataProject",
    page_icon="👌",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def getSpark():
    return SparkBuilder("appName")

def main():

    with st.spinner('Loading... Please wait...'):

        spark = getSpark()
        loglatdat = spark.query.getlatlong()
        totalNumberHotels = spark.query.getNumOfHotel()
        countryHotel = spark.query.getCountryHotel()
        dswords = spark.query.wordsFrequency()
        dstags = spark.query.mostAndLeastTagUsed()
        nationalityReviewer = spark.query.getReviewerNationality().toPandas()
        longestpositive, longestnegative, shortestpositive, shortestnegative = spark.query.longestShortestReviews()
        hotel_most, hotel_least = spark.query.hotelsWithMaxMinReviewDifference()

    st.title("Hotel Dataset Analysis")
    st.subheader("Geographic Locations")
    st.markdown(
        "The dataset contains **reviews** of hotels scattered across *Europe*."
        "It includes various attributes such as hotel address, review date,"
        "average score, reviewer nationality, negative and positive reviews,"
        "and geographical coordinates (latitude and longitude)."
    )
    st.markdown("In the map above, each point represents a :red[hotel]."
                "The locations of these points correspond to the geographical positions of the hotels.")

    st.divider()

    # Rinominiamo le colonne del dataset
    loglatdat.rename(columns={'first(lat)': 'lat'}, inplace=True)
    loglatdat.rename(columns={'first(lng)': 'lon'}, inplace=True)
    loglatdat.drop(columns=['Hotel_Name'], inplace=True)

    # Mappiamo i primi punti
    st.map(data=loglatdat, use_container_width=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Reviews")
        st.metric("", spark.dataset.count())

    with col2:
        st.subheader("Total number of Hotels")
        st.metric("", totalNumberHotels)

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Country of Hotels")
        listCountry = ''
        for i in countryHotel:
            listCountry += "- " + i['Country_Hotel'] + "\n"
        st.markdown(listCountry)
    with col4:
        st.subheader("Reviewer Nationality")
        totalNationality = len(nationalityReviewer)
        nationalityReviewer = nationalityReviewer.T
        st.dataframe(
            data=nationalityReviewer,
            hide_index=False,
        )
        st.subheader("Total Nationality")
        st.metric("", totalNationality)

    st.divider()
    st.subheader("The Most and the Least used word in a review")
    wordset = get_word_frequencies_dict(dswords)
    wordgraph = wc.WordCloud(width=800, height=250, background_color='white', max_font_size=200).generate_from_frequencies(wordset)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(wordgraph)
    plt.axis("off")
    st.pyplot(fig)

    max_w, min_w = spark.query.maxMinFrequency()

    st.markdown(f"{max_w['word'].upper()} was the **most** used word. Meanwhile {min_w['word'].upper()} was the **least** used word.")

    st.divider()
    st.subheader("The Longest Positive and Negative Reviews in the Dataset")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"The positive review is: {longestpositive.collect()[0][0]}")
    with col6:
        st.markdown(f"The negative review is: {longestnegative.collect()[0][0]}")
    st.divider()

    st.subheader("The Shortest Positive and Negative Reviews in the Dataset")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown(f"The positive review is: {shortestpositive.collect()[0][0]}")
    with col6:
        st.markdown(f"The negative review is: {shortestnegative.collect()[0][0]}")
    st.divider()
    st.subheader("The Most and the Least used Tag")
    tagfre = get_tags_frequencies_dict(dstags)
    taggraph = wc.WordCloud(width=800, height=250, background_color='white', max_font_size=200).generate_from_frequencies(tagfre)
    fig1, ax1 = plt.subplots(figsize=(14,10))
    ax1.imshow(taggraph)
    plt.axis("off")
    st.pyplot(fig1)
    st.markdown(f"{dstags.collect()[0][0]} was the most used tag!")
    st.divider()
    st.subheader("The Hotels with the most difference of positive e negative review")
    st.write(f"Name Hotel:{hotel_most[0]}")
    st.write(f"Review Difference: {hotel_most[1]}")
    st.divider()
    st.subheader("The Hotels with the least difference of positive e negative review")
    st.write(f"Name Hotel:{hotel_least[0]}")
    st.write(f"Review Difference: {hotel_least[1]}")
    st.divider()

main()
