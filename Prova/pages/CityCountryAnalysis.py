import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium

from Backend import SparkBuilder


@st.cache_resource
def getSpark():
    return SparkBuilder("appName")


def main():
    with st.spinner('Caricamento in corso, Attendere Prego...'):
        spark = getSpark()
        countryInformation = spark.query.cityHotelInformation().toPandas()
        mostused, leastused = spark.query.mostLeastUsedWordsByCity()
        totalNationality = spark.query.getNumberOfDifferentReviewerNationality().toPandas()

    st.title("City and Country Analysis")
    st.header("Total number of reviews for city")
    st.bar_chart(countryInformation, x="City_Hotel", y="Total_Reviews", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Number of Hotels for City")
        totalhotel = px.pie(countryInformation, values="Number_Hotel", names="City_Hotel")
        st.plotly_chart(totalhotel, use_container_width=True, theme="streamlit")
    with col2:
        st.subheader("Average Score for each City")
        avghotel = px.bar(countryInformation, x="Average_Score", y="City_Hotel", orientation="h")
        st.plotly_chart(avghotel, use_container_width=True, theme="streamlit")

    st.divider()

    st.subheader("Number of Positive and Negative Review for each City")
    totalposnegreview = px.bar(countryInformation, y=["TotalN", "TotalP"], x="City_Hotel", orientation="v",
                               barmode="stack")
    st.plotly_chart(totalposnegreview, use_container_width=True, theme="streamlit")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Most Used Word")
        st.table(mostused)
    with col4:
        st.subheader("Least Used Word")
        st.table(leastused)

    st.divider()

    st.subheader("Total of Different Nationality of Reviewer for City")
    totalposnegreview = px.bar(totalNationality, y="Different_Nationality", x="City_Hotel", orientation="v")
    st.plotly_chart(totalposnegreview, use_container_width=True, theme="streamlit")

    st.divider()
    st.subheader("Find the average score with a particular keyword")
    keyword = st.text_input("Enter your keyword:")

    if keyword != '':
        with st.spinner("Filtering the data..."):
            score, hotel_info = spark.query.averageRatingByKeyword(keyword)

        st.subheader(f"The Avarage Score is:{score.collect()[0][0]}")
        st.divider()
        st.subheader("The Hotel with the highest Score")
        st.write(
            f"In this section there is the hotel with the highest score which have one o more review, containing the keyword: {keyword}")

        col1, col2 = st.columns(2)
        with col1:
            mapHotel = folium.Map(location=[hotel_info[0]['lat'], hotel_info[0]['lng']], zoom_start=15)
            folium.Marker(location=[hotel_info[0]['lat'], hotel_info[0]['lng']], popup=hotel_info[0]['Hotel_Name'],
                          icon=folium.Icon(color='red')).add_to(
                mapHotel)
            st_folium(mapHotel, use_container_width=True, height=250, returned_objects=[])
        with col2:
            st.write(f"Name: <span style='font-size:25px'> <strong> {hotel_info[0]['Hotel_Name']}</strong></span>",unsafe_allow_html=True)
            st.write(
                f"Address: <span style='font-size:25px'> <strong> {hotel_info[0]['Hotel_Address']}</strong></span>",unsafe_allow_html=True)
            st.write(
                f"Average Score: <span style='font-size:25px'> <strong> {hotel_info[0]['Average_Score']}</strong></span>",unsafe_allow_html=True)


main()
