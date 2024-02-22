import folium
import streamlit as st
from streamlit_folium import st_folium

from Backend import SparkBuilder


@st.cache_resource
def getSpark():
    return SparkBuilder("appName")


def getHotel(spark, country_name):
    df = spark.query.getHotelsByCountry(country_name)
    return [df[i][0] for i in range(len(df))]


with st.spinner("Loading data..."):
    spark = getSpark()
    countrynames = spark.query.getCountryHotel()
    listcountry = [countrynames[i][0] for i in range(len(countrynames))]

st.title("Comparasion Hotel")
col1, col2 = st.columns(2)
hotel2 = ""
with col1:
    country1 = st.selectbox("Select Country Hotel 1", listcountry, index=None, key="country1")
    if country1:
        hotel1 = st.selectbox(
            "Select Hotel",
            getHotel(spark, country1),
            index=None,
            key="hotel1"
        )
with col2:
    country2 = st.selectbox("Select Country Hotel 2", listcountry, index=None, key="country2")
    if country2:
        hotel2 = st.selectbox(
            "Select Hotel",
            getHotel(spark, country2),
            index=None,
            key="hotel2"
        )

if hotel2:
    if hotel2 != "":
        with st.spinner(f"Loading data for {hotel1} and {hotel2}"):
            tags1 = spark.query.getFirstNTagMorePopular(hotel1,5)
            tags2 = spark.query.getFirstNTagMorePopular(hotel2,5)
            h1, h2 = spark.query.getH1H2statistic(hotel1, hotel2)
            lrh1, lrh2 = spark.query.getLastPositiveNegativeReviews(hotel1, hotel2)
        col4, col5 = st.columns(2)
        with col4:

            # Visualizzazione mappa
            st.subheader(f"Hotel {hotel1}")

            mapHotel1 = folium.Map(location=[h1["Latitude"], h1["Longitude"]], zoom_start=25)
            folium.Marker(location=[h1["Latitude"], h1["Longitude"]], popup=h1["Hotel_Name"]).add_to(mapHotel1)
            st_folium(mapHotel1, use_container_width=True, height=500, returned_objects=[])

            st.divider()

            st.subheader("Analysis")

            st.metric("Average Score", h1["Avg_Reviewer_Score"])
            st.metric("Total Reviews", h1["Total_Reviews"])

            st.metric("Total Positive Reviews", h1["Total_Positive_Reviews"])
            st.metric("Total Negative Reviews", h1["Total_Negative_Reviews"])

            st.metric("Max Reviews Score", h1["Max_Reviewer_Score"])
            st.metric("Min Reviews Score", h1["Min_Reviewer_Score"])

            if (h1["Avg_Additional_Number_of_Scoring"] != 0):
                st.metric("Avarage Additional Number of Scoring", h1["Avg_Additional_Number_of_Scoring"])


    with col5:
        st.subheader(f"Hotel {hotel2}")

        mapHotel2 = folium.Map(location=[h2["Latitude"], h2["Longitude"]], zoom_start=25)
        folium.Marker(location=[h2["Latitude"], h2["Longitude"]], popup=h2["Hotel_Name"]).add_to(mapHotel2)
        st_folium(mapHotel2, use_container_width=True, height=500, returned_objects=[])

        st.divider()

        st.subheader("Analysis")

        st.metric("Average Score", h2['Avg_Reviewer_Score'])
        st.metric("Total Reviews", h2["Total_Reviews"])

        st.metric("Total Positive Reviews", h2["Total_Positive_Reviews"])
        st.metric("Total Negative Reviews", h2["Total_Negative_Reviews"])

        st.metric("Max Reviews Score", h2["Max_Reviewer_Score"])
        st.metric("Min Reviews Score", h2["Min_Reviewer_Score"])

        if (h2["Avg_Additional_Number_of_Scoring"] != 0):
            st.metric("Average Additional Number of Scoring", h2["Avg_Additional_Number_of_Scoring"])

    st.divider()

    col6, col7 = st.columns(2)

    with col6:
        st.subheader(f"Lasted Positive Review left {lrh1.iat[0, 3]} days ago")
        st.write(lrh1.iat[0, 1])

        st.subheader(f"Lasted Negative Review left {lrh1.iat[0, 3]} days ago")
        st.write(lrh1.iat[0, 2])
    with col7:
        st.subheader(f"Lasted Positive Review left {lrh2.iat[0, 3]} days ago")
        st.write(lrh2.iat[0, 1])
        st.subheader(f"Lasted Negative Review left {lrh2.iat[0, 3]} days ago")
        st.write(lrh2.iat[0, 2])
    st.divider()
    col8, col9 = st.columns(2)
    with col8:
        st.subheader("Top 5 Tags")
        st.table(tags1)
    with col9:
        st.subheader("Top 5 Tags")
        st.table(tags2)