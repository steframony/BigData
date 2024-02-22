import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from Backend import *
from Utility import haversine


@st.cache_resource
def getSpark():
    return SparkBuilder("appName")


def getHotel(spark, cityname):
    df = spark.query.getHotelsByCity(cityname)
    return [df[i][0] for i in range(len(df))]


def draw_path(m, start, end):
    folium.PolyLine(locations=[start, end], color='blue').add_to(m)


with st.spinner("Loading data..."):
    spark = getSpark()
    citynames = spark.query.getCityHotel()
    listcity = [citynames[i][0] for i in range(len(citynames))]
    df_m = pd.read_csv('C:\\Users\\ste\\Desktop\\Monument.csv', delimiter=",")

st.title("How far is the hotel?")

col1, col2 = st.columns(2)
hotel = ""

city = st.selectbox("Select Country Hotel", listcity, index=None, key="country1")

if city:
    hotel = st.selectbox(
        "Select Hotel",
        getHotel(spark, city),
        index=None,
        key="hotel1"
    )

if hotel:
    if hotel != "":
        with st.spinner(f"Searching data for {hotel}"):
            df_m = df_m[df_m['City'] == city]
            h1, _ = spark.query.getH1H2statistic(hotel, hotel)

        st.divider()

        st.subheader("Distance from Monument")
        col1, col2 = st.columns(2)
        distance = {}

        with col1:
            mapHotel = folium.Map(location=[h1["Latitude"], h1["Longitude"]], zoom_start=15)
            folium.Marker(location=[h1["Latitude"], h1["Longitude"]], popup=h1["Hotel_Name"]).add_to(mapHotel)

            for index, monument in df_m.iterrows():
                monument_location = [monument["lat"], monument["long"]]
                monument_name = monument["Monument"]
                folium.Marker(location=monument_location, popup=monument_name, icon=folium.Icon(color='green')).add_to(
                    mapHotel)
                draw_path(mapHotel, monument_location, [h1["Latitude"], h1["Longitude"]])
                distance_km = haversine(float(h1["Latitude"]), float(h1["Longitude"]), float(monument_location[0]),
                                        float(monument_location[1]))
                distance[monument_name] = distance_km

            st_folium(mapHotel, width=600, height=600, returned_objects=[])

        with col2:
            for luogo, distanza in distance.items():
                distanza_arrotondata = round(distanza, 2)
                st.write(f"<span style='font-size:35px'> <strong>{luogo}</strong>: {distanza_arrotondata} km</span>",
                         unsafe_allow_html=True)

        st.divider()
