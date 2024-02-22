import pandas
import streamlit as st
import folium
from streamlit_folium import st_folium
from opencage.geocoder import OpenCageGeocode
from Backend import SparkBuilder

key = "3e65099fd5c14403a75e847746e0bf16"

@st.cache_resource
def getSpark():
    return SparkBuilder("appName")

with st.spinner("Loading data..."):
    spark = getSpark()
    geocoder = OpenCageGeocode(key)

st.title("Find Hotels nearest to you")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Insert your address")
    address = st.text_input('The text for must be: "Address","City/Country"')
with col2:
    st.subheader("Select the distance")
    distance = st.slider('Distance in KM', 0, 50)

positionUser = []

if distance and address:
    with st.spinner("Searching Data..."):
        location = geocoder.geocode(address)
        positionUser = [location[0]['geometry']['lat'], location[0]['geometry']['lng']]
        hotelnear = spark.query.hotelsWithinDistance(positionUser[0], positionUser[1], distance)

    st.divider()
    col3, col4 = st.columns(2)
    distance = {}

    with col1:
        mapHotel = folium.Map(location=[positionUser[0], positionUser[1]], zoom_start=15)
        folium.Marker(location=[positionUser[0], positionUser[1]], popup=address).add_to(mapHotel)

        for row in hotelnear.collect():
            hotel_name = row["Hotel_Name"]
            hotel_location = (row["Latitude"], row["Longitude"])
            distance_km = row["distance_km"]
            distance[hotel_name] = distance_km
            folium.Marker(location=hotel_location, popup=hotel_name, icon=folium.Icon(color='green')).add_to(
                mapHotel)

        st_folium(mapHotel, use_container_width=True, height=400, returned_objects=[])

    with col2:
        if len(distance) == 0:
            text = "No Hotel"
            st.write(f"<span style='font-size:35px'><strong>{text}</strong></span>",unsafe_allow_html=True)
        else:
            st.dataframe(pandas.DataFrame(list(distance.items()), columns=['City', 'Distance']).set_index('City'),use_container_width=True)