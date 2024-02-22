import plotly.express as px
import streamlit as st

from Backend import SparkBuilder


@st.cache_resource
def getSpark():
    return SparkBuilder("appName")


with st.spinner("Loading data... Please wait..."):
    spark = getSpark()
    as_year, as_month = spark.query.getValutationByYearAMonth()
    tr_year, tr_month = spark.query.getTotalReviewByYearAMonth()
    awp_year, awn_month = spark.query.avarageNegativeAndPositveWordsForMonthAndYear()
    muw_year, luw_year = spark.query.getMostAndLeastUsedWordPerYear()
    correlationSeason = spark.query.getCorrelationBetweenReviewAndSeason().toPandas()

st.title("Year Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    st.title("Average Score per year")
    st.table(as_year)
with col2:
    st.subheader("Distribution of the Scores per year and month")
    fig = px.line(as_month, x='Review_Month', y='avg(Average_Score)', color='Review_Year',
                  labels={'Average_Score': 'Media Valutazione', 'Review_Month': 'Mese', 'Review_Year': 'Anno'})
    st.plotly_chart(fig)

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribution of the total count of reviews per year and month")
    fig1 = px.line(tr_month, x='Review_Month', y='count', color='Review_Year',
                   labels={'count': 'Totale Recensioni', 'Review_Month': 'Mese', 'Review_Year': 'Anno'})
    st.plotly_chart(fig1)
with col4:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    st.title("Total Review per year")
    st.table(tr_year)

st.divider()

col5, col6 = st.columns(2)

with col5:
    st.subheader("Distribution of the average of positive reviews per year and month")
    fig2 = px.line(awp_year, x='Review_Month', y='Average', color='Review_Year',
                   labels={'Average': 'Media', 'Review_Month': 'Mese', 'Review_Year': 'Anno'})
    st.plotly_chart(fig2)
with col6:
    st.subheader("Distribution of the average of negative reviews per year and month")
    fig3 = px.line(awn_month, x='Review_Month', y='Average', color='Review_Year',
                   labels={'Average': 'Media', 'Review_Month': 'Mese', 'Review_Year': 'Anno'})
    st.plotly_chart(fig3)

st.divider()

col7, col8 = st.columns(2)
with col7:
    st.subheader("The most used word by year")
    st.table(muw_year.toPandas())
with col8:
    st.subheader("The least used word by year")
    st.table(luw_year)
st.divider()
col9, col10 = st.columns(2)
with col9:
    st.subheader("Total Review between Season")
    fig4 = px.bar(correlationSeason, x='Season', y='Total', title='Total for Season',
                  labels={'Season': 'Stagione', 'Total': 'Totale'})
    st.plotly_chart(fig4)
with col10:
    st.subheader("Average Hotel's score between Season")
    fig5 = px.bar(correlationSeason, x='Season', y='AScore', title='Total for Season',
                  labels={'Season': 'Stagione', 'AScore': 'Media'})
    st.plotly_chart(fig5)
