from nltk.corpus import stopwords
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (regexp_replace, split, expr, col, to_date, regexp_extract, udf, count,
                                   avg, sum, when, first, concat, max, min, countDistinct, explode, lower, lit, year,
                                   month, abs)
from pyspark.sql.types import IntegerType, FloatType, StringType, Row
from Utility import PATH_DS, estraiCitta, haversine


class SparkBuilder:

    def __init__(self, appname: str):
        self.spark = (SparkSession.builder.master("local[*]").
                      appName(appname).getOrCreate())
        self.dataset = self.spark.read.csv(PATH_DS, header=True, inferSchema=True)
        self.castDataset()

    def castDataset(self):
        df = self.dataset

        # Effettuaiamo il cast di qualche colonna
        df = df.withColumn("Additional_Number_of_Scoring", df["Additional_Number_of_Scoring"].cast(IntegerType()))
        df = df.withColumn("Review_Date", to_date(df["Review_Date"], "M/d/yyyy"))
        df = df.withColumn("Average_Score", df["Average_Score"].cast(FloatType()))
        df = df.withColumn("Review_Total_Negative_Word_Counts", df["Review_Total_Negative_Word_Counts"] \
                           .cast(IntegerType()))
        df = df.withColumn("Total_Number_of_Reviews", df["Total_Number_of_Reviews"].cast(IntegerType()))
        df = df.withColumn("Review_Total_Positive_Word_Counts", df["Review_Total_Positive_Word_Counts"] \
                           .cast(IntegerType()))
        df = df.withColumn("Total_Number_of_Reviews_Reviewer_Has_Given",
                           df["Total_Number_of_Reviews_Reviewer_Has_Given"].cast(IntegerType()))
        df = df.withColumn("Reviewer_Score", df["Reviewer_Score"].cast(FloatType()))
        df = df.withColumn("days_since_review",
                           regexp_extract(col("days_since_review"), r"(\d+)", 1).cast(IntegerType()))

        df = df.withColumn("lat", df["lat"].cast(FloatType()))
        df = df.withColumn("lng", df["lng"].cast(FloatType()))

        # Convertiamo i tag in un array di stringhe
        df = df.withColumn("Tags", regexp_replace(col("Tags"), "[\[\]']", ""))
        df = df.withColumn("Tags", split(col("Tags"), ", "))
        df = df.withColumn("Tags", expr("transform(Tags, x -> trim(x))"))

        # Aggiunta di quattro colonne per facilitare le query:
        # Country_Hotel rappresenta la nazionalità dell'hotel
        df = df.withColumn("Country_Hotel",
                           regexp_extract(col("Hotel_Address"), r'(United\s+Kingdom|\b[A-Z][a-z]+)$', 1))
        udf_estraiCitta = udf(estraiCitta, StringType())
        # City_Hotel rappresenta la città dove è ubicato l'hotel
        df = df.withColumn("City_Hotel", udf_estraiCitta(col("Hotel_Address"), col("Country_Hotel")))
        # Review_Year rappresenta l'anno di pubblicazione della recensione
        df = df.withColumn("Review_Year", year(df["Review_Date"]))
        # Review_Month rappresenta il mese di pubblicazione della recensione
        df = df.withColumn("Review_Month", month(df["Review_Date"]))

        # dataset dannaggiato
        df = df.filter(col("lat").isNotNull() & col("lat").isNotNull() & (~col("Reviewer_Nationality").like(" ")))

        self.dataset = df.cache()
        self.query = QueryManager(self)

    def closeConnection(self):
        self.spark.stop()
        print("Connessione Chiusa")


class QueryManager:
    def __init__(self, spark: SparkBuilder):
        self.spark = spark

    # ------ QUERY DI SUPPORTO --------#
    def getCountryHotel(self) -> list[Row]:
        return self.spark.dataset.select(col("Country_Hotel").alias("Country_Hotel")).distinct().collect()

    def getCityHotel(self) -> list[Row]:
        return self.spark.dataset.select(col("City_Hotel").alias("City_Hotel")).distinct().collect()

    def getHotelsByCountry(self, country_name: str) -> list[Row]:
        return self.spark.dataset.filter(col("Country_Hotel").like(country_name)).select(
            col("Hotel_Name")).distinct().collect()

    def getHotelsByCity(self, cityname: str) -> list[Row]:
        return self.spark.dataset.filter(col("City_Hotel").like(cityname)).select(
            col("Hotel_Name")).distinct().collect()

    def getlatlong(self):
        return self.spark.dataset.groupBy("Hotel_Name").agg({"lng": "first", "lat": "first"}).toPandas()

    def getNumOfHotel(self) -> int:
        return self.spark.dataset.select(col("Hotel_Name")).distinct().count()

    def getReviewerNationality(self):
        df = self.spark.dataset
        return df.select(col("Reviewer_Nationality")).distinct()

    def wordsFrequency(self):
        df = self.spark.dataset

        stop_words = set(stopwords.words('english'))

        words = df.select(
            explode(
                split(
                    lower(concat(col("Negative_Review"), col("Positive_Review"))), "\s+")).alias("word"))
        words = words.filter(~(words["word"].isin(stop_words)) & (words["word"] != ""))
        return words.groupby("word").agg(
            count("*").alias("Frequency")
        )

    def maxMinFrequency(self):
        df = self.wordsFrequency()
        max_word = df.orderBy(col("Frequency").desc()).first()
        min_word = df.orderBy(col("Frequency").asc()).first()
        return max_word, min_word

    def getH1H2statistic(self, hotelname1: str, hotelname2: str):
        df = self.hotelStatistics()
        if hotelname1 == hotelname2:
            h1_stats = df.filter(col("Hotel_Name") == hotelname1).first().asDict()
            return h1_stats,h1_stats
        else:
            h1_stats = df.filter(col("Hotel_Name") == hotelname1).first().asDict()
            h2_stats = df.filter(col("Hotel_Name") == hotelname2).first().asDict()
            return h1_stats, h2_stats

    def getLastPositiveNegativeReviews(self, hotel_name:str, hotel_name2:str):
        df = self.getWhenLastReviewWasPostedOfHotel()
        df.rename(columns={'first(days_since_review)': 'DSR'}, inplace=True)
        df.rename(columns={'first(Positive_Review)': 'Positive'}, inplace=True)
        df.rename(columns={'first(Negative_Review)': 'Negative'}, inplace=True)
        return df[df['Hotel_Name'] == hotel_name], df[df['Hotel_Name'] == hotel_name2]

    def getDatasetForClassification(self):
        df = self.spark.dataset
        review_p = df.select(col("Positive_Review").alias("Review"))
        new_df_p = review_p.withColumn("Sentiment", lit(1))
        new_df_p = new_df_p.filter(~col("Review").like("No Positive"))

        review_n = df.select(col("Negative_Review").alias("Review"))
        new_df_n = review_n.withColumn("Sentiment", lit(0))
        new_df_n = new_df_n.filter(~col("Review").like("No Negative"))

        dataset = new_df_p.union(new_df_n)

        return dataset.toPandas()

    def getNumberOfDifferentReviewerNationality(self):
        df = self.spark.dataset
        numNationality = df.groupBy("City_Hotel") \
            .agg(countDistinct("Reviewer_Nationality").alias("Different_Nationality")) \
            .orderBy("City_Hotel")

        return numNationality

    def getValutationByYearAMonth(self) -> tuple:
        df = self.spark.dataset

        average_score_per_year = df.groupBy("Review_Year").avg("Average_Score").orderBy("Review_Year")
        average_score_per_month = (df.groupBy("Review_Year", "Review_Month").avg("Average_Score")
                                   .orderBy("Review_Year","Review_Month"))
        return average_score_per_year.toPandas(), average_score_per_month.toPandas()

    def getTotalReviewByYearAMonth(self):
        df = self.spark.dataset

        reviews_count_per_year = df.groupBy("Review_Year").count().orderBy("Review_Year")

        reviews_count_per_month = df.groupBy("Review_Year", "Review_Month").count().orderBy("Review_Year",
                                                                                            "Review_Month")
        return reviews_count_per_year, reviews_count_per_month


    #----------------- QUERY 1-----------------------#
    def cityHotelInformation(self):
        df = self.spark.dataset

        all_info = df.groupby("City_Hotel").agg(
            count("*").alias("Total_Reviews"),
            countDistinct("Hotel_Name").alias("Number_Hotel"),
            avg("Average_Score").alias("Average_Score"),
            sum(when((col("Negative_Review").like("No Negative")) | (col("Negative_Review").like("Nothing")),
                     0).otherwise(1)).alias("TotalN"),
            sum(when((col("Positive_Review").like("No Positive")) | (col("Positive_Review").like("Nothing")),
                     0).otherwise(1)).alias("TotalP"),
        )
        return all_info

    #----------------------QUERY 2-------------------#
    def mostLeastUsedWordsByCity(self):
        df = self.spark.dataset

        stop_words = set(stopwords.words('english'))

        new_df = df.select(col("City_Hotel"), explode(
            split(lower(concat(col("Negative_Review"), col("Positive_Review"))), "\s+")).alias("Words"))

        new_df = new_df.filter(~(new_df["Words"].isin(stop_words)) & (new_df["Words"] != ""))

        word_frequency = new_df.groupBy("City_Hotel", "Words").agg(count("*").alias("Frequency"))

        maxword = word_frequency.groupBy("City_Hotel").agg(max("Frequency").alias("Frequency"))

        maxword = (maxword.join(word_frequency,["City_Hotel", "Frequency"], how="inner"))

        minword = word_frequency.groupBy("City_Hotel").agg(min("Frequency").alias("Frequency"))

        minword = (minword.join(word_frequency, ["City_Hotel", "Frequency"], how="inner")
        .groupBy("City_Hotel").agg(
            first("Words").alias("Words"),
            first("Frequency").alias("Frequency")
        ))

        return maxword, minword

    #-----------------------QUERY 3------------------------#
    def hotelStatistics(self):
        df = self.spark.dataset

        res = df.groupBy("Hotel_Name").agg(
            count("*").alias("Total_Reviews"),
            sum(when(col("Positive_Review") != "No Positive", 1).otherwise(0)).alias("Total_Positive_Reviews"),
            sum(when(col("Negative_Review") != "No Negative", 1).otherwise(0)).alias("Total_Negative_Reviews"),
            max("Reviewer_Score").alias("Max_Reviewer_Score"),
            min("Reviewer_Score").alias("Min_Reviewer_Score"),
            avg("Reviewer_Score").alias("Avg_Reviewer_Score"),
            first("lat").alias("Latitude"),
            first("lng").alias("Longitude"),
            avg("Additional_Number_of_Scoring").alias("Avg_Additional_Number_of_Scoring")
        )
        return res

    #------------------------QUERY 4--------------------------#
    def longestShortestReviews(self):
        df = self.spark.dataset

        max_positive_count = df.selectExpr("MAX(Review_Total_Positive_Word_Counts) as Max_Positive_Count") \
            .collect()[0]["Max_Positive_Count"]
        max_reviews_positive = df.filter((col("Review_Total_Positive_Word_Counts") == max_positive_count)) \
            .select("Positive_Review", "Review_Total_Positive_Word_Counts")

        max_negative_count = df.selectExpr("MAX(Review_Total_Negative_Word_Counts) as Max_Negative_Count") \
            .collect()[0]["Max_Negative_Count"]
        max_negative_review = df.filter(col("Review_Total_Negative_Word_Counts") == max_negative_count) \
            .select("Negative_Review", "Review_Total_Negative_Word_Counts")

        min_positive_count = df.selectExpr("MIN(Review_Total_Positive_Word_Counts) as Min_Positive_Count") \
            .collect()[0]["Min_Positive_Count"]
        min_reviews_positive = df.filter((col("Review_Total_Positive_Word_Counts") == min_positive_count)) \
            .select("Positive_Review", "Review_Total_Positive_Word_Counts")

        min_negative_count = df.selectExpr("MIN(Review_Total_Negative_Word_Counts) as Min_Negative_Count") \
            .collect()[0]["Min_Negative_Count"]
        min_negative_review = df.filter(col("Review_Total_Negative_Word_Counts") == min_negative_count) \
            .select("Negative_Review", "Review_Total_Negative_Word_Counts")

        return max_reviews_positive, max_negative_review, min_reviews_positive, min_negative_review

    """
    def longestShortestReviews(self):
        df = self.spark.dataset
    
        max_positive_count = df.agg({"Review_Total_Positive_Word_Counts": "max"}).collect()[0][0]
        max_reviews_positive = df.filter(col("Review_Total_Positive_Word_Counts") == max_positive_count) \
            .select("Positive_Review", "Review_Total_Positive_Word_Counts")
    
        max_negative_count = df.agg({"Review_Total_Negative_Word_Counts": "max"}).collect()[0][0]
        max_negative_review = df.filter(col("Review_Total_Negative_Word_Counts") == max_negative_count) \
            .select("Negative_Review", "Review_Total_Negative_Word_Counts")
    
        min_positive_count = df.agg({"Review_Total_Positive_Word_Counts": "min"}).collect()[0][0]
        min_reviews_positive = df.filter(col("Review_Total_Positive_Word_Counts") == min_positive_count) \
            .select("Positive_Review", "Review_Total_Positive_Word_Counts")
    
        min_negative_count = df.agg({"Review_Total_Negative_Word_Counts": "min"}).collect()[0][0]
        min_negative_review = df.filter(col("Review_Total_Negative_Word_Counts") == min_negative_count) \
            .select("Negative_Review", "Review_Total_Negative_Word_Counts")

    return max_reviews_positive, max_negative_review, min_reviews_positive, min_negative_review
    """
    #------------------------------- QUERY 5 --------------------#
    def mostAndLeastTagUsed(self):
        df = self.spark.dataset

        df_tags = df.select(explode("Tags").alias("word"))

        frequenza_tag = df_tags.groupBy("word").count()

        frequenza_tag = frequenza_tag.orderBy("count", ascending=False)

        return frequenza_tag

    #-----------------------------QUERY 6----------------------#
    def getWhenLastReviewWasPostedOfHotel(self) -> DataFrame:
        df = self.spark.dataset

        m_df = df.groupBy("Hotel_Name").agg(min(col("days_since_review")).alias("min_days_since_review"))
        m_df = m_df.withColumnRenamed("Hotel_Name", "Name")

        dff = df.join(m_df, (df["Hotel_Name"] == m_df["Name"]) & (
                df["days_since_review"] == m_df["min_days_since_review"]), "inner")

        reviews = dff.select("Hotel_Name", "days_since_review", "Positive_Review", "Negative_Review").orderBy(
            "days_since_review")

        recensioni = reviews.groupBy("Hotel_Name").agg(
            {"days_since_review": "first", "Positive_Review": "first", "Negative_Review": "first"}).toPandas()

        return recensioni

    #------------------------ QUERY 7----------------------#
    def avarageNegativeAndPositveWordsForMonthAndYear(self) -> tuple:
        df = self.spark.dataset

        average_negative_words = (df.groupBy("Review_Year", "Review_Month").agg(
            avg("Review_Total_Negative_Word_Counts").alias("Average"))
                                  .orderBy("Review_Year", "Review_Month"))

        average_positive_words = (df.groupBy("Review_Year", "Review_Month").agg(
            avg("Review_Total_Positive_Word_Counts").alias("Average"))
                                  .orderBy("Review_Year", "Review_Month"))

        return average_positive_words, average_negative_words

    #----------------------- QUERY 8--------------------------#
    def getMostAndLeastUsedWordPerYear(self) -> tuple:
        df = self.spark.dataset
        stop_words = set(stopwords.words('english'))

        df = df.withColumn("Full_Review", concat(col("Negative_Review"), col("Positive_Review")))

        m_df = df.select("Review_Year", explode(split(lower("Full_Review"), "\s+")).alias("Words"))
        m_df = m_df.filter(~(m_df["Words"].isin(stop_words)) & (m_df["Words"] != ""))

        word_counts_per_year = m_df.groupBy("Review_Year", "Words").count()

        word_most_used_per_year = (word_counts_per_year.orderBy("Review_Year",word_counts_per_year["count"].desc())
                                   .groupby("Review_Year").agg({'Words': 'first'}))
        word_least_used_per_year = (word_counts_per_year.orderBy("Review_Year", word_counts_per_year["count"])
                                    .groupby("Review_Year").agg({'Words': 'first'}))

        return word_most_used_per_year, word_least_used_per_year

    #------------------------------- QUERY 9------------------------#
    def getCorrelationBetweenReviewAndSeason(self) -> DataFrame:
        df = self.spark.dataset
        df = df.withColumn("Season",
                           when((df["Review_Month"] >= 3) & (df["Review_Month"] <= 5), "Spring")
                           .when((df["Review_Month"] >= 6) & (df["Review_Month"] <= 8), "Summer")
                           .when((df["Review_Month"] >= 9) & (df["Review_Month"] <= 11), "Autumn")
                           .otherwise("Winter"))

        res = df.groupBy("Season").agg(
            count("*").alias("Total"),
            avg("Average_Score").alias("AScore")
        ).orderBy("Season")

        return res

    #--------------------------------- QUERY 10----------------------------#
    def getFirstNTagMorePopular(self, hotel_name: str, n: int) -> DataFrame:
        df = self.spark.dataset

        hotel_df = df.filter(col("Hotel_Name") == hotel_name)

        exploded_tags = hotel_df.select(explode(col("Tags")).alias("Tag"))

        tag_count = exploded_tags.groupBy("Tag").agg(count("*").alias("Tag_Count"))

        top_n_tags = tag_count.orderBy(col("Tag_Count").desc()).limit(n)\
            if n <= tag_count.count() else tag_count.orderBy(col("Tag_Count").desc())

        return top_n_tags

    #--------------------------------- QUERY 11----------------------------#
    def hotelsWithMaxMinReviewDifference(self) -> tuple:
        df = self.spark.dataset

        review_difference = df.groupBy("Hotel_Name").agg(
            abs(sum(when(col("Positive_Review") != "No Positive", 1).otherwise(0))-sum(when(col("Negative_Review") != "No Negative", 1).otherwise(0))).alias("Review_Difference")
        )

        max_difference = review_difference.orderBy(col("Review_Difference").desc()).first()
        min_difference = review_difference.orderBy(col("Review_Difference").asc()).first()

        return max_difference, min_difference

    #--------------------------------- QUERY 12----------------------------#
    def averageRatingByKeyword(self, keyword: str) -> tuple:
        df = self.spark.dataset

        keyword_reviews = df.filter(
            (col("Negative_Review").contains(keyword)) | (col("Positive_Review").contains(keyword)))

        if keyword_reviews.count() == 0:
            return None, None

        avg_rating_by_keyword = keyword_reviews.agg(avg("Average_Score").alias("Average_Rating"))

        hotel_w_max_score_review = keyword_reviews.orderBy(col("Average_Score").desc()).limit(1)

        return avg_rating_by_keyword, hotel_w_max_score_review.collect()

    #--------------------------------- QUERY 13----------------------------#
    def hotelsWithinDistance(self, latitude: float, longitude: float, distance_km: int) -> DataFrame:
        df = self.hotelStatistics()

        hotel_stats = self.hotelStatistics()

        haversine_udf = udf(haversine, FloatType())

        latitude_col = lit(latitude).cast(FloatType())
        longitude_col = lit(longitude).cast(FloatType())

        hotel_stats = hotel_stats.withColumn("distance_km",
            haversine_udf(latitude_col, longitude_col, col("Latitude"), col("Longitude")))

        hotels_within_distance = (hotel_stats.filter(col("distance_km") <= distance_km)
                                  .select("Hotel_Name", "Latitude", "Longitude","distance_km"))

        return hotels_within_distance
