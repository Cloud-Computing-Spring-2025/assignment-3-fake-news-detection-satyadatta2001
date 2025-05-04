from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF, StringIndexer
from pyspark.sql.functions import split, col, size, trim
import os

# Step 1: Initialize Spark session
spark = SparkSession.builder.appName("FakeNews_Task3_FeatureExtraction").getOrCreate()

# Step 2: Load input CSV
input_csv = "/workspaces/assignment-3-fake-news-detection-satyadatta2001/output/task2_output.csv/part-00000-d24f7806-64ae-40d6-915e-f06a4811bfdd-c000.csv"
df = spark.read.option("header", True).csv(input_csv)

# Step 3: Split stringified word list into array
df = df.withColumn("filtered_words", split(df["filtered_words_str"], ", ")).drop("filtered_words_str")

# Step 4: Filter invalid rows
df = df.filter((size(col("filtered_words")) > 0) & (col("label").isNotNull()) & (trim(col("label")) != ""))

# Step 5: TF-IDF pipeline
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=50000)
featurizedData = hashingTF.transform(df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Step 6: Index label column
indexer = StringIndexer(inputCol="label", outputCol="label_index")
indexed = indexer.fit(rescaledData).transform(rescaledData)

# Step 7: Select final columns (✅ includes 'title')
final_df = indexed.select(
    "id",
    "title",  # ✅ included title
    col("filtered_words").cast("string"),
    col("features").cast("string"),
    "label_index"
)

# Step 8: Save to CSV
output_dir = "/workspaces/assignment-3-fake-news-detection-satyadatta2001/output/task3_output"
final_df.write.mode("overwrite").option("header", True).csv(output_dir)

print("✅ Task 3 completed. Output saved to task3_output with 'title' included.")
