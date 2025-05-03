from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification") \
    .getOrCreate()

# Load the CSV file
df = spark.read.option("header", "true").csv("fake_news_sample.csv", inferSchema=True)

# Task 2: Text Preprocessing
# Tokenize the text column into words
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(df)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df)

# Convert the array of words into a comma-separated string
def array_to_string(array):
    return ",".join(array)

# Register UDF
array_to_string_udf = udf(array_to_string, StringType())

# Apply the UDF to the 'filtered_words' column
cleaned_df = cleaned_df.withColumn("filtered_words_str", array_to_string_udf(col("filtered_words")))

# Show the cleaned DataFrame with tokenized and stopword-removed text
cleaned_df.select("id", "title", "filtered_words_str", "label").show(5)

# Write the tokenized output to task2_output.csv
cleaned_df.select("id", "title", "filtered_words_str", "label") \
    .write.option("header", "true") \
    .csv("output/task2_output.csv")

# Stop the Spark session
spark.stop()
