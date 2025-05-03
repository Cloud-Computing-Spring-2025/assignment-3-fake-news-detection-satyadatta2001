from pyspark.ml.feature import HashingTF, IDF, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import split, concat_ws, udf
from pyspark.sql.types import StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FakeNewsClassification") \
    .getOrCreate()

# Load the cleaned CSV file (from Task 2)
df = spark.read.option("header", "true").csv("output/task2_output.csv", inferSchema=True)

# Convert the filtered_words_str column (string) into an array of words (tokens)
df = df.withColumn("filtered_words_array", split(df["filtered_words_str"], " "))

# Task 3: Feature Extraction

# 1. HashingTF: Convert text into term frequency (TF)
hashingTF = HashingTF(inputCol="filtered_words_array", outputCol="raw_features", numFeatures=10000)

# 2. IDF: Apply Inverse Document Frequency (IDF) to get TF-IDF features
idf = IDF(inputCol="raw_features", outputCol="features")

# 3. Label Indexing: Convert "FAKE" and "REAL" into 0 and 1
indexer = StringIndexer(inputCol="label", outputCol="label_index")

# 4. Assemble all features into a single feature vector (optional if needed for specific models)
assembler = VectorAssembler(inputCols=["features"], outputCol="final_features")

# Create a pipeline with all stages
pipeline = Pipeline(stages=[hashingTF, idf, indexer, assembler])

# Fit the pipeline and transform the data
model = pipeline.fit(df)
result = model.transform(df)

# Convert the array of filtered words to a string
result = result.withColumn("filtered_words_str", concat_ws(" ", result["filtered_words_array"]))

# Convert vector features to string (CSV-compatible)
def vector_to_str(vector):
    return ",".join(map(str, vector.toArray()))

# Register UDF for converting vector to string
vector_to_str_udf = udf(vector_to_str, StringType())

# Apply the UDF to convert 'features' column to a string
result = result.withColumn("features_str", vector_to_str_udf(result["features"]))

# Show the resulting features and labels (for debugging purposes)
result.select("id", "filtered_words_str", "features_str", "label_index").show(5)

# Save the resulting features and labels to task3_output.csv
result.select("id", "filtered_words_str", "features_str", "label_index") \
    .write.option("header", "true") \
    .csv("output/task3_output.csv")

# Stop the Spark session
spark.stop()
