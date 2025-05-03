from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# Load the CSV file
df = spark.read.option("header", "true").csv("fake_news_sample.csv", inferSchema=True)

# Create Temporary View
df.createOrReplaceTempView("news_data")

# Show the first 5 rows
df.show(5)

# Count total number of articles
article_count = df.count()
print(f"Total number of articles: {article_count}")

# Retrieve distinct labels (FAKE or REAL)
distinct_labels = df.select("label").distinct().show()

# Write the DataFrame to task1_output.csv
df.write.option("header", "true").csv("output/task1_output.csv")
