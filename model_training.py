from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql.functions import udf, col
from pyspark.sql.types import DoubleType
import re

# Step 1: Start Spark session
spark = SparkSession.builder.appName("FakeNews_Task4_LogisticRegression").getOrCreate()

# Step 2: Load Task 3 output
input_path = "/workspaces/assignment-3-fake-news-detection-satyadatta2001/output/task3_output"
df = spark.read.option("header", True).csv(input_path)

# Step 3: Parse TF-IDF string into SparseVector
def parse_sparse_vector(v_str):
    match = re.match(r"\((\d+),\[(.*?)\],\[(.*?)\]\)", v_str)
    if not match:
        return SparseVector(0, [], [])
    size = int(match.group(1))
    indices = list(map(int, match.group(2).split(','))) if match.group(2) else []
    values = list(map(float, match.group(3).split(','))) if match.group(3) else []
    return SparseVector(size, indices, values)

parse_sparse_udf = udf(parse_sparse_vector, VectorUDT())
df = df.withColumn("features_vec", parse_sparse_udf("features"))
df = df.withColumn("label_index", col("label_index").cast(DoubleType()))

# Step 4: Split into train/test
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# Step 5: Train Logistic Regression
lr = LogisticRegression(featuresCol="features_vec", labelCol="label_index", maxIter=100)
model = lr.fit(train_data)

# Step 6: Predict
predictions = model.transform(test_data)

# Step 7: Select final output columns (with title)
output_df = predictions.select(
    col("id").cast("string"),
    col("title"),
    col("label_index").cast("double"),
    col("prediction").cast("double")
)

# Step 8: Save output
output_path = "/workspaces/assignment-3-fake-news-detection-satyadatta2001/output/task4_output_lr"
output_df.write.mode("overwrite").option("header", True).csv(output_path)

# Step 9: Accuracy
evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"✅ Logistic Regression Accuracy: {accuracy:.4f}")
print("📄 Task 4 predictions saved to task4_output_lr")
