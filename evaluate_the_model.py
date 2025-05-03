from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark session
spark = SparkSession.builder.appName("FakeNewsEvaluation").getOrCreate()

# Load prediction results from task4 output
predictions = spark.read.option("header", "true").csv("output/task4_output.csv")

# Cast columns to correct types if needed
from pyspark.sql.functions import col
predictions = predictions.withColumn("label_index", col("label_index").cast("double"))
predictions = predictions.withColumn("prediction", col("prediction").cast("double"))

# Evaluators
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="accuracy"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_index", predictionCol="prediction", metricName="f1"
)

# Calculate metrics
accuracy = accuracy_evaluator.evaluate(predictions)
f1_score = f1_evaluator.evaluate(predictions)

# Show as markdown
print("\n### Model Evaluation (Task 5)\n")
print("| Metric   | Value |")
print("|----------|-------|")
print(f"| Accuracy | {accuracy:.2f} |")
print(f"| F1 Score | {f1_score:.2f} |")

# Save as CSV
from pyspark.sql import Row
metrics = [Row(Metric="Accuracy", Value=round(accuracy, 4)),
           Row(Metric="F1 Score", Value=round(f1_score, 4))]

metrics_df = spark.createDataFrame(metrics)
metrics_df.write.option("header", "true").mode("overwrite").csv("output/task5_output.csv")
