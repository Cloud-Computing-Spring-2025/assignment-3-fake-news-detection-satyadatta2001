from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import lit
import os

# Step 1: Start Spark session
spark = SparkSession.builder.appName("FakeNews_Task5_Evaluation").getOrCreate()

# Step 2: Load predictions
input_path = "/workspaces/assignment-3-fake-news-detection-satyadatta2001/output/task4_output_lr"
df = spark.read.option("header", True).csv(input_path)

# Step 3: Cast label and prediction to double
from pyspark.sql.functions import col
df = df.withColumn("label_index", col("label_index").cast("double"))
df = df.withColumn("prediction", col("prediction").cast("double"))

# Step 4: Evaluate
acc_eval = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
f1_eval = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = acc_eval.evaluate(df)
f1_score = f1_eval.evaluate(df)

# Step 5: Display as Markdown table
print("\n### 📊 Model Evaluation")
print("| Metric   | Value |")
print("|----------|--------|")
print(f"| Accuracy | {accuracy:.2f} |")
print(f"| F1 Score | {f1_score:.2f} |")

# Step 6: Save to CSV
result_df = spark.createDataFrame([
    ("Accuracy", round(accuracy, 4)),
    ("F1 Score", round(f1_score, 4))
], ["Metric", "Value"])

output_path = "/workspaces/assignment-3-fake-news-detection-satyadatta2001/output/task5_output"
result_df.write.mode("overwrite").option("header", True).csv(output_path)

print("\n✅ Evaluation saved to: task5_output")
