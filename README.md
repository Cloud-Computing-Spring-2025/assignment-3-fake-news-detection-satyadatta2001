# Assignment-5-FakeNews-Detection

##  Overview

This project builds a simple **machine learning pipeline** using **Apache Spark MLlib** to classify news articles as **FAKE** or **REAL** based on their content. The pipeline includes:

- Text preprocessing
- Feature extraction using TF-IDF
- Model training with Logistic Regression
- Evaluation using Accuracy and F1 Score

---

##  Dataset

The dataset used is: `fake_news_sample.csv`  
It contains the following columns:

- `id`: Unique article ID  
- `title`: Headline of the news article  
- `text`: Full text/content of the article  
- `label`: Classification label (`FAKE` or `REAL`)

---

##  Tasks & Scripts

### Task 1: Load & Basic Exploration (`basic_task1.py`)

- Load `fake_news_sample.csv` with inferred schema.
- Create a temporary view: `news_data`.
- Perform:
  - Show first 5 rows  
  - Count total articles  
  - Retrieve distinct labels  
- Output file: `output/task1_output.csv`

---

###  Task 2: Text Preprocessing (`text_processing.py`)

- Convert text to lowercase.
- Tokenize text into words.
- Remove stopwords using `StopWordsRemover`.
- New column created: `filtered_words_str`
- Output file: `output/task2_output.csv`

---

###  Task 3: Feature Extraction (`featured_extraction.py`)

- Use `HashingTF` and `IDF` to generate TF-IDF vectors.
- Label indexed using `StringIndexer`.
- Features stored as string vector in `features_str`
- Output file: `output/task3_output.csv`

---

###  Task 4: Model Training (`model_training.py`)

- Train/test split: 80% / 20%
- Algorithm: `LogisticRegression` from Spark MLlib
- Prediction column added
- Output file: `output/task4_output.csv`

---

###  Task 5: Evaluation (`evaluate_the_model.py`)

- Metrics computed using `MulticlassClassificationEvaluator`
- Results:

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.43  |
| F1 Score  | 0.27  |

- Output file: `output/task5_output.csv`

---

###  Steps

1. Start a SparkSession:
    ```python
    pip install pyspark
    ```
2. Install a faker:
    ```python
    pip install faker
    ```
    

3. Run each task (as `.py` scripts or in Jupyter cells):
   ```python

    python basic_task1.py 
    python text_preprocessing.py
    python feature_extraction.py
    python model_training.py 
    python evaluate_the_model.py 
   
   ```

3. Output will be saved to CSV files (`taskX_output.csv`).

4. Shut down the Spark session:
    ```python
    spark.stop()
    ```

---


