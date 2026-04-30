"""
data_loader.py
-----------------------------------------
Responsible for: Loading the dataset from a local CSV into a Spark DataFrame

What this file does:
  - Reads diabetic_data.csv directly from the laptop's local file system.
  - Prints the row count, column count, and partition count.
-----------------------------------------
CS4074 - Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Effat University | Spring 2026 | Dr. Naila Marir
"""

def load_data(spark, path=r"C:\Users\iqrai\Nafeesa's Project\diabetic_data.csv"):
    """
    Load diabetic_data.csv from the local file system.
    Make sure the 'path' matches exactly where the CSV is saved on the laptop.
    """
    print("\n[1] Loading dataset from local file system...")
    print(f"   Path: {path}")

    # Read the CSV locally
    df = spark.read.csv(path, header=True, inferSchema=True)

    # Get the number of partitions created by Spark
    num_partitions = df.rdd.getNumPartitions()

    print(f"   Rows       : {df.count():,}")
    print(f"   Columns    : {len(df.columns)}")
    print(f"   Partitions : {num_partitions}  <- Spark processes these in parallel")

    return df