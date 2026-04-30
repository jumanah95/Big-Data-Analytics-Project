"""
spark_session.py
-----------------------------------------
Responsible for: Creating and returning a configured Spark Session.

What this file does:
  - Builds a SparkSession with all required configuration settings.
  - Points Spark at your local HDFS NameNode so all file reads/writes
    go through HDFS (required by the rubric).
  - Sets shuffle partitions to 8 (suitable for a single-node local cluster).
  - Sets driver memory to 4 GB.
  - Suppresses noisy INFO logs (WARN level only).

HDFS PATH NOTE:
  The line fs.defaultFS tells Spark where your HDFS NameNode lives.
  Default value here is hdfs://localhost:9000 — this is correct for most
  single-node Hadoop setups. If yours is different, change HDFS_NAMENODE
  at the top of this file.
-----------------------------------------
CS4074 - Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Effat University | Spring 2026 | Dr. Naila Marir
"""

from pyspark.sql import SparkSession

# ---------------------------------------------------------------
# CONFIGURE THIS: Change if your NameNode runs on a different host/port.
# Run this in your Ubuntu terminal to check:  hdfs getconf -confKey fs.defaultFS
# ---------------------------------------------------------------
HDFS_NAMENODE = "file:///"


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("CS4074_DiabetesReadmission")
        .master("local[*]")                                      # use all CPU cores
        .config("spark.sql.shuffle.partitions", "8")             # 8 is enough for ~100k rows
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "2g")
        #.config("spark.hadoop.fs.defaultFS", HDFS_NAMENODE)      # point Spark at HDFS
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"   Spark version  : {spark.version}")
    print(f"   HDFS NameNode  : {HDFS_NAMENODE}")
    print(f"   Master         : {spark.sparkContext.master}")
    print(f"   Spark UI       : http://localhost:4040")
    return spark
