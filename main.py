"""
main.py
-----------------------------------------
Main entry point for the CS4074 project pipeline.
Calls all modules in order.

FULL PIPELINE ORDER:
  Step 1:  Create Spark Session (points to HDFS)
  Step 2:  Load raw data from HDFS
  Step 3:  Run EDA on RAW data (before noise — shows clean baseline)
  Step 4:  Inject adversarial noise (Data Quality Degradation attack)
  Step 5:  Clean data (proves pipeline resilience)
  Step 6:  Feature engineering
  Step 7:  Cache cleaned dataset
  Step 8:  Build preprocessing stages + split data
  Step 9:  Train all three models (with class imbalance handling)
  Step 10: Visualize results + print summary
  Step 11: Plot confusion matrices
  Step 12: Scalability analysis

WHY THIS ORDER MATTERS:
  EDA runs on RAW data (Step 3) so we see the true class distribution
  and missing values BEFORE corruption. After noise injection (Step 4)
  the same stats would look different — showing the adversarial effect.
  Then clean_data (Step 5) recovers the pipeline.

-----------------------------------------
CS4074 - Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Effat University | Spring 2026 | Dr. Naila Marir
"""

from src.spark_session       import create_spark_session
from src.data_loader         import load_data
from src.eda                 import run_eda
from src.noise_injection     import inject_adversarial_noise   # NEW
from src.data_cleaning       import clean_data
from src.feature_engineering import engineer_features
from src.model_training      import build_preprocessing_stages, split_data, train_all_models
from src.results             import visualize_results, print_summary, plot_confusion_matrices
from src.scalability         import run_scalability_test


def main():
    print("=" * 60)
    print("  CS4074 - Diabetes Readmission Prediction Pipeline")
    print("  Resilient AI Pipeline in Adversarial Big-Data Ecosystems")
    print("=" * 60)

    # Step 1: Initialize Spark Session (configured for HDFS)
    print("\n[1] Initializing Spark Session...")
    spark = create_spark_session()

    # Step 2: Load data from HDFS
    df = load_data(spark)

    # Step 3: EDA on RAW data (before noise injection — shows clean baseline)
    print("\n[3] Running EDA on raw data (pre-noise baseline)...")
    run_eda(df)

    # Step 4: Inject adversarial noise (the adversarial challenge)
    df, noise_summary = inject_adversarial_noise(df)

    # Step 5: Clean data (proves pipeline resilience by recovering from noise)
    df = clean_data(df)

    # Step 6: Feature engineering
    print("\n[5] Engineering features...")
    df = engineer_features(df)

    # Step 7: Cache the cleaned+engineered dataset in memory
    print("\n[6] Caching dataset in memory...")
    df.cache()
    cached_count = df.count()
    print(f"   Cached {cached_count:,} rows in Spark memory")

    # Step 8: Build preprocessing stages and split data
    prep_stages         = build_preprocessing_stages(df)
    train_df, test_df   = split_data(df)

    # Step 9: Train all three models (with class imbalance handling)
    results, trained_models = train_all_models(prep_stages, train_df, test_df)

    # Step 10: Generate result plots and print final summary table
    print("\n[8] Generating result visualizations...")
    visualize_results(results)
    print_summary(results)

    # Step 11: Plot confusion matrix for each model
    plot_confusion_matrices(trained_models, test_df)

    # Step 12: Scalability analysis (trains on 25/50/75/100% of data)
    run_scalability_test(df, prep_stages)

    spark.stop()
    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print("  Check reports/figures/ for all output plots.")
    print("  Adversarial noise summary:")
    for attack, count in noise_summary.items():
        print(f"    {attack}: {count:,} rows corrupted")
    print("=" * 60)


if __name__ == "__main__":
    main()
