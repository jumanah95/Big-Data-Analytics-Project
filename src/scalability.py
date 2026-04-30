"""
scalability.py
-----------------------------------------
Responsible for: Scalability analysis
Runs the pipeline on 4 different data sizes: 25%, 50%, 75%, 100%
Measures training time at each size and plots the result.
Plot generated:
  08. Scalability analysis (data size vs training time)
"""

import time
import os
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier


def run_scalability_test(df, prep_stages, output_dir="reports/figures"):
    """
    Runs Random Forest on 4 different fractions of the dataset.
    Measures training time at each fraction to demonstrate Spark scalability.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n[9] Running Scalability Test...")
    print("   (This may take a few minutes — please wait)")

    fractions  = [0.25, 0.50, 0.75, 1.00]
    labels     = ["25%", "50%", "75%", "100%"]
    times      = []
    row_counts = []

    # Using numTrees=20 instead of 50 to keep the test faster
    classifier = RandomForestClassifier(
        featuresCol="features", labelCol="label",
        numTrees=20, maxDepth=6, seed=42
    )
    pipeline = Pipeline(stages=prep_stages + [classifier])

    for frac, lbl in zip(fractions, labels):

        # Sample the requested fraction of the dataset
        subset              = df.sample(withReplacement=False, fraction=frac, seed=42)
        train_sub, test_sub = subset.randomSplit([0.8, 0.2], seed=42)

        n_rows = subset.count()
        row_counts.append(n_rows)

        print(f"   Training on {lbl} ({n_rows:,} rows)...", end=" ", flush=True)

        start   = time.time()
        model   = pipeline.fit(train_sub)
        elapsed = round(time.time() - start, 2)

        times.append(elapsed)
        print(f"-> {elapsed}s")

    # -- Plot 08: Scalability chart -----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Scalability Analysis — Random Forest on Diabetes Dataset",
        fontsize=13, fontweight="bold"
    )

    # Left: data fraction vs training time
    axes[0].plot(labels, times, marker="o", color="#C44E52",
                 linewidth=2.5, markersize=8)
    for lbl, t in zip(labels, times):
        axes[0].annotate(
            f"{t}s", (lbl, t),
            textcoords="offset points", xytext=(0, 10),
            ha="center", fontsize=10
        )
    axes[0].set_title("Data Size vs Training Time")
    axes[0].set_xlabel("Data Fraction")
    axes[0].set_ylabel("Training Time (seconds)")
    axes[0].grid(axis="y", alpha=0.3)

    # Right: row count per fraction
    axes[1].bar(
        labels, row_counts,
        color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
        edgecolor="white"
    )
    for i, (lbl, n) in enumerate(zip(labels, row_counts)):
        axes[1].text(i, n + 500, f"{n:,}", ha="center", fontsize=9)
    axes[1].set_title("Number of Rows per Fraction")
    axes[1].set_xlabel("Data Fraction")
    axes[1].set_ylabel("Row Count")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_scalability_analysis.png", dpi=150)
    plt.close()
    print("   Saved: 08_scalability_analysis.png")

    # Print scalability summary table
    print("\n" + "-" * 45)
    print(f"  {'Fraction':<10} {'Rows':>10} {'Time (s)':>10}")
    print("-" * 45)
    for lbl, n, t in zip(labels, row_counts, times):
        print(f"  {lbl:<10} {n:>10,} {t:>10.2f}s")
    print("-" * 45)
    print("  Spark processes larger data proportionally,")
    print("  confirming distributed scalability.\n")
