"""
noise_injection.py
-----------------------------------------
Responsible for: Simulating adversarial data quality degradation.

What this file does:
  This is the core of the project's adversarial theme. It intentionally
  corrupts the raw dataset in four realistic ways before the cleaning
  step runs, then reports how much damage was done.

  The idea is: we PROVE our pipeline is resilient by showing it can
  recover from deliberate corruption. This directly addresses the
  project meta-theme: "Resilient AI Pipelines in Adversarial Big-Data
  Ecosystems" -> "Data Quality Degradation".

  Four adversarial attacks simulated:
  -----------------------------------------------------------------------
  Attack 1 — NULL injection into age column (10% of rows)
    Simulates sensor dropout or incomplete patient intake forms.
    Real-world: hospital systems often have missing demographics.

  Attack 2 — Sentinel value injection into race column (8% of rows)
    Replaces values with "?" — the exact sentinel that already exists
    in the real dataset. Amplifies the existing noise.

  Attack 3 — Label flipping in readmitted column (5% of rows)
    Simulates data entry errors or ETL pipeline bugs where readmission
    status is recorded incorrectly. This is the most damaging attack
    because it directly corrupts the target variable.

  Attack 4 — Extreme outlier injection into time_in_hospital (3% of rows)
    Replaces hospital stay duration with impossibly large values (999).
    Simulates sensor/logging errors in clinical systems.
  -----------------------------------------------------------------------

  After all attacks, the function prints a summary of how many rows
  were corrupted per attack and the overall corruption rate.
  The downstream clean_data() function then handles all of these.

-----------------------------------------
CS4074 - Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Effat University | Spring 2026 | Dr. Naila Marir
"""

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


# Seed for reproducibility — same corruption every run so results are comparable
NOISE_SEED = 2026


def inject_adversarial_noise(df):
    """
    Inject four types of adversarial noise into the raw DataFrame.
    Returns the corrupted DataFrame and a summary dict of corruption stats.
    """
    print("\n[2] Injecting Adversarial Noise (Data Quality Degradation)...")
    print("   Simulating real-world pipeline attacks on the raw data...")

    total_rows = df.count()
    summary = {}

    # ------------------------------------------------------------------
    # Attack 1: NULL injection into 'age' column (10% of rows)
    # Simulates: incomplete patient intake forms / sensor dropout
    # ------------------------------------------------------------------
    df = df.withColumn(
        "age",
        F.when(
            F.rand(seed=NOISE_SEED) < 0.10,   # 10% probability
            None                                # inject NULL
        ).otherwise(F.col("age"))
    )
    null_age = df.filter(F.col("age").isNull()).count()
    summary["Attack 1 - NULL in age (10%)"] = null_age
    print(f"   Attack 1 done: Injected NULL into 'age'   -> {null_age:,} rows corrupted")

    # ------------------------------------------------------------------
    # Attack 2: Sentinel '?' injection into 'race' column (8% of rows)
    # Simulates: ETL pipeline importing from a legacy system that uses
    #            '?' as a missing-value marker
    # ------------------------------------------------------------------
    df = df.withColumn(
        "race",
        F.when(
            F.rand(seed=NOISE_SEED + 1) < 0.08,
            "?"
        ).otherwise(F.col("race"))
    )
    sentinel_race = df.filter(F.col("race") == "?").count()
    summary["Attack 2 - Sentinel '?' in race (8%)"] = sentinel_race
    print(f"   Attack 2 done: Injected '?' into 'race'   -> {sentinel_race:,} rows corrupted")

    # ------------------------------------------------------------------
    # Attack 3: Label flipping in 'readmitted' column (5% of rows)
    # Simulates: data entry errors or buggy ETL transformation rules
    # This is the most adversarial — it corrupts the target variable
    # ------------------------------------------------------------------
    df = df.withColumn(
        "readmitted",
        F.when(
            (F.rand(seed=NOISE_SEED + 2) < 0.05) & (F.col("readmitted") == "<30"),
            ">30"       # flip a readmitted-within-30 to not-within-30
        ).when(
            (F.rand(seed=NOISE_SEED + 3) < 0.05) & (F.col("readmitted") == "NO"),
            "<30"       # flip a non-readmitted to readmitted
        ).otherwise(F.col("readmitted"))
    )
    summary["Attack 3 - Label flipping in readmitted (5%)"] = int(total_rows * 0.05)
    print(f"   Attack 3 done: Flipped labels in 'readmitted'  -> ~{int(total_rows * 0.05):,} rows corrupted")

    # ------------------------------------------------------------------
    # Attack 4: Outlier injection into 'time_in_hospital' (3% of rows)
    # Simulates: sensor/logging errors producing impossible values
    # Real hospital stays are 1-14 days; 999 is clearly adversarial
    # ------------------------------------------------------------------
    df = df.withColumn(
        "time_in_hospital",
        F.when(
            F.rand(seed=NOISE_SEED + 4) < 0.03,
            F.lit(999).cast(IntegerType())      # impossible outlier
        ).otherwise(F.col("time_in_hospital"))
    )
    outlier_count = df.filter(F.col("time_in_hospital") == 999).count()
    summary["Attack 4 - Outliers in time_in_hospital (3%)"] = outlier_count
    print(f"   Attack 4 done: Outlier (999) in 'time_in_hospital' -> {outlier_count:,} rows corrupted")

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    total_corrupted = sum(summary.values())
    corruption_rate = (total_corrupted / total_rows) * 100

    print(f"\n   {'='*50}")
    print(f"   ADVERSARIAL NOISE INJECTION SUMMARY")
    print(f"   {'='*50}")
    print(f"   Total rows in dataset : {total_rows:,}")
    for attack, count in summary.items():
        print(f"   {attack:<45}: {count:,}")
    print(f"   Estimated corrupted rows : ~{total_corrupted:,}")
    print(f"   Estimated corruption rate: ~{corruption_rate:.1f}%")
    print(f"   {'='*50}")
    print(f"   -> Passing corrupted data to clean_data() to test pipeline resilience...")

    return df, summary
