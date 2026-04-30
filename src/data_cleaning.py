from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType


def clean_data(df):
    print("\n[4] Cleaning data (recovering from adversarial noise)...")

    total_before = df.count()

    # -- Step 1: Replace '?' with null --------------------------------
    # ONLY apply to string columns (fixes your crash)
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            col_name = field.name
            df = df.withColumn(
                col_name,
                F.when(F.col(col_name) == "?", None).otherwise(F.col(col_name))
            )

    print("   Step 1 done: Replaced '?' with null (handles Attack 2 sentinel injection)")

    # -- Step 2: Drop columns with more than 40% missing values -------
    total = df.count()
    cols_to_drop = []

    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        if null_count / total > 0.40:
            cols_to_drop.append(col_name)

    df = df.drop(*cols_to_drop)
    print(f"   Step 2 done: Dropped high-missing columns: {cols_to_drop}")

    # -- Step 3: Drop identifier columns ------------------------------
    id_cols = [c for c in ["encounter_id", "patient_nbr"] if c in df.columns]
    df = df.drop(*id_cols)
    print(f"   Step 3 done: Dropped ID columns: {id_cols}")

    # -- Step 4: Drop rows with nulls in critical columns -------------
    critical = [c for c in ["gender", "age", "readmitted"] if c in df.columns]
    df = df.dropna(subset=critical)
    print("   Step 4 done: Dropped rows with null in critical columns")

    # -- Step 5: Remove invalid gender entries ------------------------
    df = df.filter(F.col("gender") != "Unknown/Invalid")
    print("   Step 5 done: Removed invalid gender entries")

    # -- Step 6: Remove duplicate rows --------------------------------
    df = df.dropDuplicates()
    print("   Step 6 done: Removed duplicate rows")

    # -- Step 7: Encode target ----------------------------------------
    df = df.withColumn(
        "label",
        F.when(F.col("readmitted") == "<30", 1)
         .otherwise(0)
         .cast(IntegerType())
    )
    df = df.drop("readmitted")
    print("   Step 7 done: Encoded target")

    # -- Step 8: Cap outliers -----------------------------------------
    if "time_in_hospital" in df.columns:
        p99 = df.approxQuantile("time_in_hospital", [0.99], 0.01)[0]
        df = df.withColumn(
            "time_in_hospital",
            F.when(F.col("time_in_hospital") > p99, p99)
             .otherwise(F.col("time_in_hospital"))
        )
        print(f"   Step 8 done: Capped outliers at {p99}")

    total_after = df.count()
    print(f"\n   Rows before: {total_before:,}")
    print(f"   Rows after : {total_after:,}")

    return df