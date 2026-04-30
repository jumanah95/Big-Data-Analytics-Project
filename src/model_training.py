"""
model_training.py
-----------------------------------------
Responsible for: Building the ML preprocessing pipeline and training all three models.

What this file does:
  Builds a full Spark MLlib Pipeline with feature preprocessing stages,
  then trains three classifiers. Key improvements over the original:

  CLASS IMBALANCE HANDLING (critical fix):
    The diabetes dataset has ~91% "not readmitted" vs ~9% "readmitted <30 days".
    Without correction, all models will just predict 0 every time to get 91%
    accuracy while failing to detect any actual readmissions (useless in a medical
    context). We fix this using a class weight column:
      - Calculate the majority/minority ratio from the training data
      - Add a "class_weight" column: minority class rows get weight = ratio, majority = 1.0
      - Pass weightCol="class_weight" to all classifiers
    This forces the model to penalize errors on the minority class proportionally.

  Models trained:
    - Logistic Regression  (linear baseline)
    - Decision Tree        (interpretable tree model)
    - Random Forest        (ensemble — usually best performer)

  Preprocessing pipeline stages:
    - Imputer              -> fills remaining nulls in numeric cols with median
    - StringIndexer        -> converts categorical strings to integer indices
    - OneHotEncoder        -> converts indices to binary vectors
    - VectorAssembler      -> combines all features into one vector
    - StandardScaler       -> normalizes feature scales (important for LR)

-----------------------------------------
CS4074 - Big Data Analytics
Clinical Outcome Prediction from Noisy Medical Records
Effat University | Spring 2026 | Dr. Naila Marir
"""

import time
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder,
    VectorAssembler, StandardScaler, Imputer
)
from pyspark.ml.classification import (
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
from pyspark.sql import functions as F


def build_preprocessing_stages(df):
    """Build the feature preprocessing stages for the ML pipeline."""

    categorical_cols = [
        c for c in ["gender", "race", "max_glu_serum", "A1Cresult"]
        if c in df.columns
    ]

    numeric_cols = [
        c for c in [
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses",
            "age_ordinal", "total_visits", "procedure_burden",
            "any_med_change", "on_diabetes_med"
        ]
        if c in df.columns
    ]

    # Imputer: fills remaining null values in numeric columns with the median
    imputer = Imputer(
        inputCols=numeric_cols,
        outputCols=[f"{c}_imp" for c in numeric_cols],
        strategy="median"
    )

    # StringIndexer: converts categorical strings to numeric indices
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]

    # OneHotEncoder: converts numeric indices to binary vectors
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe")
        for c in categorical_cols
    ]

    # VectorAssembler: combines all feature columns into a single vector
    assembler = VectorAssembler(
        inputCols=[f"{c}_imp" for c in numeric_cols] +
                  [f"{c}_ohe" for c in categorical_cols],
        outputCol="features_raw",
        handleInvalid="keep"
    )

    # StandardScaler: normalizes feature scales (important for Logistic Regression)
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )

    stages = indexers + encoders + [imputer, assembler, scaler]
    return stages


def split_data(df):
    """Split the dataset: 80% training, 20% testing."""
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    print(f"\n[6] Data Split:")
    print(f"   Train : {train_df.count():,} rows")
    print(f"   Test  : {test_df.count():,} rows")
    return train_df, test_df


def add_class_weights(train_df):
    """
    Compute and add a class_weight column to handle the ~91%/9% class imbalance.

    Why this is needed:
      Without weighting, models achieve ~91% accuracy by always predicting 0
      (not readmitted). They learn nothing useful about the minority class.
      The weight column tells the optimizer: "getting a minority-class row wrong
      costs {ratio}x more than getting a majority-class row wrong."

    How it works:
      weight = majority_count / minority_count for the minority class (label=1)
      weight = 1.0 for the majority class (label=0)
    """
    majority_count = train_df.filter(F.col("label") == 0).count()
    minority_count = train_df.filter(F.col("label") == 1).count()

    # Guard against division by zero
    if minority_count == 0:
        print("   WARNING: No minority class (label=1) found. Skipping class weighting.")
        return train_df.withColumn("class_weight", F.lit(1.0)), 1.0

    balance_ratio = majority_count / minority_count

    train_df = train_df.withColumn(
        "class_weight",
        F.when(F.col("label") == 1, balance_ratio).otherwise(1.0)
    )

    print(f"\n[6b] Class Imbalance Handling:")
    print(f"   Majority class (label=0 / not readmitted) : {majority_count:,} rows")
    print(f"   Minority class (label=1 / readmitted <30) : {minority_count:,} rows")
    print(f"   Imbalance ratio                           : {balance_ratio:.2f}:1")
    print(f"   -> Minority class rows given weight = {balance_ratio:.2f}")
    print(f"   -> Majority class rows given weight = 1.0")

    return train_df, balance_ratio


def train_all_models(prep_stages, train_df, test_df):
    """Train all three models and return metrics and trained model objects."""

    # Add class weights to training data before building models
    train_df_weighted, balance_ratio = add_class_weights(train_df)

    models_config = {
        "Logistic Regression": LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",       # class imbalance fix
            maxIter=20,
            regParam=0.01
        ),
        "Decision Tree": DecisionTreeClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",       # class imbalance fix
            maxDepth=6,
            seed=42
        ),
        "Random Forest": RandomForestClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",       # class imbalance fix
            numTrees=50,
            maxDepth=8,
            seed=42
        ),
    }

    binary_eval = BinaryClassificationEvaluator(
        labelCol="label", metricName="areaUnderROC"
    )
    acc_eval  = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")
    f1_eval   = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")
    prec_eval = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedPrecision")
    rec_eval  = MulticlassClassificationEvaluator(labelCol="label", metricName="weightedRecall")

    results        = {}
    trained_models = {}

    for name, classifier in models_config.items():
        print(f"\n[7] Training: {name}...")
        start = time.time()

        # Pipeline trains on weighted training data
        pipeline = Pipeline(stages=prep_stages + [classifier])
        model    = pipeline.fit(train_df_weighted)

        # Evaluation is on test data (no weights — fair comparison)
        preds    = model.transform(test_df)
        elapsed  = time.time() - start

        results[name] = {
            "AUC-ROC":        round(binary_eval.evaluate(preds), 4),
            "Accuracy":       round(acc_eval.evaluate(preds), 4),
            "F1 Score":       round(f1_eval.evaluate(preds), 4),
            "Precision":      round(prec_eval.evaluate(preds), 4),
            "Recall":         round(rec_eval.evaluate(preds), 4),
            "Train Time (s)": round(elapsed, 2),
        }
        trained_models[name] = model

        m = results[name]
        print(f"   AUC : {m['AUC-ROC']}  |  Acc : {m['Accuracy']}  "
              f"|  F1 : {m['F1 Score']}  |  Time : {m['Train Time (s)']}s")

    return results, trained_models
