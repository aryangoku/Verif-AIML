# agents/eda_agent.py

import pandas as pd
import numpy as np


class EDAAgent:
    """
    Perceives dataset statistics and target distribution.
    Reasons about class imbalance, skewness, and feature patterns.
    Recommends evaluation metric and split strategy.
    """

    def __init__(self, memory_store):
        self.memory = memory_store

    def run(self):
        """
        AGENT ROLE: Reasoning Agent
        ReAct Cycle: REASON
        - Perceives: Audit report and dataset from shared memory
        - Reasons: Analyses patterns, detects imbalance, recommends metrics
        - Acts: Writes EDA report and task type to shared memory
        - Observes: Recommends evaluation strategy
        """
        print("\n[EDA Agent]  REASON -- Starting exploratory data analysis...")

        df = self.memory.get("dataset")
        target_col = self.memory.get("target_column")

        if df is None:
            raise ValueError("No dataset found in memory.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset.")

        report = {}

        # Detect task type (ignore missing targets)
        target_series = df[target_col].dropna()
        if len(target_series) == 0:
            raise ValueError(
                f"Target column '{target_col}' has no valid values."
            )
        unique_vals = target_series.nunique()

        # Edge case: text column with too many unique values
        if target_series.dtype == "object" and unique_vals > 50:
            raise ValueError(
                f"Target column '{target_col}' has {unique_vals} unique text values. "
                f"This is not a valid target column. "
                f"Please select a numerical or binary/categorical column."
            )

        if unique_vals <= 20 and target_series.dtype in ["object", "bool", "int64"]:
            task_type = "classification"
        else:
            task_type = "regression"

        report["task_type"] = task_type
        self.memory.set("task_type", task_type)

        # Feature types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if target_col in numerical_cols:
            numerical_cols.remove(target_col)
        if target_col in categorical_cols:
            categorical_cols.remove(target_col)

        report["numerical_columns"] = numerical_cols
        report["categorical_columns"] = categorical_cols

        # Class imbalance check (classification only)
        report["class_imbalance"] = False
        report["class_distribution"] = {}

        if task_type == "classification":
            class_counts = target_series.value_counts()
            report["class_distribution"] = class_counts.to_dict()
            majority = class_counts.iloc[0]
            minority = class_counts.iloc[-1]
            imbalance_ratio = majority / minority if minority > 0 else 999
            report["imbalance_ratio"] = round(imbalance_ratio, 2)

            if imbalance_ratio > 3:
                report["class_imbalance"] = True
                print(
                    f"  [EDA Agent] Class imbalance detected! "
                    f"Ratio: {imbalance_ratio:.2f}"
                )

        # Skewness check for numerical columns
        skewed_cols = []
        for col in numerical_cols:
            try:
                skew = df[col].skew()
                if abs(skew) > 1.5:
                    skewed_cols.append(col)
            except:
                pass
        report["skewed_columns"] = skewed_cols

        # Feature correlations with target
        correlations = {}
        for col in numerical_cols:
            try:
                corr = round(df[col].corr(target_series), 4)
                correlations[col] = corr
            except:
                pass
        report["feature_correlations"] = correlations

        # Recommend evaluation metric
        if task_type == "classification":
            if report["class_imbalance"]:
                recommended_metric = "f1"
                print("  [EDA Agent] Recommending F1-score due to class imbalance.")
            else:
                recommended_metric = "accuracy"
        else:
            recommended_metric = "rmse"

        report["recommended_metric"] = recommended_metric
        self.memory.set("recommended_metric", recommended_metric)

        # Recommend train/test split strategy
        if df.shape[0] < 500:
            report["split_strategy"] = "stratified_kfold_5"
        else:
            report["split_strategy"] = "train_test_split_80_20"

        # Store report
        self.memory.set("eda_report", report)

        print(f"[EDA Agent] EDA complete.")
        print(f"  Task type: {task_type}")
        print(
            f"  Numerical cols: {len(numerical_cols)}, "
            f"Categorical cols: {len(categorical_cols)}"
        )
        print(f"  Recommended metric: {recommended_metric}")
        print(f"  Skewed columns: {skewed_cols}")

        return report