# agents/feature_engineer_agent.py

import pandas as pd
import numpy as np


class FeatureEngineerAgent:
    """
    AGENT ROLE: Feature Engineering Agent
    ReAct Cycle: PLAN → ACT
    - Perceives: Dataset and EDA report from shared memory
    - Reasons: LLM decides which features to create
    - Acts: Creates new features based on LLM suggestions
    - Observes: Updates dataset in shared memory
    """

    def __init__(self, memory_store):
        self.memory = memory_store

    def run(self):
        print(
            "\n[Feature Engineer Agent] "
            "Starting feature engineering..."
        )

        df = self.memory.get("dataset")
        target_col = self.memory.get("target_column")
        eda_report = self.memory.get("eda_report") or {}
        llm_feature_plan = (
            self.memory.get("llm_feature_plan") or {}
        )

        if df is None:
            raise ValueError("No dataset found in memory.")

        original_cols = len(df.columns)
        new_features_created = []

        # ── Apply LLM suggested features ────────────────────────
        suggested_features = llm_feature_plan.get("features", [])

        for feature in suggested_features:
            if not isinstance(feature, dict):
                continue
            try:
                feat_name = feature.get("name")
                feat_type = feature.get("type")
                col1 = feature.get("col1")
                col2 = feature.get("col2")
                operation = feature.get("operation")

                if col1 and col1 not in df.columns:
                    continue
                if col2 and col2 not in df.columns:
                    continue
                if col1 == target_col or col2 == target_col:
                    continue

                if feat_type == "interaction" and col1 and col2:
                    if operation == "multiply":
                        df[feat_name] = df[col1] * df[col2]
                    elif operation == "divide":
                        df[feat_name] = df[col1] / (
                            df[col2] + 1e-8
                        )
                    elif operation == "add":
                        df[feat_name] = df[col1] + df[col2]
                    elif operation == "subtract":
                        df[feat_name] = df[col1] - df[col2]
                    new_features_created.append(feat_name)
                    print(
                        f"  Created: {feat_name} = "
                        f"{col1} {operation} {col2}"
                    )

                elif feat_type == "aggregate" and col1:
                    if operation == "square":
                        df[feat_name] = df[col1] ** 2
                    elif operation == "sqrt":
                        df[feat_name] = np.sqrt(
                            df[col1].clip(lower=0)
                        )
                    elif operation == "log":
                        df[feat_name] = np.log1p(
                            df[col1].clip(lower=0)
                        )
                    new_features_created.append(feat_name)
                    print(
                        f"  Created: {feat_name} = "
                        f"{operation}({col1})"
                    )

            except Exception as e:
                print(f"  Feature creation failed: {e}")
                continue

        # ── Smart domain features ────────────────────────────────

        # Family size for Titanic-like datasets
        if ("SibSp" in df.columns and
                "Parch" in df.columns and
                "Family_Size" not in df.columns and
                target_col not in ["SibSp", "Parch"]):
            df["Family_Size"] = (
                df["SibSp"] + df["Parch"] + 1
            )
            df["Is_Alone"] = (
                df["Family_Size"] == 1
            ).astype(int)
            new_features_created.extend(
                ["Family_Size", "Is_Alone"]
            )
            print("  Created: Family_Size, Is_Alone")

        # Age bins
        if ("Age" in df.columns and
                "Age_Group" not in df.columns and
                target_col != "Age"):
            df["Age_Group"] = pd.cut(
                df["Age"].fillna(df["Age"].median()),
                bins=[0, 12, 18, 35, 60, 100],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
            new_features_created.append("Age_Group")
            print("  Created: Age_Group")

        # Title from Name
        if ("Name" in df.columns and
                "Title" not in df.columns and
                target_col != "Name"):
            try:
                df["Title"] = df["Name"].str.extract(
                    r" ([A-Za-z]+)\.", expand=False
                )
                title_map = {
                    "Mr": 1, "Miss": 2, "Mrs": 3,
                    "Master": 4, "Dr": 5, "Rev": 6
                }
                df["Title"] = df["Title"].map(
                    title_map
                ).fillna(0)
                new_features_created.append("Title")
                print("  Created: Title (from Name)")
            except Exception:
                pass

        # Clean invalid numeric values from engineered features
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        # Update dataset in memory
        self.memory.set("dataset", df)

        new_cols = len(df.columns) - original_cols
        print(
            f"[Feature Engineer Agent] Done. "
            f"Created {new_cols} new features: "
            f"{new_features_created}"
        )

        feature_report = {
            "original_columns": original_cols,
            "new_features": new_features_created,
            "total_features_added": len(new_features_created),
            "llm_suggested": suggested_features
        }

        self.memory.set("feature_report", feature_report)
        return feature_report