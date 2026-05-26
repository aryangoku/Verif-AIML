# agents/pipeline_builder.py

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import RobustScaler


class PipelineBuilderAgent:
    """
    Perceives feature types and audit results.
    Reasons about imputation, encoding, and scaling choices.
    Builds a unified Scikit-learn preprocessing pipeline.
    """

    def __init__(self, memory_store):
        self.memory = memory_store

    def run(self):
        """
        AGENT ROLE: Planning Agent
        ReAct Cycle: PLAN -> ACT
        - Perceives: EDA report and audit findings from shared memory
        - Reasons: Decides imputation, encoding, scaling strategies
        - Acts: Builds sklearn preprocessing pipeline
        - Observes: Validates pipeline structure
        """
        print("\n[Pipeline Builder Agent]  PLAN -> ACT -- Building preprocessing pipeline...")

        df = self.memory.get("dataset")
        target_col = self.memory.get("target_column")
        audit_report = self.memory.get("audit_report") or {}
        eda_report = self.memory.get("eda_report") or {}
        suspicious_columns = self.memory.get("suspicious_columns") or []

        if df is None:
            raise ValueError("No dataset found in memory.")

        # ── Drop columns only from Verifier fix plan ────────────────
        # Suspicious columns are NOT dropped on first run.
        # They are only dropped after the Verifier detects them
        # and sends a fix plan — this is what triggers self-healing.
        verifier_decision = self.memory.get("verifier_decision") or {}
        verifier_fix_columns = []

        for fix in verifier_decision.get("fix_plan", []):
            if not isinstance(fix, dict):
                continue
            if fix.get("type") == "drop_columns":
                verifier_fix_columns.extend(fix.get("columns", []))

        cols_to_drop = list(set(
            verifier_fix_columns +
            audit_report.get("constant_columns", []) +
            # Drop ID-like columns but NOT suspicious columns on first run
            [
                col for col in df.columns
                if any(
                    keyword in col.lower()
                    for keyword in ["id", "index", "key", "uuid"]
                )
                and col != target_col
                and col not in suspicious_columns
            ]
        ))

        # Never drop the target column
        if target_col in cols_to_drop:
            cols_to_drop.remove(target_col)

        if cols_to_drop:
            print(f"  [Pipeline Builder] Dropping columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop, errors="ignore")
            self.memory.set("dataset", df)
            self.memory.set("suspicious_columns", [])

        # Separate features and target
        X = df.drop(columns=[target_col]).copy()
        y = df[target_col].copy()

        # Normalize missing target markers (Excel/CSV exports)
        if y.dtype == object:
            y = y.replace(
                ["", " ", "NA", "N/A", "na", "null", "None"],
                np.nan,
            )
            y = y.astype(str).str.strip()
            y = y.replace({"": np.nan, "nan": np.nan, "NaN": np.nan})

        # Drop rows with missing target (required for sklearn)
        missing_target = int(y.isna().sum())
        if missing_target > 0:
            valid_mask = y.notna()
            X = X.loc[valid_mask].reset_index(drop=True)
            y = y.loc[valid_mask].reset_index(drop=True)
            df = df.loc[valid_mask].reset_index(drop=True)
            self.memory.set("dataset", df)
            print(
                f"  [Pipeline Builder] Dropped {missing_target} rows "
                f"with missing target '{target_col}'"
            )

        if len(y) < 50:
            raise ValueError(
                f"Only {len(y)} rows remain after removing missing targets. "
                f"Need at least 50 rows to train."
            )

        # Coerce numeric strings (common with Excel uploads)
        for col in X.columns:
            if X[col].dtype == object:
                converted = pd.to_numeric(X[col], errors="coerce")
                if converted.notna().mean() > 0.8:
                    X[col] = converted

        # Replace inf values from feature engineering
        num_cols = X.select_dtypes(include=[np.number]).columns
        if len(num_cols):
            X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)

        # Drop very high-cardinality text columns (names, tickets, IDs)
        max_cardinality = 30
        drop_high_card = [
            col for col in X.columns
            if X[col].dtype in ["object", "category", "bool"]
            and X[col].nunique(dropna=False) > max_cardinality
        ]
        if drop_high_card:
            print(
                f"  [Pipeline Builder] Dropping high-cardinality "
                f"columns: {drop_high_card}"
            )
            X = X.drop(columns=drop_high_card)

        # Infer feature types from the current feature matrix
        numerical_cols = X.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        skewed_cols = [
            col for col in eda_report.get("skewed_columns", [])
            if col in numerical_cols
        ]

        print(f"  Numerical features: {len(numerical_cols)}")
        print(f"  Categorical features: {len(categorical_cols)}")
        print(f"  Skewed columns (RobustScaler): {skewed_cols}")

        # Numerical pipelines
        normal_num_cols = [c for c in numerical_cols if c not in skewed_cols]
        skewed_num_cols = [c for c in numerical_cols if c in skewed_cols]

        transformers = []

        if normal_num_cols:
            normal_num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("normal_num", normal_num_pipeline, normal_num_cols))

        if skewed_num_cols:
            skewed_num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ])
            transformers.append(("skewed_num", skewed_num_pipeline, skewed_num_cols))

        if categorical_cols:
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipeline, categorical_cols))

        if not transformers:
            raise ValueError("No valid features found to build pipeline.")

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # Encode target if classification
        task_type = self.memory.get("task_type")
        if task_type == "classification":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))
            self.memory.set("label_encoder", le)
        else:
            y_numeric = pd.to_numeric(y, errors="coerce")
            if y_numeric.isna().any():
                bad = int(y_numeric.isna().sum())
                raise ValueError(
                    f"Target column '{target_col}' has {bad} non-numeric "
                    f"or missing values after cleaning."
                )
            y_encoded = y_numeric.values

        # Store everything in memory
        self.memory.set("preprocessing_pipeline", preprocessor)
        self.memory.set("X", X)
        self.memory.set("y", y_encoded)
        self.memory.set("feature_schema", list(X.columns))
        self.memory.set("numerical_cols", numerical_cols)
        self.memory.set("categorical_cols", categorical_cols)

        print(f"[Pipeline Builder Agent] Pipeline built successfully.")

        return preprocessor