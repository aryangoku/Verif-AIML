# core/baseline.py

import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
import warnings
warnings.filterwarnings("ignore")


class BaselinePipeline:
    """
    A simple non-agentic sklearn pipeline.
    Used as baseline comparison against VerifAI-ML.
    No agents, no self-healing, no verification.
    Just a standard AutoML-like pipeline.
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_schema = None
        self.task_type = None
        self.metrics = {}
        self.execution_time = 0

    def run(self, df, target_column):
        """Run the baseline pipeline on the dataset."""
        print("\n[Baseline] Running simple non-agentic pipeline...")
        start_time = time.time()

        try:
            # Basic preprocessing — no agent reasoning
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Drop obvious ID columns
            id_cols = [
                col for col in X.columns
                if any(k in col.lower() for k in ["id", "index", "uuid"])
            ]
            if id_cols:
                X = X.drop(columns=id_cols, errors="ignore")

            # Detect task type
            unique_vals = y.nunique()
            if unique_vals <= 20 and y.dtype in ["object", "bool", "int64"]:
                self.task_type = "classification"
            else:
                self.task_type = "regression"

            # Encode target
            if self.task_type == "classification":
                self.label_encoder = LabelEncoder()
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = y.values

            # Feature types
            numerical_cols = X.select_dtypes(
                include=[np.number]
            ).columns.tolist()
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            self.feature_schema = list(X.columns)

            # Build simple pipeline
            transformers = []
            if numerical_cols:
                num_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ])
                transformers.append(("num", num_pipeline, numerical_cols))

            if categorical_cols:
                cat_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ])
                transformers.append(("cat", cat_pipeline, categorical_cols))

            if not transformers:
                raise ValueError("No valid features found.")

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop"
            )

            # Simple model — just Random Forest, no comparison
            if self.task_type == "classification":
                 model = LogisticRegression(
                  max_iter=1000,
                   random_state=42
                 ) 
            else:
               model = LinearRegression()

            full_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            # Train test split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_encoded if self.task_type == "classification" else None
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded,
                    test_size=0.2,
                    random_state=42
                )

            # Train
            full_pipeline.fit(X_train, y_train)
            y_pred = full_pipeline.predict(X_test)

            # Metrics
            if self.task_type == "classification":
                self.metrics["accuracy"] = round(
                    accuracy_score(y_test, y_pred), 4
                )
                self.metrics["f1_weighted"] = round(
                    f1_score(y_test, y_pred, average="weighted"), 4
                )
                try:
                    y_prob = full_pipeline.predict_proba(X_test)
                    if y_prob.shape[1] == 2:
                        self.metrics["roc_auc"] = round(
                            roc_auc_score(y_test, y_prob[:, 1]), 4
                        )
                    else:
                        self.metrics["roc_auc"] = round(
                            roc_auc_score(
                                y_test, y_prob,
                                multi_class="ovr",
                                average="weighted"
                            ), 4
                        )
                except Exception:
                    self.metrics["roc_auc"] = "N/A"
            else:
                self.metrics["rmse"] = round(
                    np.sqrt(mean_squared_error(y_test, y_pred)), 4
                )
                self.metrics["mae"] = round(
                    mean_absolute_error(y_test, y_pred), 4
                )

            self.model = full_pipeline
            self.execution_time = round(time.time() - start_time, 2)
            self.metrics["execution_time"] = self.execution_time

            print(f"[Baseline] Done in {self.execution_time}s")
            print(f"[Baseline] Metrics: {self.metrics}")

            return {
                "success": True,
                "metrics": self.metrics,
                "task_type": self.task_type,
                "execution_time": self.execution_time,
                "model_name": "RandomForest (No Agents)",
                "description": (
                    "Simple non-agentic pipeline: "
                    "basic imputation + scaling + Random Forest. "
                    "No data auditing, no verification, no self-healing."
                )
            }

        except Exception as e:
            self.execution_time = round(time.time() - start_time, 2)
            print(f"[Baseline] Failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": self.execution_time
            }