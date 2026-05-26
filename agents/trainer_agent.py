# agents/trainer_agent.py

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression,
    Ridge, Lasso
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import (
    KNeighborsClassifier, KNeighborsRegressor
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score,
    mean_squared_error, mean_absolute_error
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from core.utils import sanitize_metrics
import warnings
warnings.filterwarnings("ignore")


class TrainerAgent:
    """
    AGENT ROLE: Action Agent
    ReAct Cycle: ACT
    - Perceives: Preprocessed data and pipeline from shared memory
    - Reasons: Compares LLM-selected candidate models via cross-validation
    - Acts: Trains best model and evaluates on test set
    - Observes: Records metrics for Verifier Agent
    Uses LLM-selected models for dynamic model selection.
    Available models: RandomForest, XGBoost, LightGBM, CatBoost,
    GradientBoosting, ExtraTrees, AdaBoost, LogisticRegression,
    Ridge, Lasso, SVM, KNN, LDA, GaussianNB
    """

    def __init__(self, memory_store):
        self.memory = memory_store

    def _get_all_models(self, task_type):
        """Return all available models by name."""
        if task_type == "classification":
            return {
                "RandomForest": RandomForestClassifier(
                    n_estimators=100, random_state=42
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=100, random_state=42,
                    eval_metric="logloss", verbosity=0
                ),
                "LightGBM": LGBMClassifier(
                    n_estimators=100, random_state=42,
                    verbose=-1
                ),
                "CatBoost": CatBoostClassifier(
                    iterations=100, random_seed=42,
                    verbose=0
                ),
                "GradientBoosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=42
                ),
                "ExtraTrees": ExtraTreesClassifier(
                    n_estimators=100, random_state=42
                ),
                "AdaBoost": AdaBoostClassifier(
                    n_estimators=100, random_state=42
                ),
                "LogisticRegression": LogisticRegression(
                    max_iter=1000, random_state=42
                ),
                "SVM": SVC(
                    probability=True, random_state=42
                ),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "LDA": LinearDiscriminantAnalysis(),
                "GaussianNB": GaussianNB(),
            }
        else:
            return {
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, random_state=42
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=100, random_state=42,
                    verbosity=0
                ),
                "LightGBM": LGBMRegressor(
                    n_estimators=100, random_state=42,
                    verbose=-1
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=100, random_seed=42,
                    verbose=0
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=100, random_state=42
                ),
                "ExtraTrees": ExtraTreesRegressor(
                    n_estimators=100, random_state=42
                ),
                "AdaBoost": AdaBoostRegressor(
                    n_estimators=100, random_state=42
                ),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=1.0, max_iter=1000),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(n_neighbors=5),
                "SVR": SVR(),
            }

    def _cv_folds(self, y, task_type):
        """Pick a safe number of CV folds for the dataset size."""
        n_samples = len(y)
        if n_samples < 10:
            return 2
        if task_type == "classification":
            min_class = min(Counter(y).values())
            return max(2, min(5, min_class, n_samples // 2))
        return max(2, min(5, n_samples // 10))

    def _build_fallback_preprocessor(self, X):
        """Minimal preprocessor when the main pipeline is missing."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()
        transformers = []
        if numerical_cols:
            transformers.append((
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numerical_cols,
            ))
        if categorical_cols:
            transformers.append((
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            max_categories=20,
                        ),
                    ),
                ]),
                categorical_cols,
            ))
        if not transformers:
            raise ValueError("No usable features after preprocessing.")
        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def _get_simple_models(self, task_type):
        """Lightweight models for self-healing or fallback training."""
        if task_type == "classification":
            return {
                "LogisticRegression": LogisticRegression(
                    max_iter=2000, random_state=42
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=50,
                    max_depth=8,
                    random_state=42,
                ),
            }
        return {
            "Ridge": Ridge(alpha=1.0),
            "RandomForest": RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
            ),
        }

    def _get_candidate_models(self, task_type, selected_model_names=None):
        """Get candidate models — either LLM selected or defaults."""
        all_models = self._get_all_models(task_type)

        if self.memory.get("use_simple_model"):
            print("  [Trainer] Using simpler models (self-healing mode)")
            return self._get_simple_models(task_type)

        if selected_model_names:
            print(
                f"  [Trainer] LLM selected models: "
                f"{selected_model_names}"
            )
            candidate_models = {}
            for name in selected_model_names:
                if name in all_models:
                    candidate_models[name] = all_models[name]
            if candidate_models:
                return candidate_models

        # Fallback to defaults
        print("  [Trainer] Using default model selection")
        if task_type == "classification":
            return {
                "RandomForest": all_models["RandomForest"],
                "XGBoost": all_models["XGBoost"],
                "LightGBM": all_models["LightGBM"],
                "CatBoost": all_models["CatBoost"],
            }
        else:
            return {
                "RandomForest": all_models["RandomForest"],
                "XGBoost": all_models["XGBoost"],
                "LightGBM": all_models["LightGBM"],
                "Ridge": all_models["Ridge"],
            }

    def run(self):
        print("\n[Trainer Agent] ACT -- Starting model training...")

        X = self.memory.get("X")
        y = self.memory.get("y")
        preprocessor = self.memory.get("preprocessing_pipeline")
        task_type = self.memory.get("task_type")
        selected_models = self.memory.get("llm_selected_models")

        if X is None or y is None:
            raise ValueError("No data found in memory.")

        if preprocessor is None:
            print(
                "  [Trainer] WARNING: No preprocessing pipeline in memory. "
                "Building fallback preprocessor."
            )
            preprocessor = self._build_fallback_preprocessor(X)
            self.memory.set("preprocessing_pipeline", preprocessor)

        try:
            y = np.asarray(y)
        except Exception as e:
            raise ValueError(
                f"Could not process target column: {str(e)}"
            )

        # Drop any remaining rows with invalid targets
        if np.issubdtype(y.dtype, np.number):
            valid_mask = np.isfinite(y)
        else:
            import pandas as pd
            valid_mask = ~pd.isna(y)

        if not valid_mask.all():
            dropped = int((~valid_mask).sum())
            print(
                f"  [Trainer] Dropping {dropped} rows with "
                f"invalid target values"
            )
            if hasattr(X, "iloc"):
                X = X.iloc[valid_mask].reset_index(drop=True)
            else:
                X = X[valid_mask]
            y = y[valid_mask]

        if len(y) < 50:
            raise ValueError(
                f"Only {len(y)} rows available after target cleaning. "
                f"Need at least 50 rows to train."
            )

        if y.dtype == object:
            unique_vals = len(set(y))
            if unique_vals > 50:
                raise ValueError(
                    f"Target column has {unique_vals} unique text values. "
                    f"This is not a valid target. "
                    f"Please select a numerical or categorical column."
                )

        if len(y) < 50:
            raise ValueError(
                f"Dataset has only {len(y)} rows. "
                f"Need at least 50 rows to train a model."
            )

        # Train test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y if task_type == "classification" else None
            )
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        self.memory.set("X_train", X_train)
        self.memory.set("X_test", X_test)
        self.memory.set("y_train", y_train)
        self.memory.set("y_test", y_test)

        candidate_models = self._get_candidate_models(
            task_type, selected_models
        )

        best_model = None
        best_score = -np.inf
        best_model_name = None
        all_results = {}

        if task_type == "classification":
            scoring = "f1_weighted"
        else:
            scoring = "neg_root_mean_squared_error"

        cv_folds = self._cv_folds(y_train, task_type)

        print(f"  Task type: {task_type}")
        print(f"  Scoring metric: {scoring}")
        print(f"  CV folds: {cv_folds}")
        print(
            f"  Training {len(candidate_models)} "
            f"candidate models..."
        )

        last_cv_error = None

        for model_name, model in candidate_models.items():
            try:
                full_pipeline = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("model", model)
                ])

                try:
                    cv_scores = cross_val_score(
                        full_pipeline,
                        X_train, y_train,
                        cv=cv_folds,
                        scoring=scoring,
                    )
                except Exception as e:
                    last_cv_error = str(e)
                    print(f"  {model_name} CV failed: {e}")
                    continue

                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)

                all_results[model_name] = {
                    "cv_mean": round(float(mean_score), 4),
                    "cv_std": round(float(std_score), 4)
                }

                print(
                    f"  {model_name}: "
                    f"CV Score = {mean_score:.4f} "
                    f"+/- {std_score:.4f}"
                )

                if mean_score > best_score:
                    best_score = mean_score
                    best_model = full_pipeline
                    best_model_name = model_name

            except Exception as e:
                print(f"  {model_name} failed: {e}")
                continue

        if best_model is None:
            print(
                "  [Trainer] All primary models failed. "
                "Retrying with fallback preprocessor and simple models..."
            )
            preprocessor = self._build_fallback_preprocessor(X_train)
            self.memory.set("preprocessing_pipeline", preprocessor)
            fallback_models = self._get_simple_models(task_type)

            for model_name, model in fallback_models.items():
                try:
                    full_pipeline = Pipeline(steps=[
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ])
                    cv_scores = cross_val_score(
                        full_pipeline,
                        X_train, y_train,
                        cv=cv_folds,
                        scoring=scoring,
                    )
                    mean_score = np.mean(cv_scores)
                    if mean_score > best_score:
                        best_score = mean_score
                        best_model = full_pipeline
                        best_model_name = model_name
                    print(
                        f"  [Fallback] {model_name}: "
                        f"CV Score = {mean_score:.4f}"
                    )
                except Exception as e:
                    last_cv_error = str(e)
                    print(f"  [Fallback] {model_name} failed: {e}")

        if best_model is None:
            hint = (
                f" Last error: {last_cv_error}"
                if last_cv_error
                else ""
            )
            raise ValueError(
                "All models failed during training. "
                "Use a numeric or categorical target column with "
                "at least 50 rows and fewer than 30 categories per "
                f"feature.{hint}"
            )

        print(f"\n  Best model: {best_model_name}")
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        metrics = {}

        if task_type == "classification":
            metrics["accuracy"] = round(
                accuracy_score(y_test, y_pred), 4
            )
            metrics["f1_weighted"] = round(
                f1_score(y_test, y_pred, average="weighted"), 4
            )
            try:
                y_prob = best_model.predict_proba(X_test)
                if y_prob.shape[1] == 2:
                    metrics["roc_auc"] = round(
                        roc_auc_score(y_test, y_prob[:, 1]), 4
                    )
                else:
                    metrics["roc_auc"] = round(
                        roc_auc_score(
                            y_test, y_prob,
                            multi_class="ovr",
                            average="weighted"
                        ), 4
                    )
            except Exception:
                metrics["roc_auc"] = "N/A"

            y_train_pred = best_model.predict(X_train)
            metrics["train_accuracy"] = round(
                accuracy_score(y_train, y_train_pred), 4
            )
            metrics["train_f1"] = round(
                f1_score(
                    y_train, y_train_pred, average="weighted"
                ), 4
            )

        else:
            metrics["rmse"] = round(
                np.sqrt(mean_squared_error(y_test, y_pred)), 4
            )
            metrics["mae"] = round(
                mean_absolute_error(y_test, y_pred), 4
            )
            y_train_pred = best_model.predict(X_train)
            metrics["train_rmse"] = round(
                np.sqrt(
                    mean_squared_error(y_train, y_train_pred)
                ), 4
            )

        metrics["all_model_results"] = all_results
        metrics = sanitize_metrics(metrics)

        print(f"  Test metrics: {metrics}")

        self.memory.set("trained_model", best_model)
        self.memory.set("model_name", best_model_name)
        self.memory.set("evaluation_metrics", metrics)

        print(
            f"[Trainer Agent] Training complete. "
            f"Best model: {best_model_name}"
        )

        return best_model, metrics