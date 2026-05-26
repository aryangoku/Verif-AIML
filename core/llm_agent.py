# core/llm_agent.py

import os
import ast
from groq import Groq
from dotenv import load_dotenv
from core.utils import safe_dict

load_dotenv()


class LLMAgent:
    """
    LLM-powered reasoning layer using Groq API.
    Used by all agents to generate natural language
    explanations, fix plans, and model documentation.
    Model: llama-3.3-70b-versatile (free tier)
    Capabilities:
    - Dynamic model selection
    - Preprocessing strategy decisions
    - Feature engineering planning
    - Hyperparameter tuning
    - Natural language explanations
    - Model card generation
    - Executive summary generation
    """

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def _call(self, system_prompt, user_prompt, max_tokens=300):
        """Make a single LLM API call."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM explanation unavailable: {str(e)}"

    def explain_audit(self, audit_report):
        """Generate plain English explanation of audit findings."""
        system = (
            "You are an expert ML engineer. "
            "Explain data audit findings clearly and concisely "
            "in 3-4 sentences. "
            "Focus on what was found and why it matters."
        )
        user = f"""
        Data Audit Report:
        - Rows: {audit_report.get('rows')}
        - Columns: {audit_report.get('columns')}
        - Missing values: {audit_report.get('missing_values')}
        - Missing percentage: {round(audit_report.get('missing_percentage', 0), 2)}%
        - Duplicate rows: {audit_report.get('duplicate_rows')}
        - Suspicious columns: {audit_report.get('suspicious_columns')}
        - Constant columns: {audit_report.get('constant_columns')}

        Explain what the Data Auditor agent found and what actions
        should be taken.
        """
        return self._call(system, user)

    def explain_eda(self, eda_report):
        """Generate plain English explanation of EDA findings."""
        system = (
            "You are an expert data scientist. "
            "Explain EDA findings clearly in 3-4 sentences. "
            "Focus on what the data patterns suggest about "
            "the best ML approach."
        )
        user = f"""
        EDA Report:
        - Task type: {eda_report.get('task_type')}
        - Numerical columns: {eda_report.get('numerical_columns')}
        - Categorical columns: {eda_report.get('categorical_columns')}
        - Class imbalance: {eda_report.get('class_imbalance')}
        - Imbalance ratio: {eda_report.get('imbalance_ratio', 'N/A')}
        - Skewed columns: {eda_report.get('skewed_columns')}
        - Recommended metric: {eda_report.get('recommended_metric')}
        - Class distribution: {eda_report.get('class_distribution')}

        Explain what the EDA agent discovered and why these findings
        matter for model selection and preprocessing.
        """
        return self._call(system, user)

    def select_models(self, eda_report, audit_report):
        """Use LLM to dynamically select which models to train."""
        system = (
            "You are an expert ML engineer. "
            "Based on dataset characteristics, recommend exactly which "
            "ML models to train. "
            "Return ONLY a Python list of model names from these options: "
            "RandomForest, XGBoost, LightGBM, CatBoost, "
            "GradientBoosting, ExtraTrees, AdaBoost, "
            "LogisticRegression, LinearRegression, "
            "Ridge, Lasso, SVM, KNN, LDA, GaussianNB. "
            "Return only the list, nothing else. "
            "Example: ['RandomForest', 'XGBoost', 'LightGBM']"
        )
        user = f"""
        Dataset characteristics:
        - Task type: {eda_report.get('task_type')}
        - Total rows: {audit_report.get('rows')}
        - Numerical features: {len(eda_report.get('numerical_columns', []))}
        - Categorical features: {len(eda_report.get('categorical_columns', []))}
        - Class imbalance: {eda_report.get('class_imbalance')}
        - Imbalance ratio: {eda_report.get('imbalance_ratio', 1.0)}
        - Skewed columns: {len(eda_report.get('skewed_columns', []))}
        - Missing percentage: {round(audit_report.get('missing_percentage', 0), 2)}%

        Which 3-4 models should be trained? Return only the Python list.
        """
        response = self._call(system, user, max_tokens=80)

        try:
            start = response.find("[")
            end = response.find("]") + 1
            if start != -1 and end != 0:
                model_list = ast.literal_eval(response[start:end])
                valid_models = [
                    "RandomForest", "XGBoost", "LightGBM",
                    "CatBoost", "GradientBoosting", "ExtraTrees",
                    "AdaBoost", "LogisticRegression",
                    "LinearRegression", "Ridge", "Lasso",
                    "SVM", "KNN", "LDA", "GaussianNB"
                ]
                filtered = [
                    m for m in model_list
                    if m in valid_models
                ]
                if filtered:
                    return filtered
        except Exception:
            pass

        task_type = eda_report.get("task_type", "classification")
        if task_type == "classification":
            return [
                "RandomForest", "XGBoost",
                "LightGBM", "CatBoost"
            ]
        else:
            return [
                "RandomForest", "XGBoost",
                "LightGBM", "Ridge"
            ]

    def decide_preprocessing(self, eda_report, audit_report):
        """Use LLM to decide preprocessing strategy."""
        system = (
            "You are an expert ML engineer. "
            "Based on dataset characteristics, decide the best "
            "preprocessing strategy. "
            "Return ONLY a Python dictionary with these exact keys: "
            "use_pca, handle_imbalance, drop_high_missing, "
            "missing_threshold, create_interactions. "
            "Example: {'use_pca': False, 'handle_imbalance': True, "
            "'drop_high_missing': True, 'missing_threshold': 0.5, "
            "'create_interactions': False} "
            "Return only the dictionary, nothing else."
        )
        user = f"""
        Dataset characteristics:
        - Task type: {eda_report.get('task_type')}
        - Total rows: {audit_report.get('rows')}
        - Total columns: {audit_report.get('columns')}
        - Numerical features: {len(eda_report.get('numerical_columns', []))}
        - Categorical features: {len(eda_report.get('categorical_columns', []))}
        - Class imbalance: {eda_report.get('class_imbalance')}
        - Imbalance ratio: {eda_report.get('imbalance_ratio', 1.0)}
        - Missing percentage: {round(audit_report.get('missing_percentage', 0), 2)}%
        - Skewed columns: {len(eda_report.get('skewed_columns', []))}

        Decide the preprocessing strategy.
        Return only the dictionary.
        """
        response = self._call(system, user, max_tokens=100)

        try:
            start = response.find("{")
            end = response.find("}") + 1
            if start != -1 and end != 0:
                strategy = ast.literal_eval(response[start:end])
                return strategy
        except Exception:
            pass

        return {
            "use_pca": False,
            "handle_imbalance": eda_report.get(
                "class_imbalance", False
            ),
            "drop_high_missing": True,
            "missing_threshold": 0.5,
            "create_interactions": False
        }

    def decide_hyperparameters(self, model_name, eda_report, audit_report):
        """Use LLM to decide hyperparameters for selected model."""
        system = (
            "You are an expert ML engineer. "
            "Based on dataset characteristics and model name, "
            "recommend hyperparameters. "
            "Return ONLY a Python dictionary of hyperparameters. "
            "Keep it simple — maximum 4 parameters. "
            "Example for RandomForest classification: "
            "{'n_estimators': 200, 'max_depth': 10, "
            "'min_samples_leaf': 2, 'class_weight': 'balanced'} "
            "Return only the dictionary, nothing else."
        )
        user = f"""
        Model: {model_name}
        Dataset characteristics:
        - Task type: {eda_report.get('task_type')}
        - Total rows: {audit_report.get('rows')}
        - Numerical features: {len(eda_report.get('numerical_columns', []))}
        - Categorical features: {len(eda_report.get('categorical_columns', []))}
        - Class imbalance: {eda_report.get('class_imbalance')}
        - Imbalance ratio: {eda_report.get('imbalance_ratio', 1.0)}
        - Missing percentage: {round(audit_report.get('missing_percentage', 0), 2)}%

        Recommend hyperparameters for {model_name}.
        Return only the dictionary.
        """
        response = self._call(system, user, max_tokens=100)

        try:
            start = response.find("{")
            end = response.find("}") + 1
            if start != -1 and end != 0:
                params = ast.literal_eval(response[start:end])
                return params
        except Exception:
            pass

        return {}

    def plan_features(self, eda_report, audit_report):
        """Use LLM to plan which features to engineer."""
        system = (
            "You are an expert ML feature engineer. "
            "Based on dataset characteristics, suggest new features. "
            "Return ONLY a Python dictionary with key 'features' "
            "containing a list of feature dictionaries. "
            "Each feature dict must have: "
            "name, type, col1, col2, operation. "
            "Types: interaction or aggregate. "
            "Operations for interaction: "
            "multiply, divide, add, subtract. "
            "Operations for aggregate: square, sqrt, log. "
            "Example: {'features': [{'name': 'Age_Fare', "
            "'type': 'interaction', 'col1': 'Age', "
            "'col2': 'Fare', 'operation': 'multiply'}, "
            "{'name': 'Fare_Log', 'type': 'aggregate', "
            "'col1': 'Fare', 'col2': None, 'operation': 'log'}]} "
            "Return only the dictionary. "
            "Suggest maximum 3 features."
        )
        user = f"""
        Dataset characteristics:
        - Task type: {eda_report.get('task_type')}
        - Numerical columns: {eda_report.get('numerical_columns')}
        - Categorical columns: {eda_report.get('categorical_columns')}
        - Skewed columns: {eda_report.get('skewed_columns')}
        - Total rows: {audit_report.get('rows')}
        - Missing percentage: {round(audit_report.get('missing_percentage', 0), 2)}%

        Suggest new features to engineer.
        Return only the dictionary.
        """
        response = self._call(system, user, max_tokens=200)

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != 0:
                plan = ast.literal_eval(response[start:end])
                if "features" in plan:
                    return plan
        except Exception:
            pass

        return {"features": []}

    def explain_verifier(self, issues, fix_plan, metrics, task_type):
        """Generate plain English explanation of verification results."""
        system = (
            "You are an expert ML engineer specializing in "
            "model validation. "
            "Explain verification results clearly in 3-4 sentences. "
            "If issues were found, explain what they mean and "
            "how they will be fixed."
        )
        user = f"""
        Verification Results:
        - Issues found: {issues}
        - Fix plan: {fix_plan}
        - Model metrics: {metrics}
        - Task type: {task_type}

        Explain what the Verifier agent found, whether the pipeline
        passed or failed, and what actions will be taken.
        """
        return self._call(system, user)

    def explain_self_healing(self, retry_count, fix_history):
        """Generate plain English explanation of self-healing."""
        system = (
            "You are an expert ML engineer. "
            "Explain the self-healing process clearly in 3-4 sentences. "
            "Focus on what problems were found and how they were fixed."
        )
        user = f"""
        Self-Healing Log:
        - Total retries: {retry_count}
        - Fix history: {fix_history}

        Explain what problems the self-healing loop detected and fixed,
        and whether the pipeline ultimately passed verification.
        """
        return self._call(system, user)

    def generate_model_card(self, model_card_data):
        """Generate a professional model card description."""
        system = (
            "You are an expert ML engineer writing professional "
            "model documentation. "
            "Write a concise model card description in 4-5 sentences. "
            "Cover: what the model does, how it was built, "
            "its performance, and important limitations."
        )
        user = f"""
        Model Card Data:
        - Model name: {model_card_data.get('model_name')}
        - Task type: {model_card_data.get('task_type')}
        - Target column: {model_card_data.get('target_column')}
        - Dataset shape: {model_card_data.get('dataset_shape')}
        - Metrics: {model_card_data.get('evaluation_metrics')}
        - Verification status: {model_card_data.get('verification_status')}
        - Self-healing applied: {model_card_data.get('self_healing_applied')}
        - Number of retries: {model_card_data.get('number_of_retries')}
        - Numerical features: {model_card_data.get('numerical_features')}
        - Categorical features: {model_card_data.get('categorical_features')}
        - Engineered features: {model_card_data.get('engineered_features')}

        Write a professional model card description.
        """
        return self._call(system, user, max_tokens=400)

    def generate_pipeline_summary(self, result):
        """Generate final plain English summary of pipeline."""
        system = (
            "You are an expert ML engineer. "
            "Write a concise executive summary of the ML pipeline run "
            "in 4-5 sentences. Cover: what was built, key findings, "
            "model performance, and any issues that were fixed."
        )
        user = f"""
        Pipeline Run Summary:
        - Model trained: {result.get('model_name')}
        - Task type: {result.get('task_type')}
        - Metrics: {result.get('metrics')}
        - Verifier status: {safe_dict(result.get('verifier_decision')).get('status')}
        - Self-healing retries: {result.get('retry_count')}
        - Fix history: {result.get('fix_history')}
        - Agentic metrics: {result.get('agentic_metrics')}
        - Missing data: {safe_dict(result.get('audit_report')).get('missing_percentage')}%
        - Suspicious columns: {safe_dict(result.get('audit_report')).get('suspicious_columns')}
        - LLM selected models: {safe_dict(result.get('llm_explanations')).get('model_selection')}
        - Features engineered: {safe_dict(result.get('feature_report')).get('total_features_added', 0)}

        Write an executive summary of this ML pipeline run.
        """
        return self._call(system, user, max_tokens=400)