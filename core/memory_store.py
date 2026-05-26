# core/memory_store.py

from core.utils import safe_dict


class MemoryStore:
    """
    Shared memory store used by all agents to read and write state.
    """

    def __init__(self):
        self.state = {
            "dataset": None,
            "target_column": None,
            "task_type": None,
            "audit_report": None,
            "eda_report": None,
            "preprocessing_pipeline": None,
            "trained_model": None,
            "model_name": None,
            "evaluation_metrics": None,
            "verifier_decision": None,
            "fix_history": [],
            "retry_count": 0,
            "suspicious_columns": [],
            "recommended_metric": None,
            "feature_schema": None,
            "model_card": None,
            "pipeline_valid": False,
        }

    def get(self, key):
        return self.state.get(key)

    def set(self, key, value):
        self.state[key] = value

    def append_fix(self, fix):
        self.state["fix_history"].append(fix)

    def increment_retry(self):
        self.state["retry_count"] += 1

    def reset_verifier(self):
        self.state["verifier_decision"] = None
        self.state["pipeline_valid"] = False

    def summary(self):
        return {
            "retry_count": self.state["retry_count"],
            "fix_history": self.state["fix_history"],
            "pipeline_valid": self.state["pipeline_valid"],
            "model_name": self.state["model_name"],
            "evaluation_metrics": self.state["evaluation_metrics"],
            "verifier_decision": safe_dict(self.state["verifier_decision"]),
        }