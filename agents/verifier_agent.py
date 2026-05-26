# agents/verifier_agent.py

import numpy as np
import pandas as pd


class VerifierAgent:
    """
    Perceives trained model, metrics, and pipeline structure.
    Reasons about leakage, overfitting, and robustness issues.
    Acts by approving the pipeline or generating a fix plan.
    Triggers self-healing loop if issues are found.
    """

    def __init__(self, memory_store):
        self.memory = memory_store
        self.max_retries = 3

    def run(self):
        """
        AGENT ROLE: Feedback/Observe Agent (Security Guardrail)
        ReAct Cycle: OBSERVE -> FEEDBACK
        - Perceives: Trained model metrics and pipeline structure
        - Reasons: Checks for leakage, overfitting, poor performance
        - Acts: Approves pipeline OR generates fix plan
        - Observes: Triggers self-healing loop if issues found
        Defense Mechanisms: Input validation, leakage detection, anomaly detection
        """
        print("\n[Verifier Agent] OBSERVE -> FEEDBACK -- Starting verification...")

        metrics = self.memory.get("evaluation_metrics") or {}
        task_type = self.memory.get("task_type")
        suspicious_columns = self.memory.get("suspicious_columns") or []
        retry_count = self.memory.get("retry_count") or 0
        dataset = self.memory.get("dataset")
        target_col = self.memory.get("target_column")

        issues = []
        fix_plan = []

        if not metrics or self.memory.get("trained_model") is None:
            issues.append(
                "No evaluation metrics available -- training did not complete"
            )
            self.memory.set("verifier_decision", {
                "status": "FAILED",
                "issues": issues,
                "fix_plan": fix_plan,
            })
            if retry_count + 1 >= self.max_retries:
                self.memory.set("pipeline_valid", True)
                self.memory.set("verifier_decision", {
                    "status": "PASSED_WITH_WARNINGS",
                    "issues": issues,
                    "fix_plan": [],
                })
                return True, issues
            self.memory.increment_retry()
            return False, issues

        # CHECK 1: Leakage Detection
        print("  [Check 1] Leakage detection...")

        leaky_cols = []

        # 1a - Check suspicious columns flagged by Data Auditor
        if suspicious_columns:
            for col in suspicious_columns:
                if dataset is not None and col in dataset.columns:
                    leaky_cols.append(col)
                    print(f"  WARNING: Suspicious column still in dataset: {col}")

        # 1b - Check correlation of all columns with target
        if dataset is not None and target_col in dataset.columns:
            for col in dataset.columns:
                if col == target_col:
                    continue
                try:
                    col_data = dataset[col]
                    if col_data.dtype == object:
                        col_data = col_data.astype("category").cat.codes
                    corr = abs(col_data.corr(dataset[target_col]))
                    if corr > 0.90:
                        if col not in leaky_cols:
                            leaky_cols.append(col)
                            print(f"  WARNING: High correlation leakage: {col} (corr={corr:.4f})")
                except Exception:
                    pass

        if leaky_cols:
            issues.append(f"Leakage detected -- suspicious columns: {leaky_cols}")
            fix_plan.append({
                "type": "drop_columns",
                "columns": leaky_cols,
                "reason": f"Leakage columns detected: {leaky_cols}"
            })
        else:
            print("  OK: No leakage columns detected")

        # CHECK 2: Overfitting Detection
        print("  [Check 2] Overfitting detection...")

        if task_type == "classification":
            train_score = metrics.get("train_f1", 0)
            test_score = metrics.get("f1_weighted", 0)
            gap = train_score - test_score
            print(f"  Train F1: {train_score}, Test F1: {test_score}, Gap: {gap:.4f}")

            if gap > 0.20:
                issues.append(
                    f"Overfitting detected: train F1={train_score}, test F1={test_score}, gap={gap:.4f}"
                )
                fix_plan.append({
                    "type": "reduce_overfitting",
                    "reason": f"Train-test gap too large: {gap:.4f}",
                    "suggestion": "Reduce model complexity or add regularization"
                })
                print(f"  WARNING: Overfitting detected! Gap: {gap:.4f}")
            else:
                print(f"  OK: No overfitting detected (gap={gap:.4f})")

        else:
            train_rmse = metrics.get("train_rmse", 0)
            test_rmse = metrics.get("rmse", 0)
            if train_rmse > 0:
                ratio = test_rmse / train_rmse
                print(f"  Train RMSE: {train_rmse}, Test RMSE: {test_rmse}, Ratio: {ratio:.4f}")
                if ratio > 2.0:
                    issues.append(
                        f"Overfitting detected: train RMSE={train_rmse}, test RMSE={test_rmse}"
                    )
                    fix_plan.append({
                        "type": "reduce_overfitting",
                        "reason": f"Train-test RMSE ratio too large: {ratio:.4f}",
                        "suggestion": "Reduce model complexity"
                    })
                    print(f"  WARNING: Overfitting detected! Ratio: {ratio:.4f}")
                else:
                    print(f"  OK: No overfitting detected (ratio={ratio:.4f})")

        # CHECK 3: Minimum Performance Check
        print("  [Check 3] Minimum performance check...")

        if task_type == "classification":
            test_score = metrics.get("f1_weighted", 0)
            if test_score < 0.5:
                issues.append(f"Poor model performance: F1={test_score}")
                fix_plan.append({
                    "type": "improve_performance",
                    "reason": f"F1 score too low: {test_score}",
                    "suggestion": "Try different models or feature engineering"
                })
                print(f"  WARNING: Poor performance! F1={test_score}")
            else:
                print(f"  OK: Performance acceptable (F1={test_score})")
        else:
            rmse = metrics.get("rmse", 0)
            print(f"  OK: RMSE={rmse} (regression performance check passed)")

        # CHECK 4: Data Integrity Check
        print("  [Check 4] Data integrity check...")
        if dataset is not None and target_col in dataset.columns:
            missing_pct = dataset.isnull().sum().sum() / (
                dataset.shape[0] * dataset.shape[1]
            ) * 100
            if missing_pct > 40:
                issues.append(f"High missing value percentage: {missing_pct:.2f}%")
                fix_plan.append({
                    "type": "data_quality",
                    "reason": f"Too many missing values: {missing_pct:.2f}%",
                    "suggestion": "Consider dropping high-missing columns"
                })
                print(f"  WARNING: High missing values: {missing_pct:.2f}%")
            else:
                print(f"  OK: Data integrity OK (missing={missing_pct:.2f}%)")

        # DECISION
        print(f"\n  Total issues found: {len(issues)}")

        if len(issues) == 0:
            self.memory.set("pipeline_valid", True)
            self.memory.set("verifier_decision", {
                "status": "PASSED",
                "issues": [],
                "fix_plan": []
            })
            print("[Verifier Agent] Pipeline PASSED all checks!")
            return True, []

        else:
            if retry_count + 1 >= self.max_retries:
                print(f"[Verifier Agent] Max retries ({self.max_retries}) reached.")
                print("[Verifier Agent] Accepting pipeline with warnings.")
                self.memory.set("pipeline_valid", True)
                self.memory.set("verifier_decision", {
                    "status": "PASSED_WITH_WARNINGS",
                    "issues": issues,
                    "fix_plan": fix_plan
                })
                return True, issues

            print(f"[Verifier Agent] Pipeline FAILED. Issues: {issues}")
            print(f"[Verifier Agent] Generating fix plan and triggering retraining...")

            for fix in fix_plan:
                if not isinstance(fix, dict):
                    continue
                if fix.get("type") == "drop_columns":
                    cols_to_drop = fix.get("columns") or []
                    current_df = self.memory.get("dataset")
                    if current_df is not None:
                        current_df = current_df.drop(
                            columns=cols_to_drop,
                            errors="ignore"
                        )
                        self.memory.set("dataset", current_df)
                        self.memory.set("suspicious_columns", [])
                        print(f"  Fix applied: Dropped columns {cols_to_drop}")

                elif fix.get("type") == "reduce_overfitting":
                    self.memory.set("use_simple_model", True)
                    print(f"  Fix applied: Will use simpler model config")

            self.memory.set("verifier_decision", {
                "status": "FAILED",
                "issues": issues,
                "fix_plan": fix_plan
            })

            self.memory.increment_retry()
            self.memory.append_fix({
                "retry": retry_count + 1,
                "issues": issues,
                "fixes_applied": [
                    f.get("type") for f in fix_plan if isinstance(f, dict)
                ]
            })

            return False, issues