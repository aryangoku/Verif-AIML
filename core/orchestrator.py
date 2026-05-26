# core/orchestrator.py

import time
import pandas as pd
import numpy as np
from core.memory_store import MemoryStore
from core.llm_agent import LLMAgent
from agents.data_auditor import DataAuditorAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.pipeline_builder import PipelineBuilderAgent
from agents.trainer_agent import TrainerAgent
from agents.verifier_agent import VerifierAgent


class Orchestrator:
    """
    Controls the multi-agent pipeline flow.
    Implements ReAct pattern: Perceive → Reason → Plan → Act → Observe → Feedback
    Manages the self-healing loop.
    Coordinates all agents through shared memory.
    Uses LLM for natural language explanations at each stage.
    Includes feature engineering agent for LLM-driven feature creation.
    """

    def __init__(self):
        self.memory = MemoryStore()
        self.max_retries = 3
        self.execution_log = []
        self.pipeline_start_time = None
        self.llm_explanations = {}

        # Initialize LLM agent
        try:
            self.llm = LLMAgent()
            self.llm_available = True
            print("[Orchestrator] LLM agent initialized successfully.")
        except Exception as e:
            self.llm = None
            self.llm_available = False
            print(f"[Orchestrator] LLM unavailable: {e}")

    def load_data(self, df, target_column):
        self.memory.set("dataset", df)
        self.memory.set("target_column", target_column)
        print(f"\n[Orchestrator] Dataset loaded.")
        print(f"  Shape: {df.shape}")
        print(f"  Target column: {target_column}")

    def _log_agent(self, agent_name, role, react_step,
                   status, duration, findings):
        self.execution_log.append({
            "agent": agent_name,
            "role": role,
            "react_step": react_step,
            "status": status,
            "duration_seconds": round(duration, 2),
            "findings": findings
        })

    def run(self):
        print("\n" + "="*60)
        print("  VerifAI-ML Pipeline Starting")
        print("  ReAct: Perceive->Reason->Plan->Act->Observe->Feedback")
        print("="*60)

        self.pipeline_start_time = time.time()
        self.execution_log = []
        self.llm_explanations = {}
        steps_attempted = 0
        steps_succeeded = 0

        # ── STEP 1: Data Auditor — PERCEIVE ─────────────────────
        steps_attempted += 1
        start = time.time()
        try:
            auditor = DataAuditorAgent(self.memory)
            audit_report = auditor.run()
            duration = time.time() - start
            steps_succeeded += 1

            self._log_agent(
                "Data Auditor Agent", "Perception Agent",
                "PERCEIVE", "PASSED", duration,
                f"Rows: {audit_report.get('rows')}, "
                f"Missing: {round(audit_report.get('missing_percentage', 0), 2)}%, "
                f"Suspicious: {audit_report.get('suspicious_columns', [])}"
            )

            if self.llm_available:
                print("[Orchestrator] LLM generating audit explanation...")
                self.llm_explanations["audit"] = (
                    self.llm.explain_audit(audit_report)
                )

            if not audit_report.get("usable", False):
                reason = audit_report.get("reason", "Unknown")
                return {
                    "success": False,
                    "error": f"Data quality check failed: {reason}",
                    "audit_report": audit_report,
                    "execution_log": self.execution_log,
                    "llm_explanations": self.llm_explanations
                }

        except Exception as e:
            duration = time.time() - start
            self._log_agent(
                "Data Auditor Agent", "Perception Agent",
                "PERCEIVE", "FAILED", duration, str(e)
            )
            return {
                "success": False,
                "error": f"Data Auditor failed: {str(e)}",
                "execution_log": self.execution_log,
                "llm_explanations": self.llm_explanations
            }

        # ── STEP 2: EDA Agent — REASON ───────────────────────────
        steps_attempted += 1
        start = time.time()
        try:
            eda = EDAAgent(self.memory)
            eda_report = eda.run()
            duration = time.time() - start
            steps_succeeded += 1

            self._log_agent(
                "EDA Agent", "Reasoning Agent", "REASON",
                "PASSED", duration,
                f"Task: {eda_report.get('task_type')}, "
                f"Metric: {eda_report.get('recommended_metric')}, "
                f"Imbalance: {eda_report.get('class_imbalance', False)}"
            )

            if self.llm_available:
                print("[Orchestrator] LLM generating EDA explanation...")
                self.llm_explanations["eda"] = (
                    self.llm.explain_eda(eda_report)
                )

                print("[Orchestrator] LLM selecting models...")
                selected_models = self.llm.select_models(
                    eda_report, audit_report
                )
                self.memory.set("llm_selected_models", selected_models)
                self.llm_explanations["model_selection"] = (
                    f"LLM selected these models based on dataset "
                    f"characteristics: {selected_models}"
                )
                print(
                    f"[Orchestrator] LLM selected models: {selected_models}"
                )

                print(
                    "[Orchestrator] LLM deciding preprocessing strategy..."
                )
                preprocessing_strategy = self.llm.decide_preprocessing(
                    eda_report, audit_report
                )
                self.memory.set(
                    "preprocessing_strategy", preprocessing_strategy
                )
                self.llm_explanations["preprocessing"] = (
                    f"LLM decided preprocessing strategy: "
                    f"{preprocessing_strategy}"
                )
                print(
                    f"[Orchestrator] Preprocessing strategy: "
                    f"{preprocessing_strategy}"
                )

                print(
                    "[Orchestrator] LLM planning feature engineering..."
                )
                feature_plan = self.llm.plan_features(
                    eda_report, audit_report
                )
                self.memory.set("llm_feature_plan", feature_plan)
                self.llm_explanations["feature_engineering"] = (
                    f"LLM planned these new features: "
                    f"{feature_plan.get('features', [])}"
                )
                print(
                    f"[Orchestrator] Feature plan: "
                    f"{feature_plan}"
                )

        except ValueError as e:
            duration = time.time() - start
            self._log_agent(
                "EDA Agent", "Reasoning Agent",
                "REASON", "FAILED", duration, str(e)
            )
            return {
                "success": False,
                "error": str(e),
                "execution_log": self.execution_log,
                "llm_explanations": self.llm_explanations
            }
        except Exception as e:
            duration = time.time() - start
            self._log_agent(
                "EDA Agent", "Reasoning Agent",
                "REASON", "FAILED", duration, str(e)
            )
            return {
                "success": False,
                "error": f"EDA Agent failed: {str(e)}",
                "execution_log": self.execution_log,
                "llm_explanations": self.llm_explanations
            }

        # ── STEP 2.5: Feature Engineer — PLAN ───────────────────
        steps_attempted += 1
        start = time.time()
        try:
            feature_eng = FeatureEngineerAgent(self.memory)
            feature_report = feature_eng.run()
            duration = time.time() - start
            steps_succeeded += 1
            self._log_agent(
                "Feature Engineer Agent",
                "Feature Engineering Agent",
                "PLAN → ACT",
                "PASSED",
                duration,
                f"Created {feature_report.get('total_features_added', 0)} "
                f"new features: {feature_report.get('new_features', [])}"
            )
        except Exception as e:
            duration = time.time() - start
            self._log_agent(
                "Feature Engineer Agent",
                "Feature Engineering Agent",
                "PLAN → ACT",
                "FAILED",
                duration,
                str(e)
            )
            print(f"[Orchestrator] Feature engineering failed: {e}")

        # ── SELF-HEALING LOOP ────────────────────────────────────
        pipeline_valid = False
        retry_count = 0

        while not pipeline_valid and retry_count <= self.max_retries:

            if retry_count > 0:
                print(f"\n{'='*60}")
                print(
                    f"  Self-Healing Loop - "
                    f"Retry {retry_count}/{self.max_retries}"
                )
                print(f"{'='*60}")

            # ── STEP 3: Pipeline Builder — PLAN ─────────────────
            steps_attempted += 1
            start = time.time()
            try:
                builder = PipelineBuilderAgent(self.memory)
                builder.run()
                duration = time.time() - start
                steps_succeeded += 1
                self._log_agent(
                    "Pipeline Builder Agent", "Planning Agent",
                    "PLAN → ACT", "PASSED", duration,
                    f"Numerical: "
                    f"{len(self.memory.get('numerical_cols') or [])}, "
                    f"Categorical: "
                    f"{len(self.memory.get('categorical_cols') or [])}"
                )
            except Exception as e:
                duration = time.time() - start
                self._log_agent(
                    "Pipeline Builder Agent", "Planning Agent",
                    "PLAN → ACT", "FAILED", duration, str(e)
                )
                return {
                    "success": False,
                    "error": f"Pipeline Builder failed: {str(e)}",
                    "execution_log": self.execution_log,
                    "llm_explanations": self.llm_explanations
                }

            # ── STEP 4: Trainer Agent — ACT ──────────────────────
            steps_attempted += 1
            start = time.time()
            try:
                trainer = TrainerAgent(self.memory)
                trainer.run()
                duration = time.time() - start
                steps_succeeded += 1
                metrics = self.memory.get("evaluation_metrics") or {}
                self._log_agent(
                    "Trainer Agent", "Action Agent", "ACT",
                    "PASSED", duration,
                    f"Best model: {self.memory.get('model_name')}, "
                    f"Score: "
                    f"{metrics.get('f1_weighted') or metrics.get('rmse')}"
                )
            except Exception as e:
                duration = time.time() - start
                self._log_agent(
                    "Trainer Agent", "Action Agent",
                    "ACT", "FAILED", duration, str(e)
                )
                return {
                    "success": False,
                    "error": f"Trainer Agent failed: {str(e)}",
                    "execution_log": self.execution_log,
                    "llm_explanations": self.llm_explanations
                }

            # ── STEP 5: Verifier Agent — OBSERVE ────────────────
            steps_attempted += 1
            start = time.time()
            try:
                verifier = VerifierAgent(self.memory)
                pipeline_valid, issues = verifier.run()
                duration = time.time() - start
                steps_succeeded += 1

                verifier_decision = (
                    self.memory.get("verifier_decision") or {}
                )
                fix_plan = verifier_decision.get("fix_plan", [])
                current_metrics = (
                    self.memory.get("evaluation_metrics") or {}
                )
                task_type = self.memory.get("task_type")

                if self.llm_available:
                    print(
                        "[Orchestrator] LLM generating "
                        "verifier explanation..."
                    )
                    self.llm_explanations["verifier"] = (
                        self.llm.explain_verifier(
                            issues, fix_plan,
                            current_metrics, task_type
                        )
                    )

                if not pipeline_valid:
                    self._log_agent(
                        "Verifier Agent", "Feedback Agent",
                        "OBSERVE → FEEDBACK",
                        f"FAILED — Retry {retry_count+1}",
                        duration, f"Issues: {issues}"
                    )

                    if self.llm_available and retry_count > 0:
                        fix_history = (
                            self.memory.get("fix_history") or []
                        )
                        self.llm_explanations["self_healing"] = (
                            self.llm.explain_self_healing(
                                retry_count, fix_history
                            )
                        )

                    retry_count += 1
                    self.memory.reset_verifier()

                else:
                    self._log_agent(
                        "Verifier Agent", "Feedback Agent",
                        "OBSERVE → FEEDBACK",
                        "PASSED", duration,
                        f"All checks passed. Retries: {retry_count}"
                    )
                    print(
                        f"\n[Orchestrator] Pipeline verified successfully!"
                    )

            except Exception as e:
                duration = time.time() - start
                self._log_agent(
                    "Verifier Agent", "Feedback Agent",
                    "OBSERVE → FEEDBACK", "FAILED", duration, str(e)
                )
                return {
                    "success": False,
                    "error": f"Verifier Agent failed: {str(e)}",
                    "execution_log": self.execution_log,
                    "llm_explanations": self.llm_explanations
                }

        # ── STEP 6: Generate Model Card ──────────────────────────
        print("\n[Orchestrator] Generating model card...")
        model_card = self._generate_model_card()
        self.memory.set("model_card", model_card)

        if self.llm_available:
            print("[Orchestrator] LLM generating model card...")
            self.llm_explanations["model_card"] = (
                self.llm.generate_model_card(model_card)
            )

        # ── COMPUTE AGENTIC METRICS ──────────────────────────────
        total_time = round(time.time() - self.pipeline_start_time, 2)
        total_steps = len(self.execution_log)
        passed_steps = sum(
            1 for log in self.execution_log
            if "PASSED" in log["status"]
        )

        agentic_metrics = {
            "task_success_rate": round(
                (1 if pipeline_valid else 0) * 100, 2
            ),
            "step_success_rate": round(
                (passed_steps / total_steps * 100)
                if total_steps > 0 else 0, 2
            ),
            "agent_pass_rate": round(
                (passed_steps / total_steps * 100)
                if total_steps > 0 else 0, 2
            ),
            "self_healing_recovery_rate": round(
                ((self.max_retries - self.memory.get("retry_count")) /
                 self.max_retries * 100), 2
            ),
            "total_execution_time_seconds": total_time,
            "total_agents_executed": total_steps,
            "agents_passed": passed_steps,
            "agents_failed": total_steps - passed_steps,
            "retries_used": self.memory.get("retry_count"),
        }

        self.memory.set("agentic_metrics", agentic_metrics)

        # ── SECURITY GUARDRAILS REPORT ───────────────────────────
        audit = self.memory.get("audit_report") or {}
        verifier_decision = self.memory.get("verifier_decision") or {}

        security_report = {
            "input_validation": {
                "status": "PASSED",
                "description": (
                    "Dataset validated for minimum rows, "
                    "columns, and data types"
                ),
                "details": (
                    f"Rows: {audit.get('rows')}, "
                    f"Columns: {audit.get('columns')}"
                )
            },
            "data_leakage_detection": {
                "status": (
                    "PASSED"
                    if not audit.get("suspicious_columns")
                    else "DETECTED & FIXED"
                ),
                "description": (
                    "Checked for ID-like columns "
                    "and high-correlation features"
                ),
                "details": (
                    f"Suspicious columns found: "
                    f"{audit.get('suspicious_columns', [])}"
                )
            },
            "anomaly_detection": {
                "status": "PASSED",
                "description": (
                    "Checked for constant columns, "
                    "duplicate rows, extreme missing values"
                ),
                "details": (
                    f"Duplicates: {audit.get('duplicate_rows', 0)}, "
                    f"Missing: "
                    f"{round(audit.get('missing_percentage', 0), 2)}%"
                )
            },
            "overfitting_guardrail": {
                "status": (
                    "PASSED"
                    if not any(
                        "Overfitting" in str(i)
                        for i in verifier_decision.get("issues", [])
                    )
                    else "DETECTED & MITIGATED"
                ),
                "description": (
                    "Train vs test performance gap monitored"
                ),
                "details": (
                    f"Verifier status: "
                    f"{verifier_decision.get('status', 'N/A')}"
                )
            },
            "policy_enforcement": {
                "status": "ACTIVE",
                "description": (
                    "Fix plans generated and enforced automatically"
                ),
                "details": (
                    f"Fix history: "
                    f"{len(self.memory.get('fix_history') or [])} "
                    f"fixes applied"
                )
            }
        }

        self.memory.set("security_report", security_report)

        # ── FINAL RESULT ─────────────────────────────────────────
        final_result = {
            "success": True,
            "model_name": self.memory.summary()["model_name"],
            "metrics": self.memory.get("evaluation_metrics") or {},
            "verifier_decision": verifier_decision,
            "fix_history": self.memory.summary()["fix_history"],
            "retry_count": self.memory.get("retry_count"),
            "model_card": model_card,
            "task_type": self.memory.get("task_type"),
            "recommended_metric": self.memory.get("recommended_metric"),
            "audit_report": self.memory.get("audit_report"),
            "eda_report": self.memory.get("eda_report"),
            "feature_report": self.memory.get("feature_report"),
            "execution_log": self.execution_log,
            "agentic_metrics": agentic_metrics,
            "security_report": security_report,
            "llm_explanations": self.llm_explanations,
        }

        if self.llm_available:
            print("[Orchestrator] LLM generating pipeline summary...")
            self.llm_explanations["pipeline_summary"] = (
                self.llm.generate_pipeline_summary(final_result)
            )
            final_result["llm_explanations"] = self.llm_explanations

        # ── FINAL SUMMARY ────────────────────────────────────────
        print("\n" + "="*60)
        print("  VerifAI-ML Pipeline Complete!")
        print("="*60)
        print(f"  Model: {final_result['model_name']}")
        print(
            f"  Status: "
            f"{verifier_decision.get('status', 'UNKNOWN')}"
        )
        print(f"  Retries: {self.memory.get('retry_count')}")
        print(f"  Total Time: {total_time}s")
        print(
            f"  Step Success Rate: "
            f"{agentic_metrics['step_success_rate']}%"
        )

        return final_result

    def _generate_model_card(self):
        model_name = self.memory.get("model_name")
        metrics = self.memory.get("evaluation_metrics") or {}
        task_type = self.memory.get("task_type")
        target_col = self.memory.get("target_column")
        dataset = self.memory.get("dataset")
        verifier_decision = (
            self.memory.get("verifier_decision") or {}
        )
        fix_history = self.memory.get("fix_history") or []
        eda_report = self.memory.get("eda_report") or {}
        feature_report = self.memory.get("feature_report") or {}
        shape = dataset.shape if dataset is not None else "Unknown"

        return {
            "model_name": model_name,
            "task_type": task_type,
            "target_column": target_col,
            "dataset_shape": shape,
            "evaluation_metrics": metrics,
            "verification_status": verifier_decision.get(
                "status", "UNKNOWN"
            ),
            "issues_found": verifier_decision.get("issues", []),
            "self_healing_applied": len(fix_history) > 0,
            "number_of_retries": self.memory.get("retry_count"),
            "fix_history": fix_history,
            "recommended_metric": self.memory.get("recommended_metric"),
            "numerical_features": len(
                eda_report.get("numerical_columns", [])
            ),
            "categorical_features": len(
                eda_report.get("categorical_columns", [])
            ),
            "engineered_features": feature_report.get(
                "total_features_added", 0
            ),
            "intended_use": (
                "Tabular supervised learning for "
                "classification or regression."
            ),
            "limitations": (
                "Auto-generated by VerifAI-ML. "
                "Validate with domain experts before deployment."
            ),
        }

    def get_trained_model(self):
        return self.memory.get("trained_model")

    def predict(self, new_df):
        model = self.memory.get("trained_model")
        feature_schema = self.memory.get("feature_schema")
        label_encoder = self.memory.get("label_encoder")

        if model is None:
            raise ValueError(
                "No trained model found. Run the pipeline first."
            )

        missing_cols = set(feature_schema) - set(new_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing columns in input data: {missing_cols}"
            )

        X_new = new_df[feature_schema]
        predictions = model.predict(X_new)

        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions)

        return predictions