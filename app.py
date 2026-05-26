# app.py

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from core.langgraph_orchestrator import LangGraphOrchestrator
from core.baseline import BaselinePipeline
from core.utils import safe_dict

# ── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="VerifAI-ML",
    page_icon="🤖",
    layout="wide"
)

# ── Session State ────────────────────────────────────────────────
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = None
if "result" not in st.session_state:
    st.session_state.result = None
if "pipeline_run" not in st.session_state:
    st.session_state.pipeline_run = False
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "baseline_result" not in st.session_state:
    st.session_state.baseline_result = None
if "df_cache" not in st.session_state:
    st.session_state.df_cache = None

# ── Header ───────────────────────────────────────────────────────
st.title("VerifAI-ML")
st.subheader(
    "A Verifiable Multi-Agent ML Engineer with Self-Healing Pipelines"
)
st.markdown(
    "*LangGraph ReAct Pattern: Perceive → Reason → Plan → Act → "
    "Observe → Feedback*"
)
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("How It Works")
    st.markdown("""
    1. **Upload** your CSV or Excel file
    2. **Select** the target column
    3. **Run** the pipeline
    4. **View** results and model card

    ### Agent Pipeline (LangGraph + ReAct)
    - **Data Auditor** — PERCEIVE
    - **EDA Agent** — REASON
    - **Feature Engineer** — PLAN
    - **Pipeline Builder** — PLAN → ACT
    - **Trainer** — ACT
    - **Verifier** — OBSERVE → FEEDBACK
    """)

    st.markdown("---")
    st.markdown("### Self-Healing Loop")
    st.markdown("""
    If the Verifier finds issues like data leakage,
    overfitting, or poor performance, LangGraph
    automatically routes back to Pipeline Builder
    and retrains.
    """)

    st.markdown("---")
    st.markdown("### Security Guardrails")
    st.markdown("""
    - Input validation
    - Leakage detection
    - Anomaly detection
    - Overfitting guardrail
    - Policy enforcement
    """)

    st.markdown("---")
    st.markdown("### Shared Memory Store")
    st.markdown("""
    All agents communicate through a
    shared memory store — no direct
    agent-to-agent calls. This implements
    the blackboard architecture pattern.
    """)

    st.markdown("---")
    st.markdown("### LLM Reasoning Layer")
    st.markdown("""
    Llama 3.3 70B via Groq API:
    - Selects models dynamically
    - Decides preprocessing strategy
    - Plans feature engineering
    - Sets hyperparameters
    - Explains all decisions
    """)

# ── Main Content ─────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Your Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        try:
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.pipeline_run = False
                st.session_state.result = None
                st.session_state.orchestrator = None
                st.session_state.baseline_result = None
                st.session_state.df_cache = None
                st.session_state.last_uploaded_file = uploaded_file.name

            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.session_state.df_cache = df

            st.success(f"File uploaded: {uploaded_file.name}")
            st.write(
                f"Shape: **{df.shape[0]} rows x "
                f"{df.shape[1]} columns**"
            )
            st.dataframe(df.head(10), use_container_width=True)

            st.markdown("---")
            st.subheader("Select Target Column")
            target_column = st.selectbox(
                "Which column do you want to predict?",
                options=df.columns.tolist()
            )

            st.info(f"Selected target: **{target_column}**")

            st.markdown("**Target Distribution:**")
            if df[target_column].nunique() <= 20:
                st.bar_chart(df[target_column].value_counts())
            else:
                st.line_chart(
                    df[target_column].value_counts().head(50)
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")
            uploaded_file = None

with col2:
    st.header("Run Pipeline")

    if uploaded_file is not None:
        st.markdown("### Pipeline Configuration")
        st.write(f"Dataset: `{uploaded_file.name}`")
        st.write(f"Target: `{target_column}`")
        st.write(f"Shape: `{df.shape}`")
        st.markdown("---")

        run_button = st.button(
            "Run VerifAI-ML Pipeline",
            type="primary",
            use_container_width=True
        )

        if run_button:
            st.session_state.pipeline_run = False
            st.session_state.result = None
            st.session_state.orchestrator = None
            st.session_state.baseline_result = None

            st.markdown("---")
            st.markdown("### Pipeline Running...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("Running baseline pipeline...")
                progress_bar.progress(10)
                baseline = BaselinePipeline()
                baseline_result = baseline.run(
                    df.copy(), target_column
                )
                st.session_state.baseline_result = baseline_result
                progress_bar.progress(30)

                # Use LangGraph Orchestrator
                orchestrator = LangGraphOrchestrator()
                orchestrator.load_data(df.copy(), target_column)

                agent_steps = [
                    "Data Auditor Agent — PERCEIVE",
                    "EDA Agent — REASON",
                    "Feature Engineer Agent — PLAN",
                    "Pipeline Builder Agent — PLAN → ACT",
                    "Trainer Agent — ACT",
                    "Verifier Agent — OBSERVE"
                ]

                for i, step in enumerate(agent_steps):
                    progress_bar.progress(
                        int(40 + (i / len(agent_steps)) * 50)
                    )
                    status_text.text(f"Running: {step}...")
                    time.sleep(0.1)

                progress_bar.progress(90)
                status_text.text("Finalizing pipeline...")

                with st.spinner(
                    "Running LangGraph multi-agent pipeline..."
                ):
                    result = orchestrator.run()

                progress_bar.progress(100)

                if result["success"]:
                    status_text.text(
                        "Pipeline completed successfully!"
                    )
                    st.session_state.orchestrator = orchestrator
                    st.session_state.result = result
                    st.session_state.pipeline_run = True
                    st.success(
                        "Pipeline completed! "
                        "Scroll down to see results."
                    )
                else:
                    status_text.text("Pipeline failed.")
                    st.error(
                        f"Pipeline failed: "
                        f"{result.get('error', 'Unknown error')}"
                    )
                    failed_log = result.get("execution_log") or []
                    if failed_log:
                        st.markdown("**Agent log (last failures):**")
                        for log in failed_log:
                            if "FAIL" in str(log.get("status", "")):
                                st.write(
                                    f"- **{log.get('agent')}**: "
                                    f"{log.get('findings')}"
                                )

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Please upload a dataset first.")

# ── Results Section ──────────────────────────────────────────────
if st.session_state.pipeline_run and st.session_state.result:
    result = st.session_state.result
    orchestrator = st.session_state.orchestrator
    baseline_result = st.session_state.baseline_result

    st.markdown("---")
    st.header("Pipeline Results")

    verifier_status = safe_dict(
        result.get("verifier_decision")
    ).get("status", "UNKNOWN")

    if verifier_status == "PASSED":
        st.success(
            "Pipeline Status: PASSED — "
            "All verification checks passed!"
        )
    elif verifier_status == "PASSED_WITH_WARNINGS":
        st.warning(
            "Pipeline Status: PASSED WITH WARNINGS — "
            "Some issues were found but max retries reached."
        )
    else:
        st.error("Pipeline Status: FAILED")

    st.subheader("Model Performance")
    metrics = result.get("metrics", {})
    task_type = result.get("task_type", "")
    agentic_metrics = result.get("agentic_metrics", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Best Model", result.get("model_name", "N/A"))
    with col2:
        st.metric(
            "Self-Healing Retries",
            result.get("retry_count", 0)
        )
    with col3:
        st.metric("Task Type", task_type.capitalize())
    with col4:
        st.metric(
            "Execution Time",
            f"{agentic_metrics.get('total_execution_time_seconds', 0)}s"
        )

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    if task_type == "classification":
        with col1:
            st.metric(
                "Accuracy",
                f"{metrics.get('accuracy', 'N/A')}"
            )
        with col2:
            st.metric(
                "F1 Score (Weighted)",
                f"{metrics.get('f1_weighted', 'N/A')}"
            )
        with col3:
            st.metric(
                "ROC-AUC",
                f"{metrics.get('roc_auc', 'N/A')}"
            )
    else:
        with col1:
            st.metric("RMSE", f"{metrics.get('rmse', 'N/A')}")
        with col2:
            st.metric("MAE", f"{metrics.get('mae', 'N/A')}")
        with col3:
            st.metric(
                "Train RMSE",
                f"{metrics.get('train_rmse', 'N/A')}"
            )

    # ── TABS ─────────────────────────────────────────────────────
    st.markdown("---")
    (tab1, tab2, tab3, tab4, tab5, tab6,
     tab7, tab8, tab9, tab10, tab11, tab12) = st.tabs([
        "Audit Report",
        "EDA Report",
        "Verifier Report",
        "Self-Healing Log",
        "Agent Execution Log",
        "Agentic Metrics",
        "Security Guardrails",
        "Visualizations",
        "Model Card",
        "Memory Store",
        "LLM Explanations",
        "Feature Engineering"
    ])

    # ── TAB 1: Audit Report ──────────────────────────────────────
    with tab1:
        st.subheader("Data Audit Report — PERCEIVE Stage")
        audit = safe_dict(result.get("audit_report"))
        if audit:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", audit.get("rows", "N/A"))
            with col2:
                st.metric("Columns", audit.get("columns", "N/A"))
            with col3:
                st.metric(
                    "Missing %",
                    f"{round(audit.get('missing_percentage', 0), 2)}%"
                )

            if audit.get("missing_values"):
                st.markdown("**Missing Values per Column:**")
                missing_df = pd.DataFrame(
                    list(audit["missing_values"].items()),
                    columns=["Column", "Missing Count"]
                )
                st.dataframe(missing_df, use_container_width=True)

            if audit.get("suspicious_columns"):
                st.warning(
                    f"Suspicious columns detected: "
                    f"{audit['suspicious_columns']}"
                )
            else:
                st.success("No suspicious columns detected")

            if audit.get("duplicate_rows", 0) > 0:
                st.warning(
                    f"Duplicate rows found: {audit['duplicate_rows']}"
                )
            else:
                st.success("No duplicate rows found")

    # ── TAB 2: EDA Report ────────────────────────────────────────
    with tab2:
        st.subheader("EDA Report — REASON Stage")
        eda = safe_dict(result.get("eda_report"))
        if eda:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Numerical Features",
                    len(eda.get("numerical_columns", []))
                )
                st.metric(
                    "Recommended Metric",
                    eda.get("recommended_metric", "N/A").upper()
                )
            with col2:
                st.metric(
                    "Categorical Features",
                    len(eda.get("categorical_columns", []))
                )
                st.metric(
                    "Split Strategy",
                    eda.get("split_strategy", "N/A")
                )

            if eda.get("class_imbalance"):
                st.warning(
                    f"Class imbalance detected! "
                    f"Ratio: {eda.get('imbalance_ratio', 'N/A')}"
                )

            if eda.get("skewed_columns"):
                st.info(
                    f"Skewed columns (RobustScaler applied): "
                    f"{eda['skewed_columns']}"
                )

            if eda.get("class_distribution"):
                st.markdown("**Class Distribution:**")
                class_df = pd.DataFrame(
                    list(eda["class_distribution"].items()),
                    columns=["Class", "Count"]
                )
                st.bar_chart(
                    class_df.set_index("Class"),
                    use_container_width=True
                )

    # ── TAB 3: Verifier Report ───────────────────────────────────
    with tab3:
        st.subheader("Verifier Report — OBSERVE Stage")
        verifier = safe_dict(result.get("verifier_decision"))
        if verifier:
            status = verifier.get("status", "UNKNOWN")
            if status == "PASSED":
                st.success("All verification checks passed!")
            elif status == "PASSED_WITH_WARNINGS":
                st.warning("Passed with warnings")
            else:
                st.error("Verification failed")

            issues = verifier.get("issues", [])
            if issues:
                st.markdown("**Issues Found:**")
                for issue in issues:
                    st.warning(f"{issue}")
            else:
                st.success("No issues found")

    # ── TAB 4: Self-Healing Log ──────────────────────────────────
    with tab4:
        st.subheader("Self-Healing Log — FEEDBACK Stage")
        fix_history = result.get("fix_history", [])
        retry_count = result.get("retry_count", 0)

        if retry_count == 0:
            st.success(
                "Pipeline passed on first attempt — "
                "no self-healing needed!"
            )
        else:
            st.info(
                f"Self-healing triggered {retry_count} time(s)"
            )
            for i, fix in enumerate(fix_history):
                with st.expander(
                    f"Retry {fix.get('retry', i+1)}"
                ):
                    st.markdown(
                        f"**Issues:** {fix.get('issues', [])}"
                    )
                    st.markdown(
                        f"**Fixes Applied:** "
                        f"{fix.get('fixes_applied', [])}"
                    )

    # ── TAB 5: Agent Execution Log ───────────────────────────────
    with tab5:
        st.subheader(
            "Agent Execution Log — LangGraph ReAct Timeline"
        )
        execution_log = result.get("execution_log", [])

        if execution_log:
            st.markdown(
                "Each agent is a node in the LangGraph. "
                "The graph routes between nodes based on "
                "verification results: "
                "**Perceive → Reason → Plan → Act → "
                "Observe → Feedback**"
            )
            st.markdown("---")

            for log in execution_log:
                with st.expander(
                    f"{log['react_step']}  |  {log['agent']}  "
                    f"|  {log['status']}  |  "
                    f"{log['duration_seconds']}s"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"**Agent:** {log['agent']}"
                        )
                        st.markdown(
                            f"**Role:** {log['role']}"
                        )
                        st.markdown(
                            f"**ReAct Step:** {log['react_step']}"
                        )
                    with col2:
                        st.markdown(
                            f"**Status:** {log['status']}"
                        )
                        st.markdown(
                            f"**Duration:** "
                            f"{log['duration_seconds']}s"
                        )
                    st.markdown(
                        f"**Findings:** {log['findings']}"
                    )

            st.markdown("---")
            st.markdown("**Agent Execution Time (seconds):**")
            log_df = pd.DataFrame(execution_log)
            if "duration_seconds" in log_df.columns:
                chart_df = log_df[
                    ["agent", "duration_seconds"]
                ].set_index("agent")
                st.bar_chart(
                    chart_df, use_container_width=True
                )

    # ── TAB 6: Agentic Metrics ───────────────────────────────────
    with tab6:
        st.subheader("Agentic Evaluation Metrics")
        st.markdown(
            "Based on **Week 9: Evaluating Agentic AI** — "
            "metrics for perception, planning, tools, "
            "memory, and action."
        )
        am = result.get("agentic_metrics", {})

        if am:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Task Success Rate",
                    f"{am.get('task_success_rate', 0)}%"
                )
                st.metric(
                    "Total Agents Executed",
                    am.get("total_agents_executed", 0)
                )
            with col2:
                st.metric(
                    "Step Success Rate",
                    f"{am.get('step_success_rate', 0)}%"
                )
                st.metric(
                    "Agents Passed",
                    am.get("agents_passed", 0)
                )
            with col3:
                st.metric(
                    "Self-Healing Recovery Rate",
                    f"{am.get('self_healing_recovery_rate', 0)}%"
                )
                st.metric(
                    "Total Execution Time",
                    f"{am.get('total_execution_time_seconds', 0)}s"
                )

            st.markdown("---")
            st.markdown("**Metrics Explained:**")
            st.markdown("""
            - **Task Success Rate** — Did the full pipeline complete successfully?
            - **Step Success Rate** — What percentage of individual agent steps passed?
            - **Self-Healing Recovery Rate** — How well did the system recover?
            - **Total Execution Time** — End-to-end pipeline execution time
            """)

    # ── TAB 7: Security Guardrails ───────────────────────────────
    with tab7:
        st.subheader("Security Guardrails Report")
        st.markdown(
            "Based on **Week 11: Securing LLM and AI Agents** — "
            "defense mechanisms implemented in VerifAI-ML."
        )
        security = result.get("security_report", {})

        if security:
            for check_name, check_data in security.items():
                status = check_data.get("status", "")
                label = check_name.replace("_", " ").title()

                if "PASSED" in status or "ACTIVE" in status:
                    st.success(f"**{label}** — {status}")
                else:
                    st.warning(f"**{label}** — {status}")

                with st.expander(f"Details: {label}"):
                    st.markdown(
                        f"**Description:** "
                        f"{check_data.get('description', 'N/A')}"
                    )
                    st.markdown(
                        f"**Details:** "
                        f"{check_data.get('details', 'N/A')}"
                    )

            st.markdown("---")
            st.markdown("**Defense Mechanisms Applied:**")
            st.markdown("""
            | Defense Mechanism | Implementation in VerifAI-ML |
            |---|---|
            | Input Validation | Data Auditor Agent checks data quality |
            | Input Sanitization | Suspicious column detection and removal |
            | Anomaly Detection | Missing values, duplicates, constant columns |
            | Policy Enforcement | Verifier generates and enforces fix plans |
            | Monitoring and Detection | Self-healing loop with execution logging |
            """)

    # ── TAB 8: Visualizations ────────────────────────────────────
    with tab8:
        st.subheader("Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("**Feature Importance:**")
            try:
                model_pipeline = orchestrator.memory.get(
                    "trained_model"
                )
                if model_pipeline:
                    final_model = model_pipeline.named_steps["model"]
                    if hasattr(final_model, "feature_importances_"):
                        preprocessor = model_pipeline.named_steps[
                            "preprocessor"
                        ]
                        try:
                            feature_names = (
                                preprocessor.get_feature_names_out()
                            )
                            feature_names = [
                                name.split("__")[-1]
                                if "__" in name else name
                                for name in feature_names
                            ]
                        except Exception:
                            feature_names = [
                                f"feature_{i}"
                                for i in range(
                                    len(
                                        final_model.feature_importances_
                                    )
                                )
                            ]

                        importances = final_model.feature_importances_
                        fi_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values(
                            "Importance", ascending=False
                        ).head(15)

                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.barplot(
                            data=fi_df,
                            x="Importance",
                            y="Feature",
                            ax=ax,
                            palette="viridis"
                        )
                        ax.set_title("Top 15 Feature Importances")
                        ax.set_xlabel("Importance Score")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info(
                            "Feature importance not available "
                            "for this model type."
                        )
            except Exception as e:
                st.info(f"Feature importance: {str(e)}")

        with viz_col2:
            st.markdown("**Confusion Matrix:**")
            try:
                if task_type == "classification":
                    model_pipeline = orchestrator.memory.get(
                        "trained_model"
                    )
                    X_test = orchestrator.memory.get("X_test")
                    y_test = orchestrator.memory.get("y_test")
                    label_encoder = orchestrator.memory.get(
                        "label_encoder"
                    )

                    if (model_pipeline is not None
                            and X_test is not None
                            and y_test is not None):
                        y_pred = model_pipeline.predict(X_test)

                        if label_encoder is not None:
                            labels = label_encoder.classes_
                        else:
                            labels = sorted(set(y_test))

                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(
                            cm,
                            annot=True,
                            fmt="d",
                            cmap="Blues",
                            xticklabels=labels,
                            yticklabels=labels,
                            ax=ax
                        )
                        ax.set_title("Confusion Matrix")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.info(
                        "Confusion matrix is for "
                        "classification tasks only."
                    )
            except Exception as e:
                st.info(f"Confusion matrix: {str(e)}")

        st.markdown("---")
        st.markdown("**VerifAI-ML vs Baseline Comparison:**")

        if baseline_result and baseline_result.get("success"):
            baseline_metrics = baseline_result.get("metrics", {})
            verifai_metrics = result.get("metrics", {})
            verifai_am = result.get("agentic_metrics", {})

            if task_type == "classification":
                comparison_data = {
                    "Metric": [
                        "Accuracy",
                        "F1 Score (Weighted)",
                        "ROC-AUC",
                        "Execution Time (s)",
                        "Leakage Detection",
                        "Self-Healing",
                        "Verification",
                        "LLM Reasoning",
                        "Feature Engineering",
                        "LangGraph Orchestration"
                    ],
                    "Baseline (Logistic Regression)": [
                        baseline_metrics.get("accuracy", "N/A"),
                        baseline_metrics.get("f1_weighted", "N/A"),
                        baseline_metrics.get("roc_auc", "N/A"),
                        baseline_result.get("execution_time", "N/A"),
                        "No", "No", "No", "No", "No", "No"
                    ],
                    "VerifAI-ML (6 Agents + LLM + LangGraph)": [
                        verifai_metrics.get("accuracy", "N/A"),
                        verifai_metrics.get("f1_weighted", "N/A"),
                        verifai_metrics.get("roc_auc", "N/A"),
                        verifai_am.get(
                            "total_execution_time_seconds", "N/A"
                        ),
                        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"
                    ]
                }
            else:
                comparison_data = {
                    "Metric": [
                        "RMSE",
                        "MAE",
                        "Execution Time (s)",
                        "Leakage Detection",
                        "Self-Healing",
                        "Verification",
                        "LLM Reasoning",
                        "Feature Engineering",
                        "LangGraph Orchestration"
                    ],
                    "Baseline (Linear Regression)": [
                        baseline_metrics.get("rmse", "N/A"),
                        baseline_metrics.get("mae", "N/A"),
                        baseline_result.get("execution_time", "N/A"),
                        "No", "No", "No", "No", "No", "No"
                    ],
                    "VerifAI-ML (6 Agents + LLM + LangGraph)": [
                        verifai_metrics.get("rmse", "N/A"),
                        verifai_metrics.get("mae", "N/A"),
                        verifai_am.get(
                            "total_execution_time_seconds", "N/A"
                        ),
                        "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"
                    ]
                }

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(
                comparison_df.set_index("Metric"),
                use_container_width=True
            )
            st.info(
                "VerifAI-ML uses LangGraph orchestration with "
                "LLM-selected models, intelligent preprocessing, "
                "and automatic feature engineering. "
                "Baseline uses simple Logistic Regression with "
                "no agent verification, auditing, or self-healing."
            )
        else:
            st.warning("Baseline comparison not available.")

    # ── TAB 9: Model Card ────────────────────────────────────────
    with tab9:
        st.subheader("Model Card")
        model_card = safe_dict(result.get("model_card"))
        if model_card:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Model Name:** "
                    f"{model_card.get('model_name', 'N/A')}"
                )
                st.markdown(
                    f"**Task Type:** "
                    f"{model_card.get('task_type', 'N/A')}"
                )
                st.markdown(
                    f"**Target Column:** "
                    f"{model_card.get('target_column', 'N/A')}"
                )
                st.markdown(
                    f"**Dataset Shape:** "
                    f"{model_card.get('dataset_shape', 'N/A')}"
                )
                st.markdown(
                    f"**Numerical Features:** "
                    f"{model_card.get('numerical_features', 'N/A')}"
                )
                st.markdown(
                    f"**Categorical Features:** "
                    f"{model_card.get('categorical_features', 'N/A')}"
                )
                st.markdown(
                    f"**Engineered Features:** "
                    f"{model_card.get('engineered_features', 0)}"
                )
            with col2:
                st.markdown(
                    f"**Verification Status:** "
                    f"{model_card.get('verification_status', 'N/A')}"
                )
                st.markdown(
                    f"**Self-Healing Applied:** "
                    f"{model_card.get('self_healing_applied', False)}"
                )
                st.markdown(
                    f"**Number of Retries:** "
                    f"{model_card.get('number_of_retries', 0)}"
                )
                st.markdown(
                    f"**Recommended Metric:** "
                    f"{model_card.get('recommended_metric', 'N/A').upper()}"
                )

            st.markdown("---")
            st.markdown(
                f"**Intended Use:** "
                f"{model_card.get('intended_use', 'N/A')}"
            )
            st.warning(
                f"**Limitations:** "
                f"{model_card.get('limitations', 'N/A')}"
            )

            model_card_json = json.dumps(
                model_card, indent=2, default=str
            )
            st.download_button(
                label="Download Model Card (JSON)",
                data=model_card_json,
                file_name="model_card.json",
                mime="application/json"
            )

    # ── TAB 10: Memory Store ─────────────────────────────────────
    with tab10:
        st.subheader("Shared Memory Store")
        st.markdown(
            "All agents communicate through a shared memory store. "
            "This implements the **blackboard architecture** pattern "
            "from multi-agent systems theory."
        )

        if orchestrator:
            memory = orchestrator.memory
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Dataset Info:**")
                dataset = memory.get("dataset")
                if dataset is not None:
                    st.write(f"Shape: {dataset.shape}")
                    st.write(
                        f"Target: {memory.get('target_column')}"
                    )
                    st.write(
                        f"Task Type: {memory.get('task_type')}"
                    )

                st.markdown("**Agent Outputs Written to Memory:**")
                st.write(
                    f"Audit Report: "
                    f"{'Yes' if memory.get('audit_report') else 'No'}"
                )
                st.write(
                    f"EDA Report: "
                    f"{'Yes' if memory.get('eda_report') else 'No'}"
                )
                st.write(
                    f"Feature Report: "
                    f"{'Yes' if memory.get('feature_report') else 'No'}"
                )
                st.write(
                    f"Preprocessing Pipeline: "
                    f"{'Yes' if memory.get('preprocessing_pipeline') else 'No'}"
                )
                st.write(
                    f"Trained Model: "
                    f"{'Yes' if memory.get('trained_model') else 'No'}"
                )
                st.write(
                    f"Verifier Decision: "
                    f"{'Yes' if memory.get('verifier_decision') else 'No'}"
                )
                st.write(
                    f"Model Card: "
                    f"{'Yes' if memory.get('model_card') else 'No'}"
                )
                st.write(
                    f"LLM Selected Models: "
                    f"{memory.get('llm_selected_models') or 'N/A'}"
                )

            with col2:
                st.markdown("**Memory State Summary:**")
                summary = memory.summary()
                st.write(
                    f"Model Name: "
                    f"{summary.get('model_name', 'N/A')}"
                )
                st.write(
                    f"Retry Count: "
                    f"{summary.get('retry_count', 0)}"
                )
                st.write(
                    f"Pipeline Valid: "
                    f"{summary.get('pipeline_valid', False)}"
                )
                st.write(
                    f"Verifier Decision: "
                    f"{safe_dict(summary.get('verifier_decision')).get('status', 'N/A')}"
                )

                st.markdown("**Feature Schema:**")
                feature_schema = memory.get("feature_schema")
                if feature_schema:
                    schema_df = pd.DataFrame(
                        {"Feature": feature_schema}
                    )
                    st.dataframe(
                        schema_df,
                        use_container_width=True
                    )

            st.markdown("---")
            st.markdown(
                "**Why shared memory?** Each agent reads from and "
                "writes to this central store. No agent calls another "
                "agent directly. This makes the system modular, "
                "auditable, and easy to debug — a core principle of "
                "safe agentic system design."
            )

    # ── TAB 11: LLM Explanations ─────────────────────────────────
    with tab11:
        st.subheader("LLM Explanations — AI Reasoning Layer")
        st.markdown(
            "Each agent decision is explained in plain English "
            "by **Llama 3.3 70B** via Groq API. This implements "
            "the hybrid agentic architecture — deterministic ML "
            "decisions with LLM-powered natural language reasoning, "
            "dynamic model selection, intelligent preprocessing, "
            "and feature engineering."
        )

        llm_explanations = result.get("llm_explanations", {})

        if not llm_explanations:
            st.warning(
                "LLM explanations not available. "
                "Check your GROQ_API_KEY in the .env file."
            )
        else:
            if llm_explanations.get("pipeline_summary"):
                st.markdown("---")
                st.markdown("### Executive Summary")
                st.info(llm_explanations["pipeline_summary"])

            if llm_explanations.get("model_selection"):
                st.markdown("---")
                st.markdown("### Dynamic Model Selection")
                st.info(llm_explanations["model_selection"])

            if llm_explanations.get("preprocessing"):
                st.markdown("---")
                st.markdown("### Preprocessing Strategy")
                st.info(llm_explanations["preprocessing"])

            if llm_explanations.get("feature_engineering"):
                st.markdown("---")
                st.markdown("### Feature Engineering")
                st.info(llm_explanations["feature_engineering"])

            if llm_explanations.get("audit"):
                st.markdown("---")
                st.markdown(
                    "### Data Auditor Agent — PERCEIVE Stage"
                )
                st.info(llm_explanations["audit"])

            if llm_explanations.get("eda"):
                st.markdown("---")
                st.markdown("### EDA Agent — REASON Stage")
                st.info(llm_explanations["eda"])

            if llm_explanations.get("verifier"):
                st.markdown("---")
                st.markdown(
                    "### Verifier Agent — OBSERVE Stage"
                )
                st.info(llm_explanations["verifier"])

            if llm_explanations.get("self_healing"):
                st.markdown("---")
                st.markdown(
                    "### Self-Healing Loop — FEEDBACK Stage"
                )
                st.warning(llm_explanations["self_healing"])

            if llm_explanations.get("model_card"):
                st.markdown("---")
                st.markdown("### Model Card — LLM Generated")
                st.info(llm_explanations["model_card"])

            st.markdown("---")
            st.markdown(
                "**Architecture Note:** VerifAI-ML uses a hybrid "
                "approach powered by LangGraph. All ML pipeline "
                "decisions are deterministic and reproducible. "
                "The LLM layer adds natural language reasoning, "
                "dynamic model selection, intelligent preprocessing "
                "decisions, feature engineering, and hyperparameter "
                "tuning — making the system both reliable and "
                "interpretable."
            )

    # ── TAB 12: Feature Engineering ──────────────────────────────
    with tab12:
        st.subheader("Feature Engineering — LLM Planned")
        st.markdown(
            "The LLM analyses dataset characteristics and "
            "suggests new features to create. The Feature "
            "Engineer Agent then builds these features automatically."
        )

        feature_report = safe_dict(result.get("feature_report"))

        if feature_report:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Original Columns",
                    feature_report.get("original_columns", "N/A")
                )
            with col2:
                st.metric(
                    "New Features Created",
                    feature_report.get("total_features_added", 0)
                )

            if feature_report.get("new_features"):
                st.markdown("**Features Created:**")
                for feat in feature_report["new_features"]:
                    st.success(f"New feature added: {feat}")
            else:
                st.info(
                    "No new features were created "
                    "for this dataset."
                )

            if feature_report.get("llm_suggested"):
                st.markdown("**LLM Suggested Features:**")
                for feat in feature_report["llm_suggested"]:
                    if not isinstance(feat, dict):
                        continue
                    st.markdown(
                        f"- **{feat.get('name')}**: "
                        f"{feat.get('col1')} "
                        f"{feat.get('operation')} "
                        f"{feat.get('col2') or ''}"
                    )
        else:
            st.info(
                "Feature engineering report not available."
            )

    # ── Prediction Section ───────────────────────────────────────
    st.markdown("---")
    st.header("Make Predictions")
    st.markdown(
        "Upload a new CSV file with the same columns "
        "(without the target column) to get predictions."
    )

    pred_file = st.file_uploader(
        "Upload file for predictions",
        type=["csv", "xlsx"],
        key="prediction_file"
    )

    if pred_file is not None:
        try:
            if pred_file.name.endswith(".csv"):
                pred_df = pd.read_csv(pred_file)
            else:
                pred_df = pd.read_excel(pred_file)

            st.write(f"Prediction file shape: {pred_df.shape}")

            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Generating predictions..."):
                    predictions = orchestrator.predict(pred_df)
                    pred_df["Prediction"] = predictions
                    st.success("Predictions generated!")
                    st.dataframe(
                        pred_df, use_container_width=True
                    )

                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center>VerifAI-ML — CIS 600 Applied Agentic AI Systems — "
    "Syracuse University</center>",
    unsafe_allow_html=True
)