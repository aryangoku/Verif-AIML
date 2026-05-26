# VerifAI-ML

### A Verifiable Multi-Agent ML Engineer with Self-Healing Pipelines

CIS 600 - Applied Agentic AI Systems | Syracuse University | Spring 2026

---

## Overview

VerifAI-ML is a production-grade multi-agent system that automatically
builds, verifies, and self-heals machine learning pipelines for tabular data.
It combines LangGraph orchestration, deterministic ML agents, LLM reasoning,
RAG-based domain knowledge, and self-healing feedback loops to deliver
trustworthy, explainable, and self-correcting ML pipelines.

Unlike traditional AutoML tools that simply build a model, VerifAI-ML
acts like a senior ML engineer - auditing your data, reasoning about patterns,
engineering new features, dynamically selecting models, verifying results,
and fixing problems automatically.

---

## Architecture

User uploads CSV
      |
      v
LangGraph Graph Starts
      |
      v
Data Auditor Agent - PERCEIVE
LLM explains audit findings
RAG retrieves data quality best practices
      |
      v
EDA Agent - REASON
LLM explains EDA + selects models dynamically
LLM decides preprocessing strategy
LLM plans feature engineering
RAG retrieves ML best practices
      |
      v
Feature Engineer Agent - PLAN
Creates LLM-suggested features
Creates domain-specific features automatically
      |
      v
Pipeline Builder Agent - PLAN to ACT
Applies LLM preprocessing decisions
      |
      v
Trainer Agent - ACT
Trains LLM-selected models with LLM hyperparameters
      |
      v
Verifier Agent - OBSERVE to FEEDBACK
      |
      v (if issues found)
LangGraph routes back to Pipeline Builder
Self-Healing Loop triggered
LLM generates fix plan
      |
      v
LLM writes model card and executive summary
Results displayed in Streamlit UI

---

## Agents

Agent                | ReAct Step          | Role
---                  | ---                 | ---
Data Auditor         | PERCEIVE            | Checks data quality, missing values, suspicious columns
EDA Agent            | REASON              | Analyses patterns, detects imbalance, recommends metrics
Feature Engineer     | PLAN                | Creates LLM-suggested and domain features
Pipeline Builder     | PLAN to ACT         | Builds preprocessing pipeline with LLM decisions
Trainer              | ACT                 | Trains LLM-selected models with LLM hyperparameters
Verifier             | OBSERVE to FEEDBACK | Checks leakage, overfitting, triggers self-healing

---

## LLM Integration

VerifAI-ML uses Llama 3.3 70B via Groq API for:

- Natural language explanations of each agent decision
- Dynamic model selection based on dataset characteristics
- Preprocessing strategy decisions
- Feature engineering planning
- Hyperparameter tuning recommendations
- Human readable fix plans when verification fails
- Professional model card generation
- Executive pipeline summary

---

## RAG Domain Knowledge

VerifAI-ML uses Retrieval Augmented Generation to enhance agent decisions:

- Data quality best practices retrieved for Data Auditor
- ML preprocessing best practices retrieved for EDA Agent
- Model selection guidelines retrieved for Trainer Agent
- Verification best practices retrieved for Verifier Agent
- Knowledge base built from ML research and engineering guidelines

---

## Available Models

The LLM dynamically selects from these models:

Classification:
RandomForest, XGBoost, LightGBM, CatBoost,
GradientBoosting, ExtraTrees, AdaBoost,
LogisticRegression, SVM, KNN, LDA, GaussianNB

Regression:
RandomForest, XGBoost, LightGBM, CatBoost,
GradientBoosting, ExtraTrees, AdaBoost,
Ridge, Lasso, LinearRegression, KNN, SVR

---

## Self-Healing Loop

LangGraph routes between nodes based on verification results:

Train Model
      |
      v
Verifier checks for:
- Data leakage
- Overfitting
- Poor performance
- Data integrity issues
      |
      v (if issues found)
LangGraph routes back to Pipeline Builder
LLM generates fix plan
      |
      v
Apply fixes automatically
      |
      v
Retrain model
      |
      v
Verify again (max 3 retries)
      |
      v
Deploy model

---

## Security Guardrails

Based on Week 11: Securing LLM and AI Agents

Defense Mechanism      | Implementation
---                    | ---
Input Validation       | Data Auditor checks data quality
Input Sanitization     | Suspicious column detection and removal
Anomaly Detection      | Missing values, duplicates, constant columns
Policy Enforcement     | Verifier generates and enforces fix plans
Monitoring Detection   | Self-healing loop with execution logging

---

## Tech Stack

- Python 3.10+
- LangGraph (graph-based agent orchestration)
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Groq API (Llama 3.3 70B)
- ChromaDB (vector database for RAG)
- Sentence Transformers (embeddings for RAG)
- Streamlit
- Pandas and NumPy
- Matplotlib and Seaborn
- MLflow (experiment tracking)

---

## Setup

Clone the repo

    git clone https://github.com/aryangoku/Verif-AIML.git
    cd Verif-AIML

Create virtual environment

    python3 -m venv venv
    source venv/bin/activate

Install dependencies

    pip install -r requirements.txt

Add your Groq API key

    Create a .env file in the root folder
    Add this line: GROQ_API_KEY=your_key_here

Run the app

    streamlit run app.py

---

## Project Structure

    verifai-ml/
    ├── agents/
    │   ├── data_auditor.py            PERCEIVE agent
    │   ├── eda_agent.py               REASON agent
    │   ├── feature_engineer_agent.py  PLAN agent
    │   ├── pipeline_builder.py        PLAN to ACT agent
    │   ├── trainer_agent.py           ACT agent
    │   └── verifier_agent.py          OBSERVE to FEEDBACK agent
    ├── core/
    │   ├── memory_store.py            Shared blackboard memory
    │   ├── langgraph_orchestrator.py  LangGraph pipeline coordinator
    │   ├── llm_agent.py               LLM reasoning layer
    │   ├── rag_agent.py               RAG domain knowledge
    │   ├── baseline.py                Baseline comparison
    │   └── multi_dataset_benchmark.py Multi-dataset benchmarking
    ├── app.py                         Streamlit UI
    ├── requirements.txt
    ├── .env                           API keys not committed
    └── README.md

---

## Usage

1. Open the app at http://localhost:8501
2. Upload a CSV or Excel file
3. Select the target column
4. Click Run VerifAI-ML Pipeline
5. View results across 13 tabs

Tabs available:
- Audit Report
- EDA Report
- Verifier Report
- Self-Healing Log
- Agent Execution Log
- Agentic Metrics
- Security Guardrails
- Visualizations
- Model Card
- Memory Store
- LLM Explanations
- Feature Engineering
- Multi-Dataset Benchmark

---

## Evaluation Results

Tested on Titanic dataset with target Survived

Metric               | Baseline  | VerifAI-ML
---                  | ---       | ---
Accuracy             | 0.8268    | 0.8212
F1 Score             | 0.8249    | 0.8146
ROC-AUC              | 0.8551    | 0.8485
Leakage Detection    | No        | Yes
Self-Healing         | No        | Yes
Verification         | No        | Yes
LLM Reasoning        | No        | Yes
Feature Engineering  | No        | Yes
RAG Knowledge        | No        | Yes
LangGraph            | No        | Yes

---

## Team

- Aryan Sadvelkar (361854585)
- Jeevan Tumkur Venkatesh (740874557)
- Prathamesh Parab (212089207)
- Vedant Pednekar (291366032)

---

## Course

CIS 600 - Applied Agentic AI Systems
Syracuse University
Spring 2026

---

## Future Work

- MLflow experiment tracking for full reproducibility
- Multi-agent debate for model selection consensus
- Continuous monitoring and drift detection
- REST API deployment with FastAPI
- Support for image and text datasets
- Reinforcement learning for agent improvement

---

## References

1. Vaswani et al., Attention Is All You Need, NeurIPS 2017
2. Hutter et al., Automated Machine Learning, Springer 2019
3. Feurer et al., Efficient and Robust AutoML, NeurIPS 2015
4. Sculley et al., Hidden Technical Debt in ML Systems, NeurIPS 2015
5. Mitchell et al., Model Cards for Model Reporting, FAT 2019
6. Amershi et al., Software Engineering for ML, IEEE Software 2019
7. Yao et al., ReAct: Synergizing Reasoning and Acting in LLMs, 2022
8. Liu et al., Formalizing Prompt Injection Attacks, USENIX 2024
9. Chase, LangChain: Building applications with LLMs, 2022
10. LangGraph: Multi-agent orchestration framework, Anthropic 2024