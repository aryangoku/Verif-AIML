# agents/data_auditor.py

import pandas as pd
import numpy as np


class DataAuditorAgent:
    """
    Perceives the raw dataset, checks for quality issues,
    and writes an audit report to shared memory.
    """

    def __init__(self, memory_store):
        self.memory = memory_store

    def run(self):
        """
        AGENT ROLE: Perception Agent
        ReAct Cycle: PERCEIVE
        - Perceives: Raw dataset from shared memory
        - Reasons: Identifies quality issues, missing values, suspicious columns
        - Acts: Writes audit report to shared memory
        - Observes: Flags dataset as usable or not
        """
        print("\n[Data Auditor Agent]  PERCEIVE -- Starting audit...")

        df = self.memory.get("dataset")
        target_col = self.memory.get("target_column")

        if df is None:
            raise ValueError("No dataset found in memory.")

        report = {}

        # Basic shape
        report["rows"] = df.shape[0]
        report["columns"] = df.shape[1]

        # Missing values
        missing = df.isnull().sum()
        report["missing_values"] = missing[missing > 0].to_dict()

        # Duplicate rows
        report["duplicate_rows"] = int(df.duplicated().sum())

        # Column data types
        report["column_types"] = df.dtypes.astype(str).to_dict()

        # Constant columns (zero variance)
        constant_cols = [
            col for col in df.columns
            if df[col].nunique() <= 1
        ]
        report["constant_columns"] = constant_cols

        # Suspicious columns (possible leakage)
        suspicious = []
        if target_col and target_col in df.columns:
            for col in df.columns:
                if col == target_col:
                    continue
                # Flag ID-like columns
                if any(keyword in col.lower() for keyword in ["index", "key", "uuid"]):
                    suspicious.append(col)
                # Flag columns with very high correlation to target
                try:
                    if df[col].dtype in [np.float64, np.int64]:
                        corr = abs(df[col].corr(df[target_col]))
                        if corr > 0.95:
                            suspicious.append(col)
                except:
                    pass

        report["suspicious_columns"] = suspicious
        self.memory.set("suspicious_columns", suspicious)

        # Missing value percentage
        report["missing_percentage"] = (
            df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        )

        # Flag if data is usable
        if report["missing_percentage"] > 60:
            report["usable"] = False
            report["reason"] = "Too many missing values (>60%)"
        elif df.shape[0] < 50:
            report["usable"] = False
            report["reason"] = "Too few rows (<50)"
        else:
            report["usable"] = True
            report["reason"] = "Data looks usable"

        # Store report
        self.memory.set("audit_report", report)

        print(f"[Data Auditor Agent] Audit complete.")
        print(f"  Rows: {report['rows']}, Columns: {report['columns']}")
        print(f"  Missing: {round(report['missing_percentage'], 2)}%")
        print(f"  Duplicates: {report['duplicate_rows']}")
        print(f"  Suspicious columns: {suspicious}")
        print(f"  Usable: {report['usable']}")

        return report