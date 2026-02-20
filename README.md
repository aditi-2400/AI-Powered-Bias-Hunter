# Bias Audit Agent — Fairness Diagnosis System

## Overview
This project implements an automated fairness auditing system that detects, analyzes, and explains potential bias in machine learning decision models.

Disparity alone is not evidence of discrimination. Unexplained disparity is.

---

## Core Idea
Not every statistical difference is bias.

Bias exists when a protected attribute influences predictions in a way that cannot be explained by legitimate predictive features.

---

## System Pipeline
Data → Model → Fairlearn Metrics → Diagnostics → Explanation → Recommendation

---

## Fairness Metrics Used (via Fairlearn)

- Demographic Parity — outcome differences
- Equal Opportunity — TPR differences
- Equalized Odds — TPR + FPR differences
- Predictive Parity — precision differences
- Calibration — score reliability

Metrics are signals, not verdicts.

---

## Diagnosis Logic
If disparity exists:
1. Check if it disappears after controlling for valid features
2. Test counterfactual changes
3. Detect proxy variables
4. Analyze error distribution

Outcome classification:
- No bias
- Explainable disparity
- Potential bias
- Likely bias

---

## Dataset
Categorical German Credit Dataset (UCI)

Chosen because interpretability is required for fairness diagnosis.

---

## Example Reasoning Output
Incorrect:
Older applicants approved more → bias

Correct:
Older applicants have higher approval rates, but the difference disappears after controlling for employment and income. Therefore disparity is likely explained.

---

## Design Principles
- Interpretable
- Reproducible
- Evidence‑based
- Modular
- Auditable

---

## Installation
pip install fairlearn pandas scikit-learn numpy matplotlib pyyaml

---

## Run
python src/train.py --config config/audit_config.yaml

---
s
