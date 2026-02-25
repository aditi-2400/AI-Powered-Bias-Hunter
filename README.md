# Bias Audit Agent — Fairness Diagnosis System

## Overview
## Overview

This project implements an automated fairness auditing system that
detects, analyzes, and explains potential bias in machine learning
decision models.

Disparity alone is not evidence of discrimination. **Unexplained
disparity is.**

------------------------------------------------------------------------

## Core Idea

Not every statistical difference is bias.

Bias exists when a protected attribute influences predictions in a way
that cannot be explained by legitimate predictive features.

------------------------------------------------------------------------

## System Pipeline

Data → Model → Fairlearn Metrics → Agent Reasoning → Explanation →
Recommendations

The system separates responsibilities across layers:

  Layer                Responsibility
  -------------------- -----------------------------------------------
  Deterministic code   Computes metrics and structured evidence
  LLM Agent            Interprets results and generates explanations
  UI                   Displays findings and recommendations

This separation ensures reliability, interpretability, and
reproducibility.

------------------------------------------------------------------------

## Fairness Metrics Used (via Fairlearn)

-   **Demographic Parity** --- outcome differences\
-   **Equal Opportunity** --- TPR differences\
-   **Equalized Odds** --- TPR + FPR differences\
-   **Predictive Parity** --- precision differences\
-   **Calibration** --- score reliability

Metrics are **signals, not verdicts**.

------------------------------------------------------------------------

## Diagnosis Logic

When disparities are detected, the agent analyzes the evidence and
recommends further investigation steps.

Suggested diagnostics may include:

1.  Feature distribution analysis\
2.  Counterfactual testing\
3.  Proxy detection\
4.  Error distribution analysis

These diagnostics are **not executed automatically yet**.\
They are proposed by the agent as targeted next steps to determine
whether a disparity is explainable or potentially problematic.

### Outcome Classification

Based on available evidence, the agent classifies findings as:

-   No bias\
-   Explainable disparity\
-   Potential bias\
-   Likely bias

These classifications are interpretations based on statistical signals
and should not be treated as legal or causal conclusions.

------------------------------------------------------------------------

## Dataset

**German Credit Dataset (UCI)**

Chosen because interpretability is required for fairness diagnosis.

Sensitive attributes used:

-   sex\
-   age_group

------------------------------------------------------------------------

## LLM Agent Layer

The system includes a local reasoning agent that:

-   Reads computed fairness evidence\
-   Explains disparities\
-   Classifies severity\
-   Proposes diagnostic tests\
-   Suggests mitigation strategies

**Design constraint:**\
The agent does **not** compute metrics.\
It only reasons from provided evidence.

This guarantees reproducibility and prevents hallucinated values.

------------------------------------------------------------------------

## User Interface (Streamlit Dashboard)

The project includes an interactive UI for inspecting audit results.

The dashboard displays:

-   Group-level fairness tables\
-   Disparity summaries\
-   Detected issues\
-   Agent explanations\
-   Recommended investigations\
-   Mitigation suggestions

Run UI:

    streamlit run ui/app.py

------------------------------------------------------------------------

## Local AI Model

The reasoning agent runs on a **local LLM via Ollama**.

Benefits:

-   No API cost\
-   Private inference\
-   Reproducible outputs\
-   Offline capability

Install Ollama and download the model:

    brew install ollama
    ollama pull llama3.2

------------------------------------------------------------------------

## Project Structure

    project/
    │
    ├── config/
    │   └── audit_config.yaml
    │
    ├── src/
    │   ├── init.py
    │   ├── train.py
    │   ├── evaluate_fairness_metrics.py
    │   ├── agent.py
    │   ├── model.py
    │   └── reporting.py
    │
    ├── ui/
    │   └── app.py
    │
    ├── outputs/
    │   └── runs/latest/
    │       ├── fairness_report.json
    │       └── agent_report.json
    │
    └── README.md

------------------------------------------------------------------------

## Design Principles

-   Interpretable\
-   Reproducible\
-   Evidence-based\
-   Modular\
-   Auditable\
-   Deterministic + Agent hybrid architecture

------------------------------------------------------------------------

## Installation

Create environment:

    python -m venv venv
    source venv/bin/activate

Install dependencies:

    pip install fairlearn pandas scikit-learn numpy matplotlib pyyaml streamlit

Install local LLM runtime:

    brew install ollama
    ollama pull llama3.2

------------------------------------------------------------------------

## Run Pipeline

Train and evaluate:

    python src/train.py --config config/audit_config.yaml

Run agent:

    python -m src.agent

Launch UI:

    streamlit run ui/app.py

------------------------------------------------------------------------

## Why This Project Is Different

Most fairness projects stop at computing metrics.

This system:

-   Detects disparities\
-   Explains them\
-   Hypothesizes causes\
-   Recommends investigations\
-   Proposes mitigation strategies

That makes it a reasoning system, not just a metric calculator.

------------------------------------------------------------------------

## One-Line Summary

An automated fairness auditor that not only measures disparities but
interprets and explains them.
