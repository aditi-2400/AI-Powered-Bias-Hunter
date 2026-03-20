from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import load_dataset_from_config
from src.agent_plan import run_agent_plan
from src.agent_report import run_agent_report
from src.evaluate_fairness_metrics import evaluate_fairness
from src.run_diagnostics import run_diagnostics
from src.train import run_training

DEFAULT_RUN_DIR = "outputs/runs/latest"
DEFAULT_CONFIG_PATH = "config/audit_config.yaml"
UPLOAD_ROOT = Path("/tmp/bias_hunter_uploads")


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_uploaded_file(uploaded_file, target_dir: Path) -> str:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / uploaded_file.name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(out_path)


def write_yaml(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def fairness_tables(fairness_report: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for attr, payload in fairness_report.items():
        by_group = payload.get("by_group", {})
        groups = set()
        for _, group_map in by_group.items():
            groups |= set(group_map.keys())

        rows = []
        for g in sorted(groups):
            row = {"group": g}
            for metric, group_map in by_group.items():
                row[metric] = group_map.get(g)
            rows.append(row)

        if rows:
            out[attr] = pd.DataFrame(rows).set_index("group")
    return out


def issue_df(agent_report: Dict[str, Any]) -> pd.DataFrame:
    issues = agent_report.get("detected_issues", [])
    if not issues:
        return pd.DataFrame(columns=["attribute", "metric", "issue_type", "severity", "evidence"])

    flat = []
    for it in issues:
        flat.append(
            {
                "attribute": it.get("attribute"),
                "metric": it.get("metric"),
                "issue_type": it.get("issue_type"),
                "severity": it.get("severity"),
                "evidence": "\n".join(it.get("evidence", [])),
            }
        )
    return pd.DataFrame(flat)


def resolve_artifact_path(run_dir: str, output_file: str) -> str:
    if os.path.exists(output_file):
        return output_file
    prefix = "outputs/runs/latest/"
    if output_file.startswith(prefix):
        rel = output_file[len(prefix) :]
        return os.path.join(run_dir, rel)
    return output_file


def diagnostics_highlights(run_dir: str, diagnostics_summary: Dict[str, Any]) -> List[str]:
    lines: List[Dict[str, Any]] = []
    for item in diagnostics_summary.get("executed", []) or []:
        tool = item.get("tool", "unknown")
        args = item.get("args", {}) or {}
        artifact = resolve_artifact_path(run_dir, item.get("output_file", ""))
        payload = load_json(artifact)

        if payload is None:
            lines.append(
                {
                    "tool": tool,
                    "attribute": args.get("attribute", "-"),
                    "detail": "Artifact missing",
                    "n": None,
                    "selection_rate": None,
                    "true_positive_rate": None,
                    "false_positive_rate": None,
                }
            )
            continue

        if tool == "run_threshold_sensitivity":
            for attr, block in payload.items():
                results = block.get("threshold_results", [])
                if not results:
                    continue
                best = min(
                    results,
                    key=lambda r: float(r.get("difference", {}).get("true_positive_rate", 1e9)),
                )
                d = best.get("difference", {})
                lines.append(
                    {
                        "tool": tool,
                        "attribute": attr,
                        "detail": f"Best TPR-gap threshold = {best.get('threshold')}",
                        "n": None,
                        "selection_rate": None,
                        "true_positive_rate": d.get("true_positive_rate"),
                        "false_positive_rate": d.get("false_positive_rate"),
                    }
                )
        elif tool == "run_slice_scan":
            top = payload.get("top_slices", [])[:3]
            if not top:
                lines.append(
                    {
                        "tool": tool,
                        "attribute": args.get("attribute", "-"),
                        "detail": "No slices found",
                        "n": None,
                        "selection_rate": None,
                        "true_positive_rate": None,
                        "false_positive_rate": None,
                    }
                )
            for row in top:
                lines.append(
                    {
                        "tool": tool,
                        "attribute": args.get("attribute", "-"),
                        "detail": f"{row.get('feature')}={row.get('value')}",
                        "n": row.get("n"),
                        "selection_rate": row.get("selection_rate"),
                        "true_positive_rate": row.get("true_positive_rate"),
                        "false_positive_rate": row.get("false_positive_rate"),
                    }
                )
        else:
            lines.append(
                {
                    "tool": tool,
                    "attribute": args.get("attribute", "-"),
                    "detail": "Output captured",
                    "n": None,
                    "selection_rate": None,
                    "true_positive_rate": None,
                    "false_positive_rate": None,
                }
            )
    return lines


def render_string_list(title: str, items: List[str], empty_text: str) -> None:
    st.markdown(title)
    if items:
        for item in items:
            st.write(f"- {item}")
    else:
        st.caption(empty_text)


def prepare_uploaded_inputs(uploaded_config, uploaded_dataset) -> Optional[str]:
    if uploaded_config is None:
        st.sidebar.error("Upload a config YAML file first.")
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = UPLOAD_ROOT / stamp

    config_path = save_uploaded_file(uploaded_config, target_dir)
    config = load_yaml(config_path)
    if not config:
        st.sidebar.error("Uploaded config could not be parsed as YAML.")
        return None

    if uploaded_dataset is not None:
        dataset_path = save_uploaded_file(uploaded_dataset, target_dir)
        config.setdefault("dataset", {})
        config["dataset"]["path"] = dataset_path

    effective_config_path = str(target_dir / "effective_config.yaml")
    write_yaml(effective_config_path, config)

    st.session_state["effective_config_path"] = effective_config_path
    st.session_state["effective_upload_dir"] = str(target_dir)
    return effective_config_path


def run_pipeline(effective_config_path: str) -> None:
    config = load_yaml(effective_config_path)
    if not config:
        st.error("Effective config is missing or invalid.")
        return

    try:
        with st.spinner("Running training..."):
            train_result = run_training(config)
            run_dir = train_result["out_dir"]

        with st.spinner("Running fairness evaluation..."):
            evaluate_fairness(run_dir, config)

        with st.spinner("Running agent plan..."):
            plan = run_agent_plan(run_dir=run_dir, config_path=effective_config_path)
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, "agent_plan.json"), "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2)

        with st.spinner("Running diagnostics..."):
            run_diagnostics(run_dir=run_dir, config_path=effective_config_path)

        with st.spinner("Running agent report..."):
            report = run_agent_report(run_dir=run_dir, config_path=effective_config_path)
            with open(os.path.join(run_dir, "agent_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        latest_dir = DEFAULT_RUN_DIR
        st.session_state["run_dir"] = latest_dir
        st.success(f"Pipeline completed. Latest artifacts are available in {latest_dir}.")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")


def load_dataset_preview(config: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    if not config:
        return None
    try:
        return load_dataset_from_config(config)
    except Exception:
        return None


def render_column_section(df: pd.DataFrame, col: str) -> None:
    series = df[col]
    st.markdown(f"**{col}**")
    st.write(f"dtype: `{series.dtype}` | non-null: `{int(series.notna().sum())}` | null: `{int(series.isna().sum())}`")

    if pd.api.types.is_numeric_dtype(series):
        st.dataframe(series.describe().to_frame(name="value"), use_container_width=True)
    else:
        counts = series.astype(str).value_counts(dropna=False).head(15)
        st.dataframe(counts.to_frame(name="count"), use_container_width=True)


st.set_page_config(page_title="Fairness Auditor", layout="wide")
st.title("Fairness Auditor")

if "run_dir" not in st.session_state:
    st.session_state["run_dir"] = DEFAULT_RUN_DIR

st.sidebar.header("Inputs")
mode = st.sidebar.radio("Source", ["Existing artifacts", "Upload config + dataset"], index=0)

if mode == "Upload config + dataset":
    uploaded_config = st.sidebar.file_uploader("Config YAML", type=["yaml", "yml"])
    uploaded_dataset = st.sidebar.file_uploader("Dataset file", type=["csv", "tsv", "txt", "data"])

    if st.sidebar.button("Use uploaded files"):
        cfg_path = prepare_uploaded_inputs(uploaded_config, uploaded_dataset)
        if cfg_path:
            st.sidebar.success("Uploaded files are active for this session.")

    if st.sidebar.button("Run pipeline with uploaded files"):
        cfg_path = st.session_state.get("effective_config_path")
        if not cfg_path:
            cfg_path = prepare_uploaded_inputs(uploaded_config, uploaded_dataset)
        if cfg_path:
            run_pipeline(cfg_path)

run_dir = st.sidebar.text_input("Run Directory", value=st.session_state.get("run_dir", DEFAULT_RUN_DIR))
if st.sidebar.button("Refresh"):
    st.session_state["run_dir"] = run_dir
    st.rerun()

active_config_path = st.session_state.get("effective_config_path", DEFAULT_CONFIG_PATH)
config = load_yaml(active_config_path)
fairness = load_json(os.path.join(run_dir, "fairness_report.json"))
agent = load_json(os.path.join(run_dir, "agent_report.json"))
metrics = load_json(os.path.join(run_dir, "metrics.json"))
diag_summary = load_json(os.path.join(run_dir, "diagnostics_run_summary.json"))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{metrics.get('accuracy', 'NA'):.3f}" if metrics and metrics.get("accuracy") is not None else "NA")
with col2:
    st.metric("Train Rows", str(metrics.get("n_train", "NA")) if metrics else "NA")
with col3:
    st.metric("Test Rows", str(metrics.get("n_test", "NA")) if metrics else "NA")

tabs = st.tabs(["Overview", "Fairness by Column", "Diagnostics", "Agent", "Dataset Columns", "Downloads"])

with tabs[0]:
    st.subheader("Artifact status")
    st.write("fairness_report.json:", "✅" if fairness else "❌")
    st.write("agent_report.json:", "✅" if agent else "❌")
    st.write("metrics.json:", "✅" if metrics else "❌")
    st.write("diagnostics_run_summary.json:", "✅" if diag_summary else "❌")

    with st.expander("Active config"):
        st.json(config if config else {})

with tabs[1]:
    st.subheader("Fairness by sensitive attribute")
    if not fairness:
        st.info("Missing fairness_report.json")
    else:
        tables = fairness_tables(fairness)
        for attr in fairness.keys():
            with st.expander(f"{attr}", expanded=False):
                if attr in tables:
                    st.markdown("By-group metrics")
                    st.dataframe(tables[attr], use_container_width=True)

                diff = fairness[attr].get("difference", {})
                ratio = fairness[attr].get("ratio", {})
                flags = fairness[attr].get("flags", {})
                rows = []
                for metric in sorted(set(diff.keys()) | set(ratio.keys()) | set(flags.keys())):
                    rows.append(
                        {
                            "metric": metric,
                            "difference": diff.get(metric),
                            "ratio": ratio.get(metric),
                            "flag": flags.get(metric),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tabs[2]:
    st.subheader("Diagnostics")
    if not diag_summary:
        st.info("No diagnostics summary found.")
    else:
        executed = diag_summary.get("executed", []) or []
        errors = diag_summary.get("errors", []) or []

        st.markdown("Executed diagnostics")
        if executed:
            st.dataframe(pd.DataFrame(executed), use_container_width=True)
        else:
            st.caption("No diagnostics executed.")

        st.markdown("Highlights")
        highlights = diagnostics_highlights(run_dir, diag_summary)
        if highlights:
            hdf = pd.DataFrame(highlights)
            for col in ["selection_rate", "true_positive_rate", "false_positive_rate"]:
                if col in hdf.columns:
                    hdf[col] = pd.to_numeric(hdf[col], errors="coerce").round(4)
            st.dataframe(hdf, use_container_width=True)
        else:
            st.caption("No highlights available.")

        if errors:
            st.markdown("Errors")
            st.dataframe(pd.DataFrame(errors), use_container_width=True)

with tabs[3]:
    st.subheader("Agent explanation")
    if not agent:
        st.info("Missing agent_report.json")
    else:
        st.dataframe(issue_df(agent), use_container_width=True)

        causes = agent.get("likely_causes", []) or []
        render_string_list("Likely causes", causes, "No likely causes provided.")

        st.markdown("Narrative")
        st.markdown(agent.get("narrative_markdown", "_No narrative provided_"))

        st.markdown("Recommendations")
        render_string_list("Recommended tests", agent.get("recommended_tests", []) or [], "No additional tests recommended.")
        render_string_list("Mitigations", agent.get("mitigations", []) or [], "No mitigations provided.")
        render_string_list("Limits", agent.get("limits", []) or [], "No explicit limits listed.")

with tabs[4]:
    st.subheader("Dataset columns")
    dataset_df = load_dataset_preview(config)
    if dataset_df is None:
        st.info("Dataset could not be loaded from current config.")
    else:
        st.write(f"Rows: {len(dataset_df)} | Columns: {len(dataset_df.columns)}")
        for col in dataset_df.columns:
            with st.expander(col, expanded=False):
                render_column_section(dataset_df, col)

with tabs[5]:
    st.subheader("Download artifacts")
    if fairness:
        st.download_button(
            "Download fairness_report.json",
            data=json.dumps(fairness, indent=2),
            file_name="fairness_report.json",
            mime="application/json",
        )
    if agent:
        st.download_button(
            "Download agent_report.json",
            data=json.dumps(agent, indent=2),
            file_name="agent_report.json",
            mime="application/json",
        )
    if diag_summary:
        st.download_button(
            "Download diagnostics_run_summary.json",
            data=json.dumps(diag_summary, indent=2),
            file_name="diagnostics_run_summary.json",
            mime="application/json",
        )
