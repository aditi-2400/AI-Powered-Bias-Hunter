from __future__ import annotations
import json, os, yaml
import streamlit as st
import pandas as pd
from typing import Any, Dict

DEFAULT_RUN_DIR = "outputs/runs/latest"
DEFAULT_CONFIG_PATH = "config/config_audit.yaml"

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding="utf-8") as f:
        return json.load(f)
    
def load_yaml(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
def fairness_tables(fairness_report):
    out = {}
    for attr, payload in fairness_report.items():
        by_group = payload.get("by_group", {})
        groups =  set()
        for metric, group_map in by_group.items():
            groups |= set(group_map.keys())
        
        rows = []
        for g in sorted(groups):
            row = {"group": g}
            for metric, group_map in by_group.items():
                row[metric] = group_map.get(g)
            rows.append(row)

        df = pd.DataFrame(rows).set_index("group")
        out[attr] = df
    return out


def issue_df(agent_report):
    issues = agent_report.get("detected_issues", [])
    if not issues:
        return pd.DataFrame(columns=["attribute","metric","issue_type","severity","evidence"])
    flat = []
    for it in issues:
        flat.append({
            "attribute": it.get("attribute"),
            "metric": it.get("metric"),
            "issue_type": it.get("issue_type"),
            "severity": it.get("severity"),
            "evidence": "\n".join(it.get("evidence")),          
        })
    return pd.DataFrame(flat)

st.set_page_config(page_title="Fairness Auditor",layout="wide")
st.title("AI-Powered Fairness Auditor")

#Sidebar controls
st.sidebar.header("Run Selection")
run_dir = st.sidebar.text_input("Run Directory", value=DEFAULT_RUN_DIR)
config_path = st.sidebar.text_input("Config path", value=DEFAULT_CONFIG_PATH)

refresh = st.sidebar.button("Refresh")
config = load_yaml(config_path)
fairness = load_json(os.path.join(run_dir, "fairness_report.json"))
agent = load_json(os.path.join(run_dir, "agent_report.json"))
metrics = load_json(os.path.join(run_dir, "metrics.json"))

colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Artifacts")
    st.write("fairness_report.json:", "✅" if fairness else "❌")
    st.write("agent_report.json:", "✅" if agent else "❌")
    st.write("metrics.json:", "✅" if metrics else "❌")

with colB:
    st.subheader("Config")
    if config:
        st.json({
            "dataset": config.get("dataset"),
            "label_col": config.get("label_col"),
            "positive_label": config.get("positive_label"),
            "sensitive_cols": config.get("sensitive_cols"),
            "fairness_threshold": config.get("fairness_threshold"),
            "min_group_size": config.get("min_group_size"),
            "model": config.get("model"),
        })
    else:
        st.warning(f"Missing config: {config_path}")

with colC:
    st.subheader("Agent Summary")
    if agent:
        st.write(agent.get("summary",""))
    else:
        st.info("Run the agent to generate agent_report.json")

st.divider()
left, right = st.columns(2)
with left:
    st.header("Fairness Report")
    
    if not fairness:
        st.warning("Missing fairness_report.json")
    else:
        tables = fairness_tables(fairness)
        for attr, df in tables.items():
            st.subheader(f"{attr}: By group metrics")
            st.dataframe(df, use_container_width=True)

            diff = fairness[attr].get("difference",{})
            ratio = fairness[attr].get("ratio",{})
            flags = fairness[attr].get("flags",{})

            summary_rows = []
            for metric in sorted(set(diff.keys()) | set(ratio.keys()) | set(flags.keys())):
                summary_rows.append({
                    "metric": metric,
                    "difference": diff.get(metric),
                    "ratio": ratio.get(metric),
                    "flag": flags.get(metric),
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
with right:
    st.header("Agent Report")

    if not agent:
        st.warning("Missing agent_report.json")
    else:
        st.subheader("Detected Issues")
        st.dataframe(issue_df(agent), use_container_width=True)

        st.subheader("Narrative")
        narrative = agent.get("narrative_markdown","")
        st.markdown(narrative if narrative else "_No narrative provided_")

        st.subheader("Recommendations")
        st.write("Recommended Tests:", agent.get("recommended_tests"),[])
        st.write("Mitigations:", agent.get("mitigation"),[])
        st.write("Limits:", agent.get("limits"), [])

st.divider()

st.header("Download Artifacts")
dcol1, dcol2 = st.columns(2)
with dcol1:
    if fairness:
        st.download_button(
            "Download fairness_report.json",
            data=json.dumps(fairness, indent=2),
            file_name="fairness_report.json",
            mime="application/json",
        )
with dcol2:
    if agent:
        st.download_button(
            "Download agent_report.json",
            data=json.dumps(agent, indent=2),
            file_name="agent_report.json",
            mime="application/json",
        )
