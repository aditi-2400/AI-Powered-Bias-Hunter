from __future__ import annotations 
import json 
import os 
from typing import Any, Dict, List, Tuple 
from src.agent_common import ( 
    DEFAULT_CONFIG_PATH, 
    DEFAULT_MODEL, 
    DEFAULT_RUN_DIR, 
    call_ollama, 
    load_evidence, 
    load_yaml, 
    parse_model_json, 
    write_json, 
    write_text, 
    ) 
REQUIRED_OUTPUT_KEYS = ["requested_diagnostics"] 
ALLOWED_TOOLS = { 
    "check_group_sample_sizes", 
    "run_feature_distribution_comparison", 
    "run_proxy_detection", 
    "run_slice_scan", 
    "run_threshold_sensitivity", 
    } 
def build_plan_prompt(evidence: Dict[str, Any]) -> str: 
    cfg = evidence.get("audit_config", {}) 
    threshold = cfg.get("fairness_threshold", 0.05) 
    evidence_str = json.dumps(evidence, indent=2, ensure_ascii=False) 
    return f""" 
    You are an AI fairness audit planner. You will receive JSON evidence produced by a deterministic pipeline.

    GOAL:
    Choose the most useful next deterministic diagnostics to run that have NOT already been executed.

    STRICT RULES:
    - Use ONLY the provided JSON evidence. Do NOT invent numbers.
    - Output MUST be valid JSON (one top-level object) and NOTHING else.
    - Return only diagnostics from this allowed set:
    - check_group_sample_sizes
    - run_feature_distribution_comparison
    - run_proxy_detection
    - run_slice_scan
    - run_threshold_sensitivity
    - Tool names MUST be exactly one of the allowed names above.
    - Put the sensitive attribute only inside args, never inside the tool name.
    - Example valid request:
    {{"tool": "run_threshold_sensitivity", "args": {{"attribute": "<sensitive_attribute>"}}, "reason": "TPR/FPR disparities are flagged for this attribute."}}
    - Example invalid request:
    {{"tool": "run_threshold_sensitivity__<sensitive_attribute>", "args": {{"attribute": "<sensitive_attribute>"}}, "reason": "..." }}
    - Do NOT request the same tool multiple times with identical args.
    - If no additional diagnostics are needed, return an empty list.
    - Prefer targeted diagnostics tied to flagged disparities.
    - Do NOT request diagnostics that have already been run and whose artifacts are already present in evidence.
    - If group_sizes exists, do NOT request check_group_sample_sizes again.
    - If diagnostics.feature_distribution__<attribute> exists, do NOT request run_feature_distribution_comparison again for that attribute.
    - If diagnostics.proxy_detection__<attribute> exists, do NOT request run_proxy_detection again for that attribute.
    - If diagnostics.threshold_sensitivity__<attribute> exists, do NOT request run_threshold_sensitivity again for that attribute.
    - If diagnostics.slice_scan exists, do NOT request run_slice_scan again.
    - If group_sizes exists and any group count is below audit_config.min_group_size, use that as planning context rather than requesting check_group_sample_sizes again.
    - If selection_rate disparity is flagged for an attribute and no feature_distribution artifact exists for that attribute, consider run_feature_distribution_comparison and run_proxy_detection for that attribute.
    - If true_positive_rate or false_positive_rate disparity is flagged for an attribute and metrics.has_y_score is true and no threshold_sensitivity artifact exists for that attribute, consider run_threshold_sensitivity for that attribute.
    - If true_positive_rate or false_positive_rate disparity is flagged for an attribute and no slice_scan artifact exists, consider run_slice_scan.
    - For tool args:
    - check_group_sample_sizes uses {{}}
    - run_feature_distribution_comparison may use {{"attribute": "<sensitive_attribute>"}}
    - run_proxy_detection may use {{"attribute": "<sensitive_attribute>"}}
    - run_slice_scan may use {{}}
    - run_threshold_sensitivity may use {{"attribute": "<sensitive_attribute>"}}

    PROJECT CONTEXT:
    - fairness_threshold = {threshold}
    - If metrics.has_y_score is true, threshold sensitivity can be run.

    OUTPUT JSON SCHEMA:
    {{
    "requested_diagnostics": [
        {{
        "tool": string,
        "args": object,
        "reason": string
        }}
    ]
    }}

    EVIDENCE JSON:
    {evidence_str}
    """.strip()

def validate_plan_output(obj: Dict[str, Any]) -> Tuple[bool, List[str]]: 
    errors: List[str] = [] 
    for k in REQUIRED_OUTPUT_KEYS: 
        if k not in obj: errors.append(f"Missing key {k}") 
        
    items = obj.get("requested_diagnostics") 
    if items is not None and not isinstance(items, list): 
        errors.append("requested_diagnostics must be a list") 
        return (len(errors) == 0), errors 
    
    seen = set() 
    for i, item in enumerate(items or []): 
        if not isinstance(item, dict): 
            errors.append(f"requested_diagnostics[{i}] must be an object") 
            continue 
        tool = item.get("tool") 
        args = item.get("args") 
        reason = item.get("reason") 
    
        if tool not in ALLOWED_TOOLS: 
            errors.append( f"requested_diagnostics[{i}].tool must be one of {sorted(ALLOWED_TOOLS)}" ) 
        
        if not isinstance(args, dict): 
            errors.append(f"requested_diagnostics[{i}].args must be an object") 
        
        if reason is not None and not isinstance(reason, str): 
            errors.append(f"requested_diagnostics[{i}].reason must be a string") 
            
        if isinstance(args, dict): 
            key = (tool, json.dumps(args, sort_keys=True)) 
            if key in seen: 
                errors.append(f"Duplicate diagnostic request for tool={tool} args={args}") 
            seen.add(key) 
            
    return (len(errors) == 0), errors 
        
    
def run_agent_plan( 
        run_dir: str = DEFAULT_RUN_DIR, 
        model: str = DEFAULT_MODEL, 
        config_path: str = DEFAULT_CONFIG_PATH, ) -> Dict[str, Any]: 
    if not os.path.exists(config_path): 
        return { "requested_diagnostics": [], "errors": [f"Config not found: {config_path}"], } 
    
    config = load_yaml(config_path) 
    evidence = load_evidence(run_dir, config) 
    if evidence["_meta"]["missing_required"]: 
        return { "requested_diagnostics": [], "errors": [f"Missing required files: {', '.join(evidence['_meta']['missing_required'])}"], } 
    
    prompt = build_plan_prompt(evidence) 
    write_text(os.path.join(run_dir, "agent_plan_prompt.txt"), prompt) 
    
    raw = call_ollama(model, prompt) 
    try: 
        obj = parse_model_json(raw) 
    except ValueError as e: 
        write_text(os.path.join(run_dir, "agent_plan_raw.txt"), raw) 
        return { 
            "requested_diagnostics": [], 
            "errors": [str(e)], 
            } 
        
    ok, errors = validate_plan_output(obj) 
    if not ok: 
        write_text(os.path.join(run_dir, "agent_plan_raw.txt"), raw) 
        return { "requested_diagnostics": [], "errors": errors, } 
    
    return obj 

def main(): 
    run_dir = os.environ.get("FAIRNESS_RUN_DIR", DEFAULT_RUN_DIR) 
    model = os.environ.get("FAIRNESS_AGENT_MODEL", DEFAULT_MODEL) 
    config_path = os.environ.get("FAIRNESS_CONFIG", DEFAULT_CONFIG_PATH) 
    plan = run_agent_plan(run_dir=run_dir, model=model, config_path=config_path) 
    out_path = os.path.join(run_dir, "agent_plan.json") 
    write_json(out_path, plan) 
    print(f"Wrote {out_path}") 
    

if __name__ == "__main__": 
    main()
