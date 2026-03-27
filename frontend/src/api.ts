export type RunListItem = {
  run_id: string;
  path: string;
  has_metrics: boolean;
  has_fairness_report: boolean;
  has_agent_report: boolean;
};

export type RunSummary = {
  run_id: string;
  metrics?: Record<string, unknown> | null;
  fairness_report?: Record<string, unknown> | null;
  diagnostics_run_summary?: Record<string, unknown> | null;
  agent_plan?: Record<string, unknown> | null;
  agent_report?: Record<string, unknown> | null;
};

const API_BASE = "/api";

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<{ status: string }> {
  return fetchJson("/health");
}

export async function getRuns(): Promise<{ runs: RunListItem[] }> {
  return fetchJson("/runs");
}

export async function getRun(runId: string): Promise<RunSummary> {
  return fetchJson(`/runs/${runId}`);
}

export async function createRun(configFile: File, datasetFile?: File | null): Promise<RunSummary> {
  const formData = new FormData();
  formData.append("config_file", configFile);
  if (datasetFile) {
    formData.append("dataset_file", datasetFile);
  }
  return fetchJson("/runs", {
    method: "POST",
    body: formData,
  });
}
