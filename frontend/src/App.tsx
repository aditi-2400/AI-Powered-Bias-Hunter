import { FormEvent, useEffect, useState } from "react";
import { createRun, getHealth, getRun, getRuns, RunListItem, RunSummary } from "./api";

function pretty(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

function App() {
  const [health, setHealth] = useState<string>("checking");
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("latest");
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [configFile, setConfigFile] = useState<File | null>(null);
  const [datasetFile, setDatasetFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  async function refreshRuns() {
    const data = await getRuns();
    setRuns(data.runs);
  }

  async function refreshSummary(runId: string) {
    const data = await getRun(runId);
    setSummary(data);
    setSelectedRun(runId);
  }

  useEffect(() => {
    async function boot() {
      try {
        const healthData = await getHealth();
        setHealth(healthData.status);
        await refreshRuns();
        await refreshSummary("latest");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Unknown error");
      }
    }
    void boot();
  }, []);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!configFile) {
      setError("Choose a config YAML before creating a run.");
      return;
    }

    setError("");
    setLoading(true);
    try {
      const result = await createRun(configFile, datasetFile);
      await refreshRuns();
      setSummary(result);
      setSelectedRun(result.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run creation failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <p className="eyebrow">Fairness Audit Dashboard</p>
          <h1>Bias Hunter</h1>
          <p className="muted">React frontend for the FastAPI backend and fairness pipeline.</p>
        </div>

        <div className="panel">
          <h2>Backend status</h2>
          <p className={`status ${health === "ok" ? "up" : "down"}`}>{health}</p>
        </div>

        <div className="panel">
          <h2>Create run</h2>
          <form onSubmit={onSubmit} className="upload-form">
            <label>
              Config YAML
              <input
                type="file"
                accept=".yaml,.yml"
                onChange={(e) => setConfigFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <label>
              Dataset
              <input
                type="file"
                accept=".csv,.tsv,.txt,.data"
                onChange={(e) => setDatasetFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <button type="submit" disabled={loading}>
              {loading ? "Running..." : "Run pipeline"}
            </button>
          </form>
        </div>

        <div className="panel">
          <div className="row-heading">
            <h2>Runs</h2>
            <button type="button" className="ghost" onClick={() => void refreshRuns()}>
              Refresh
            </button>
          </div>
          <button
            type="button"
            className={`run-item ${selectedRun === "latest" ? "active" : ""}`}
            onClick={() => void refreshSummary("latest")}
          >
            latest
          </button>
          {runs.map((run) => (
            <button
              key={run.run_id}
              type="button"
              className={`run-item ${selectedRun === run.run_id ? "active" : ""}`}
              onClick={() => void refreshSummary(run.run_id)}
            >
              <span>{run.run_id}</span>
              <small>{run.has_agent_report ? "complete" : "partial"}</small>
            </button>
          ))}
        </div>
      </aside>

      <main className="content">
        <header className="hero">
          <div>
            <p className="eyebrow">Selected run</p>
            <h2>{summary?.run_id ?? selectedRun}</h2>
          </div>
          {error ? <p className="error-banner">{error}</p> : null}
        </header>

        <section className="grid grid-metrics">
          <article className="card">
            <h3>Metrics</h3>
            <pre>{pretty(summary?.metrics ?? {})}</pre>
          </article>
          <article className="card">
            <h3>Fairness report</h3>
            <pre>{pretty(summary?.fairness_report ?? {})}</pre>
          </article>
        </section>

        <section className="grid">
          <article className="card">
            <h3>Diagnostics</h3>
            <pre>{pretty(summary?.diagnostics_run_summary ?? {})}</pre>
          </article>
          <article className="card">
            <h3>Agent plan</h3>
            <pre>{pretty(summary?.agent_plan ?? {})}</pre>
          </article>
        </section>

        <section className="card full-width">
          <h3>Agent report</h3>
          <pre>{pretty(summary?.agent_report ?? {})}</pre>
        </section>
      </main>
    </div>
  );
}

export default App;
