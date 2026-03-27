import { FormEvent, useEffect, useState } from "react";
import { createRun, getHealth, getRun, getRuns, RunListItem, RunSummary } from "./api";

type Dict = Record<string, unknown>;

function formatNumber(value: unknown, digits = 3): string {
  return typeof value === "number" ? value.toFixed(digits) : "NA";
}

function titleize(value: string): string {
  return value.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function toDict(value: unknown): Dict {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Dict) : {};
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function fairnessSections(summary: RunSummary | null): Array<{
  attribute: string;
  difference: Dict;
  ratio: Dict;
  flags: Dict;
  byGroup: Dict;
}> {
  const fairness = toDict(summary?.fairness_report);
  return Object.entries(fairness).map(([attribute, raw]) => {
    const section = toDict(raw);
    return {
      attribute,
      difference: toDict(section.difference),
      ratio: toDict(section.ratio),
      flags: toDict(section.flags),
      byGroup: toDict(section.by_group),
    };
  });
}

function issueCards(summary: RunSummary | null): Array<{
  attribute: string;
  metric: string;
  severity: string;
  issueType: string;
  evidence: string[];
}> {
  const report = toDict(summary?.agent_report);
  const issues = Array.isArray(report.detected_issues) ? report.detected_issues : [];
  return issues
    .map((issue) => toDict(issue))
    .map((issue) => ({
      attribute: String(issue.attribute ?? "unknown"),
      metric: String(issue.metric ?? "unknown"),
      severity: String(issue.severity ?? "unknown"),
      issueType: String(issue.issue_type ?? "unknown"),
      evidence: toStringArray(issue.evidence),
    }));
}

function diagnosticsItems(summary: RunSummary | null): Array<{ tool: string; args: string; output: string }> {
  const diagnostics = toDict(summary?.diagnostics_run_summary);
  const executed = Array.isArray(diagnostics.executed) ? diagnostics.executed : [];
  return executed.map((item) => {
    const row = toDict(item);
    return {
      tool: String(row.tool ?? "unknown"),
      args: JSON.stringify(row.args ?? {}),
      output: String(row.output_file ?? ""),
    };
  });
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
        setHealth("down");
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
      setHealth("ok");
      await refreshRuns();
      setSummary(result);
      setSelectedRun(result.run_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Run creation failed");
      setHealth("down");
    } finally {
      setLoading(false);
    }
  }

  const metrics = toDict(summary?.metrics);
  const report = toDict(summary?.agent_report);
  const plan = toDict(summary?.agent_plan);
  const fair = fairnessSections(summary);
  const issues = issueCards(summary);
  const diagnostics = diagnosticsItems(summary);
  const requestedDiagnostics = Array.isArray(plan.requested_diagnostics) ? plan.requested_diagnostics.length : 0;
  const recommendedTests = toStringArray(report.recommended_tests);
  const mitigations = toStringArray(report.mitigations);
  const limits = toStringArray(report.limits);
  const likelyCauses = toStringArray(report.likely_causes);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <p className="eyebrow">Fairness Audit Dashboard</p>
          <h1>Bias Hunter</h1>
          <p className="muted">Readable fairness findings, diagnostics, and recommendations from the pipeline.</p>
        </div>

        <div className="panel">
          <h2>Backend status</h2>
          <p className={`status ${health === "ok" ? "up" : "down"}`}>{health}</p>
        </div>

        <div className="panel">
          <h2>Create run</h2>
          <form onSubmit={onSubmit} className="upload-form">
            <label className="file-field">
              <span>Config YAML</span>
              <input
                type="file"
                accept=".yaml,.yml"
                onChange={(e) => setConfigFile(e.target.files?.[0] ?? null)}
              />
              <span className="file-name">{configFile?.name ?? "No file chosen"}</span>
            </label>
            <label className="file-field">
              <span>Dataset</span>
              <input
                type="file"
                accept=".csv,.tsv,.txt,.data"
                onChange={(e) => setDatasetFile(e.target.files?.[0] ?? null)}
              />
              <span className="file-name">{datasetFile?.name ?? "No file chosen"}</span>
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
            <p className="muted">Review model performance, fairness disparities, diagnostics, and next actions.</p>
          </div>
          {error ? <p className="error-banner">{error}</p> : null}
        </header>

        <section className="stats-row">
          <article className="stat-card">
            <span>Accuracy</span>
            <strong>{formatNumber(metrics.accuracy)}</strong>
          </article>
          <article className="stat-card">
            <span>Train rows</span>
            <strong>{String(metrics.n_train ?? "NA")}</strong>
          </article>
          <article className="stat-card">
            <span>Test rows</span>
            <strong>{String(metrics.n_test ?? "NA")}</strong>
          </article>
          <article className="stat-card">
            <span>Planned diagnostics</span>
            <strong>{requestedDiagnostics}</strong>
          </article>
        </section>

        <section className="grid">
          <article className="card">
            <div className="section-heading">
              <h3>Detected issues</h3>
              <span>{issues.length}</span>
            </div>
            {issues.length ? (
              <div className="stack">
                {issues.map((issue, index) => (
                  <div key={`${issue.attribute}-${issue.metric}-${index}`} className="issue-card">
                    <div className="issue-top">
                      <strong>{titleize(issue.attribute)}</strong>
                      <span className={`pill ${issue.severity}`}>{issue.severity}</span>
                    </div>
                    <p>{titleize(issue.metric)} · {titleize(issue.issueType)}</p>
                    <ul className="clean-list">
                      {issue.evidence.map((item) => (
                        <li key={item}>{item}</li>
                      ))}
                    </ul>
                  </div>
                ))}
              </div>
            ) : (
              <p className="empty-state">No detected issues available for this run.</p>
            )}
          </article>

          <article className="card">
            <div className="section-heading">
              <h3>Agent summary</h3>
              <span>{String(report.summary ?? "No summary")}</span>
            </div>
            <div className="stack">
              <div>
                <h4>Likely causes</h4>
                {likelyCauses.length ? (
                  <ul className="clean-list">
                    {likelyCauses.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">No likely causes provided.</p>
                )}
              </div>
              <div>
                <h4>Recommended tests</h4>
                {recommendedTests.length ? (
                  <ul className="clean-list">
                    {recommendedTests.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">No additional tests recommended.</p>
                )}
              </div>
              <div>
                <h4>Mitigations</h4>
                {mitigations.length ? (
                  <ul className="clean-list">
                    {mitigations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">No mitigations proposed.</p>
                )}
              </div>
              <div>
                <h4>Limits</h4>
                {limits.length ? (
                  <ul className="clean-list">
                    {limits.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="empty-state">No explicit limits listed.</p>
                )}
              </div>
            </div>
          </article>
        </section>

        <section className="card full-width">
          <div className="section-heading">
            <h3>Fairness by attribute</h3>
            <span>{fair.length} attributes</span>
          </div>
          {fair.length ? (
            <div className="attribute-grid">
              {fair.map((section) => {
                const metricsList = Object.keys(section.difference);
                return (
                  <div key={section.attribute} className="attribute-card">
                    <h4>{titleize(section.attribute)}</h4>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Metric</th>
                          <th>Diff</th>
                          <th>Ratio</th>
                          <th>Flag</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metricsList.map((metric) => (
                          <tr key={metric}>
                            <td>{titleize(metric)}</td>
                            <td>{formatNumber(section.difference[metric], 4)}</td>
                            <td>{formatNumber(section.ratio[metric], 4)}</td>
                            <td>{section.flags[metric] ? "Yes" : "No"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>

                    <div className="group-grid">
                      {Object.entries(section.byGroup).map(([metric, groups]) => (
                        <div key={metric} className="group-box">
                          <h5>{titleize(metric)}</h5>
                          <ul className="compact-list">
                            {Object.entries(toDict(groups)).map(([group, value]) => (
                              <li key={group}>
                                <span>{group}</span>
                                <strong>{formatNumber(value, 4)}</strong>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="empty-state">No fairness report available.</p>
          )}
        </section>

        <section className="grid">
          <article className="card">
            <div className="section-heading">
              <h3>Diagnostics</h3>
              <span>{diagnostics.length} executed</span>
            </div>
            {diagnostics.length ? (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Tool</th>
                      <th>Args</th>
                      <th>Output</th>
                    </tr>
                  </thead>
                  <tbody>
                    {diagnostics.map((item) => (
                      <tr key={`${item.tool}-${item.output}`}>
                        <td>{item.tool}</td>
                        <td className="wrap-cell">{item.args}</td>
                        <td className="wrap-cell">{item.output}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="empty-state">No diagnostics executed yet.</p>
            )}
          </article>
        </section>

        <section className="card full-width">
          <div className="section-heading">
            <h3>Narrative</h3>
            <span>From agent report</span>
          </div>
          <div className="narrative-box">
            {typeof report.narrative_markdown === "string" && report.narrative_markdown ? (
              <article className="narrative-markdown">
                {report.narrative_markdown.split("\n").map((line, index) => {
                  if (line.startsWith("## ")) {
                    return <h3 key={index}>{line.slice(3)}</h3>;
                  }
                  if (line.startsWith("# ")) {
                    return <h2 key={index}>{line.slice(2)}</h2>;
                  }
                  if (line.startsWith("* ")) {
                    return <p key={index} className="narrative-bullet">{line.slice(2)}</p>;
                  }
                  if (line.startsWith("- ")) {
                    return <p key={index} className="narrative-bullet">{line.slice(2)}</p>;
                  }
                  if (!line.trim()) {
                    return <div key={index} className="narrative-gap" />;
                  }
                  return <p key={index}>{line}</p>;
                })}
              </article>
            ) : (
              <p className="empty-state">No narrative available.</p>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
