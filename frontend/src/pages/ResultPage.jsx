function getRiskColor(risk) {
  if (risk === "LOW") return "green"
  if (risk === "MEDIUM") return "yellow"
  if (risk === "HIGH") return "red"
  return "blue"
}

function getDecisionClass(decision) {
  if (decision === "APPROVED") return "approve"
  return "reject"
}

export default function ResultPage({ result, onBack }) {
  if (!result) {
    return (
      <div className="result-page animate-in">
        <p style={{ color: "var(--muted)" }}>
          No result yet. Go back and run a prediction.
        </p>

        <button
          className="back-btn"
          onClick={onBack}
        >
          ← Back to Form
        </button>
      </div>
    )
  }

  const riskColor = getRiskColor(result.risk_level)

  const probability =
    result.default_probability !== null &&
    result.default_probability !== undefined
      ? result.default_probability
      : null

  const probabilityPercent =
    probability !== null
      ? (probability * 100).toFixed(1)
      : null

  return (
    <div className="result-page animate-in">

      {/* Verdict */}
      <div
        className={`verdict-banner ${getDecisionClass(
          result.decision
        )}`}
      >
        <div className="verdict-icon">
          {result.decision === "APPROVED" ? "✅" : "❌"}
        </div>

        <div className="verdict-text">
          <h2>{result.decision}</h2>

          <p>
            {result.eligible
              ? "Applicant passed the eligibility rules and was evaluated by the default-risk model."
              : result.reason}
          </p>
        </div>
      </div>

      {/* Main metrics */}
      <div className="result-grid">

        {/* Eligibility */}
        <div className="metric-card">
          <div className="metric-label">
            Eligibility
          </div>

          <div
            className={`metric-value ${
              result.eligible ? "green" : "red"
            }`}
          >
            {result.eligible
              ? "Eligible"
              : "Not Eligible"}
          </div>
        </div>

        {/* Decision */}
        <div className="metric-card">
          <div className="metric-label">
            Final Decision
          </div>

          <div
            className={`metric-value ${
              result.decision === "APPROVED"
                ? "green"
                : "red"
            }`}
          >
            {result.decision}
          </div>
        </div>

        {/* Risk */}
        <div className="metric-card">
          <div className="metric-label">
            Default Risk
          </div>

          <div
            className={`metric-value ${riskColor}`}
          >
            {result.risk_level ?? "—"}
          </div>

          <div
            style={{
              marginTop: "0.5rem",
              fontSize: "0.82rem",
              color: "var(--muted)",
            }}
          >
            ML risk assessment
          </div>
        </div>

        {/* Default Probability */}
        <div className="metric-card">
          <div className="metric-label">
            Default Probability
          </div>

          <div
            className={`metric-value ${riskColor}`}
          >
            {probabilityPercent !== null
              ? `${probabilityPercent}%`
              : "—"}
          </div>

          <div
            style={{
              marginTop: "0.5rem",
              fontSize: "0.82rem",
              color: "var(--muted)",
            }}
          >
            {result.eligible
              ? `Decision threshold: ${result.threshold}`
              : "ML prediction skipped"}
          </div>
        </div>

      </div>

      {/* Failed eligibility rules */}
      {result.failed_rules?.length > 0 && (
        <div
          className="card card-sm"
          style={{ marginBottom: "1rem" }}
        >
          <div
            className="metric-label"
            style={{ marginBottom: "0.5rem" }}
          >
            Eligibility Issues
          </div>

          <ul className="reasons-list">
            {result.failed_rules.map((rule, index) => (
              <li key={index}>
                {rule.replace(/_/g, " ")}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Explanation */}
      {result.eligible && (
        <div
          className="card card-sm"
          style={{ marginBottom: "1rem" }}
        >
          <div
            className="metric-label"
            style={{ marginBottom: "0.5rem" }}
          >
            How the decision was made
          </div>

          <p style={{ color: "var(--muted)" }}>
            The applicant passed the deterministic eligibility
            rules. The Logistic Regression model then estimated
            the probability of loan default.
          </p>

          <p style={{ color: "var(--muted)" }}>
            If the default probability is below{" "}
            <strong>{result.threshold}</strong>, the application
            is approved. Otherwise, it is rejected.
          </p>
        </div>
      )}

      <button
        className="back-btn"
        onClick={onBack}
      >
        ← New Prediction
      </button>

    </div>
  )
}