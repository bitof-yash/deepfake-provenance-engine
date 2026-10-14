import { useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const TONE = {
  ACCEPT: "good",
  ACCEPT_UNVERIFIED: "ok",
  REVIEW: "warn",
  ESCALATE: "warn",
  REJECT: "bad",
};

const PROV_TEXT = {
  valid: "Valid",
  invalid: "Invalid",
  absent: "Absent",
  present_unverified: "Present, unverified",
  signed_untrusted: "Signed, issuer untrusted",
};

const MODALITY_TEXT = {
  video: "Video",
  audio: "Audio",
  "audio+video": "Audio + video",
  "cross-modal": "Cross-modal",
  lipsync: "Lip-sync",
};

function scoreColor(s) {
  if (s == null) return "var(--cell-null)";
  const hue = 130 - 130 * Math.min(1, Math.max(0, s));
  return `hsl(${hue} 65% ${48 - 6 * s}%)`;
}

function Gauge({ value, tone }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 130 130" className={`gauge ${tone}`} role="img" aria-label={`Trust score ${value}`}>
      <circle cx="65" cy="65" r={r} className="gauge-bg" />
      <circle
        cx="65"
        cy="65"
        r={r}
        className="gauge-fg"
        strokeDasharray={c}
        strokeDashoffset={c * (1 - value / 100)}
        transform="rotate(-90 65 65)"
      />
      <text x="65" y="63" textAnchor="middle" className="gauge-num">
        {Math.round(value)}
      </text>
      <text x="65" y="82" textAnchor="middle" className="gauge-sub">
        trust score
      </text>
    </svg>
  );
}

function Timeline({ result }) {
  const { windows, segments } = result;
  const tracks = [
    ["Video", "video"],
    ["Audio", "audio"],
    ["Lip-sync", "lipsync"],
    ["Fused", "fused"],
  ];
  return (
    <section className="card">
      <h2>Audio-visual timeline</h2>
      <div className="timeline">
        {tracks.map(([label, key]) => (
          <div className="track" key={key}>
            <span className="track-label">{label}</span>
            <div className="cells" style={{ gridTemplateColumns: `repeat(${windows.length}, 1fr)` }}>
              {windows.map((w) => (
                <div
                  key={w.start}
                  className={`cell ${key === "fused" && w.flagged ? "flag" : ""}`}
                  style={{ background: scoreColor(w[key]) }}
                  title={`${w.start}s–${w.end}s · ${label}: ${w[key] == null ? "n/a" : w[key].toFixed(2)}`}
                />
              ))}
            </div>
          </div>
        ))}
        <div className="track">
          <span className="track-label" />
          <div className="axis" style={{ gridTemplateColumns: `repeat(${windows.length}, 1fr)` }}>
            {windows.map((w) => (
              <span key={w.start}>{Math.round(w.start)}</span>
            ))}
          </div>
        </div>
      </div>
      <div className="legend">
        <span>low suspicion</span>
        <div className="legend-bar" />
        <span>high suspicion</span>
        <span className="legend-na">
          <i /> not available
        </span>
      </div>
      <h3>Flagged segments</h3>
      {segments.length === 0 ? (
        <p className="muted">No segments exceeded the manipulation threshold.</p>
      ) : (
        <ul className="segments">
          {segments.map((s, i) => (
            <li key={i}>
              <b>
                {s.start.toFixed(1)}s – {s.end.toFixed(1)}s
              </b>
              <span>{MODALITY_TEXT[s.modality] || s.modality}</span>
              <span className="muted">peak {s.peak_score.toFixed(2)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Results({ result, onReset }) {
  const { routing, provenance } = result;
  const tone = TONE[routing.decision] || "ok";
  return (
    <>
      <section className={`card verdict ${tone}`}>
        <Gauge value={routing.trust_score} tone={tone} />
        <div className="verdict-body">
          <div className="badge">{routing.label}</div>
          <h2>{routing.decision.replaceAll("_", " ")}</h2>
          <ul className="reasons">
            {routing.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
          <p className="muted small">
            {result.filename} · {result.analyzed_seconds}s analysed · detection score {result.detection_score.toFixed(2)}
          </p>
        </div>
      </section>

      {routing.conflict && (
        <div className="callout">
          Provenance and detection disagree. The trust-routing policy escalated this file instead of trusting either
          signal alone.
        </div>
      )}

      <Timeline result={result} />

      <section className="card">
        <h2>Provenance (C2PA)</h2>
        <dl className="kv">
          <dt>Status</dt>
          <dd>
            <span className={`pill ${provenance.status}`}>{PROV_TEXT[provenance.status] || provenance.status}</span>
          </dd>
          <dt>Verifier</dt>
          <dd>{provenance.verifier}</dd>
          {provenance.issuer && (
            <>
              <dt>Issuer</dt>
              <dd>{provenance.issuer}</dd>
            </>
          )}
          <dt>Notes</dt>
          <dd>{provenance.notes.join(" · ")}</dd>
        </dl>
      </section>

      {result.notes.length > 0 && (
        <section className="card">
          <h2>Analysis notes</h2>
          <ul className="reasons">
            {result.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        </section>
      )}

      <p className="disclaimer">{result.disclaimer}</p>
      <button className="btn" onClick={onReset}>
        Analyze another video
      </button>
    </>
  );
}

export default function App() {
  const [state, setState] = useState("idle"); // idle | scanning | done | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const inputRef = useRef(null);

  async function analyze(file) {
    if (!file) return;
    setState("scanning");
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const res = await fetch(`${API}/analyze`, { method: "POST", body });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Request failed (${res.status})`);
      }
      setResult(await res.json());
      setState("done");
    } catch (e) {
      setError(e.message === "Failed to fetch" ? `Cannot reach the API at ${API}. Is the backend running?` : e.message);
      setState("error");
    }
  }

  function reset() {
    setState("idle");
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <main>
      <header>
        <h1>Deepfake Forensics Provenance Engine</h1>
        <p className="muted">
          Detection, localization and C2PA provenance fused into one trust decision.
        </p>
      </header>

      {(state === "idle" || state === "error") && (
        <>
          <div
            className={`drop ${drag ? "drag" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              analyze(e.dataTransfer.files[0]);
            }}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
          >
            <div className="drop-icon">⬆</div>
            <b>Drop a video to analyze</b>
            <span className="muted">or click to browse · MP4, MOV, MKV, WEBM, AVI · up to 200 MB</span>
            <input
              ref={inputRef}
              type="file"
              accept="video/*,.mkv"
              hidden
              onChange={(e) => analyze(e.target.files[0])}
            />
          </div>
          {state === "error" && <div className="error">{error}</div>}
        </>
      )}

      {state === "scanning" && (
        <div className="drop scanning">
          <div className="scanbar" />
          <b>Analyzing video, audio and provenance…</b>
          <span className="muted">This can take a few seconds for longer clips.</span>
        </div>
      )}

      {state === "done" && result && <Results result={result} onReset={reset} />}
    </main>
  );
}
