import { useEffect, useMemo, useState } from "react";

import { loadAppData } from "./data.js";
import {
  beamCurve,
  formatNumber,
  interpretation,
  mirroredPath,
  nearestSweepRow,
  prbMirrorPoints,
  scaledCase,
} from "./geometry.js";
import { PRESETS } from "./presets.js";
import ThreePreview from "./ThreePreview.jsx";

export default function App() {
  const [appData, setAppData] = useState(null);
  const [error, setError] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState(6);
  const [progress, setProgress] = useState(1);
  const [isPlaying, setIsPlaying] = useState(false);
  const [showFea, setShowFea] = useState(true);
  const [showPrb, setShowPrb] = useState(true);
  const [showCurve, setShowCurve] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1);

  useEffect(() => {
    loadAppData().then(setAppData).catch((loadError) => setError(loadError.message));
  }, []);

  useEffect(() => {
    if (!isPlaying) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      setProgress((current) => {
        const next = current + 0.008 * animationSpeed;
        return next > 1 ? 0 : next;
      });
    }, 28);
    return () => window.clearInterval(timer);
  }, [isPlaying, animationSpeed]);

  const selectedCase = useMemo(() => {
    if (!appData?.cases?.length) {
      return null;
    }
    return appData.cases.find((item) => Number(item.case_id) === Number(selectedCaseId)) ?? appData.cases[0];
  }, [appData, selectedCaseId]);

  const sweepRow = useMemo(() => {
    if (!appData?.sweep?.length || !selectedCase) {
      return null;
    }
    return nearestSweepRow(appData.sweep, selectedCase.kappa, selectedCase.theta0_target_rad);
  }, [appData, selectedCase]);

  const displayCase = selectedCase ? scaledCase(selectedCase, progress) : null;

  if (error) {
    return <main className="app-shell"><p className="error">Could not load app data: {error}</p></main>;
  }

  if (!appData || !displayCase) {
    return <main className="app-shell"><p className="loading">Loading compliant gripper validation data...</p></main>;
  }

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">ME7751 compliant mechanisms project</p>
          <h1>PRB 3R compliant gripper validation</h1>
          <p className="hero-copy">
            A mirrored flexure gripper built from the same large-deflection cantilever model used in the paper.
            Compare the analytical prediction against the completed SolidWorks FEA validation cases.
          </p>
        </div>
        <div className="hero-metrics">
          <Metric label="Tip error" value={`${formatNumber(selectedCase.tip_error_percent, 2)}%`} />
          <Metric label="Safety factor" value={formatNumber(selectedCase.stress_safety_factor, 2)} />
          <Metric label="Max stress" value={`${formatNumber(selectedCase.max_von_mises_mpa, 2)} MPa`} />
        </div>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <h2>Controls</h2>
          <label className="field-label" htmlFor="case-select">Validation case</label>
          <select id="case-select" value={selectedCaseId} onChange={(event) => setSelectedCaseId(Number(event.target.value))}>
            {appData.cases.map((item) => (
              <option key={item.case_id} value={item.case_id}>
                Case {item.case_id}: kappa {item.kappa}, theta {formatNumber(item.theta0_target_rad, 1)}
              </option>
            ))}
          </select>

          <div className="preset-grid">
            {PRESETS.map((preset) => (
              <button
                key={preset.id}
                className={Number(selectedCaseId) === preset.caseId ? "preset active" : "preset"}
                onClick={() => {
                  setSelectedCaseId(preset.caseId);
                  setProgress(1);
                }}
              >
                <span>{preset.label}</span>
                <small>{preset.description}</small>
              </button>
            ))}
          </div>

          <div className="play-row">
            <button className="primary-button" onClick={() => setIsPlaying((value) => !value)}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <input
              aria-label="Deformation"
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={progress}
              onChange={(event) => {
                setIsPlaying(false);
                setProgress(Number(event.target.value));
              }}
            />
          </div>

          <label className="field-label slider-label" htmlFor="speed-slider">
            Animation speed <span>{formatNumber(animationSpeed, 1)}x</span>
          </label>
          <input
            id="speed-slider"
            aria-label="Animation speed"
            type="range"
            min="0.25"
            max="3"
            step="0.25"
            value={animationSpeed}
            onChange={(event) => setAnimationSpeed(Number(event.target.value))}
          />

          <div className="toggle-list">
            <label><input type="checkbox" checked={showCurve} onChange={(event) => setShowCurve(event.target.checked)} /> continuous beam</label>
            <label><input type="checkbox" checked={showPrb} onChange={(event) => setShowPrb(event.target.checked)} /> PRB 3R skeleton</label>
            <label><input type="checkbox" checked={showFea} onChange={(event) => setShowFea(event.target.checked)} /> SolidWorks FEA marker</label>
          </div>

          <div className="case-note">
            <strong>Case {selectedCase.case_id}</strong>
            <p>{selectedCase.load_description}</p>
            <p>{interpretation(selectedCase)}</p>
          </div>
        </aside>

        <section className="visual-panel">
          <GripperSvg
            caseRow={displayCase}
            sweepRow={sweepRow}
            progress={progress}
            showFea={showFea}
            showPrb={showPrb}
            showCurve={showCurve}
          />
        </section>

        <aside className="insight-panel">
          <h2>Validation</h2>
          <Metric label="Analytical tip" value={`(${formatNumber(selectedCase.Qx_ref, 3)}, ${formatNumber(selectedCase.Qy_ref, 3)})`} />
          <Metric label="FEA tip" value={`(${formatNumber(selectedCase.Qx_fea, 3)}, ${formatNumber(selectedCase.Qy_fea, 3)})`} />
          <Metric label="Tip gap" value={`${formatNumber(selectedCase.tip_error_mm, 2)} mm`} />
          <Metric label="Stress margin" value={`${formatNumber(selectedCase.stress_margin_MPa, 2)} MPa`} />

          <div className="preview-card">
            <h3>3D flexure preview</h3>
            <ThreePreview caseRow={displayCase} />
          </div>

          <div className="small-copy">
            FEA values shown here are the completed SolidWorks validation cases. The animation interpolates from the
            undeformed shape to the selected validation state.
          </div>
        </aside>
      </section>
    </main>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function GripperSvg({ caseRow, sweepRow, progress, showFea, showPrb, showCurve }) {
  const refCurve = beamCurve(caseRow.Qx_ref_display, caseRow.Qy_ref_display);
  const feaCurve = beamCurve(caseRow.Qx_fea_display, caseRow.Qy_fea_display);
  const upperRef = mirroredPath(refCurve, "upper");
  const lowerRef = mirroredPath(refCurve, "lower");
  const upperFea = mirroredPath(feaCurve, "upper");
  const lowerFea = mirroredPath(feaCurve, "lower");
  const upperPrb = prbMirrorPoints(sweepRow, progress, "upper");
  const lowerPrb = prbMirrorPoints(sweepRow, progress, "lower");

  const upperRefTip = upperRef.at(-1);
  const lowerRefTip = lowerRef.at(-1);
  const upperFeaTip = upperFea.at(-1);
  const lowerFeaTip = lowerFea.at(-1);

  return (
    <svg className="gripper-svg" viewBox="0 0 1000 640" role="img" aria-label="Mirrored compliant gripper validation visual">
      <defs>
        <linearGradient id="beamGradient" x1="0%" x2="100%">
          <stop offset="0%" stopColor="#3f7f75" />
          <stop offset="100%" stopColor="#60b5a6" />
        </linearGradient>
      </defs>

      <rect x="0" y="0" width="1000" height="640" rx="22" className="scene-bg" />
      <line x1="110" y1="320" x2="850" y2="320" className="centerline" />
      <rect x="78" y="44" width="42" height="552" rx="6" className="fixture" />
      <text x="78" y="30" className="svg-label">fixed base</text>
      <text x="740" y="112" className="svg-label">tip discrepancy</text>

      {showCurve && (
        <>
          <path d={pathFromPoints(upperRef)} className="beam-path reference" />
          <path d={pathFromPoints(lowerRef)} className="beam-path reference" />
        </>
      )}
      {showFea && (
        <>
          <path d={pathFromPoints(upperFea)} className="beam-path fea" />
          <path d={pathFromPoints(lowerFea)} className="beam-path fea" />
          <line {...lineProps(upperRefTip, upperFeaTip)} className="tip-gap" />
          <line {...lineProps(lowerRefTip, lowerFeaTip)} className="tip-gap" />
          <circle {...circleProps(upperFeaTip)} r="8" className="fea-tip" />
          <circle {...circleProps(lowerFeaTip)} r="8" className="fea-tip" />
        </>
      )}
      {showPrb && (
        <>
          <polyline points={polylineFromPoints(upperPrb)} className="prb-link" />
          <polyline points={polylineFromPoints(lowerPrb)} className="prb-link" />
          {[...upperPrb, ...lowerPrb].map((point, index) => (
            <circle key={`${point.x}-${point.y}-${index}`} {...circleProps(point)} r="5" className="prb-joint" />
          ))}
        </>
      )}

      <circle {...circleProps(upperRefTip)} r="9" className="ref-tip" />
      <circle {...circleProps(lowerRefTip)} r="9" className="ref-tip" />
      <text x="720" y="530" className="legend-text">solid: analytical reference</text>
      <text x="720" y="554" className="legend-text">dashed: SolidWorks FEA</text>
      <text x="720" y="578" className="legend-text">nodes: PRB 3R chain</text>
    </svg>
  );
}

function pathFromPoints(points) {
  if (!points.length) {
    return "";
  }
  const [first, ...rest] = points.map(toScreen);
  return `M ${first.x} ${first.y} ${rest.map((point) => `L ${point.x} ${point.y}`).join(" ")}`;
}

function polylineFromPoints(points) {
  return points.map(toScreen).map((point) => `${point.x},${point.y}`).join(" ");
}

function circleProps(point) {
  const screen = toScreen(point);
  return { cx: screen.x, cy: screen.y };
}

function lineProps(start, end) {
  const a = toScreen(start);
  const b = toScreen(end);
  return { x1: a.x, y1: a.y, x2: b.x, y2: b.y };
}

function toScreen(point) {
  return {
    x: 110 + Number(point.x) * 690,
    y: 320 - Number(point.y) * 310,
  };
}
