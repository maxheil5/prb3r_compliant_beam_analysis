import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { PRESETS } from "../src/presets.js";
import { jointPointsFromTheta, nearestSweepRow } from "../src/geometry.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dataDir = path.join(root, "public", "data");

const cases = readJson("fea_cases.json");
const sweep = readJson("prb_sweep_phi90.json");
const summary = readJson("validation_summary.json");

assert(cases.length === 10, `Expected 10 FEA cases, found ${cases.length}.`);
assert(summary.length >= 6, "Expected validation summary records.");

for (const preset of PRESETS) {
  assert(cases.some((item) => Number(item.case_id) === preset.caseId), `Preset ${preset.id} has no matching FEA case.`);
}

for (const item of cases) {
  for (const key of ["Qx_ref", "Qy_ref", "Qx_fea", "Qy_fea", "tip_error_percent", "max_von_mises_mpa"]) {
    assert(Number.isFinite(Number(item[key])), `Case ${item.case_id} has invalid ${key}.`);
  }
  const row = nearestSweepRow(sweep, item.kappa, item.theta0_target_rad);
  assert(row, `No sweep row found for case ${item.case_id}.`);
}

const points = jointPointsFromTheta([0.15, 0.1, 0.05]);
assert(points.length === 5, "PRB joint point count should be 5.");
for (const point of points) {
  assert(Number.isFinite(point.x) && Number.isFinite(point.y), "PRB joint point contains non-finite values.");
}

console.log("Web app data checks passed.");

function readJson(filename) {
  return JSON.parse(fs.readFileSync(path.join(dataDir, filename), "utf8"));
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}
