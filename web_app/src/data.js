export async function loadAppData() {
  const [cases, sweep, validation, manifest] = await Promise.all([
    fetchJson("data/fea_cases.json"),
    fetchJson("data/prb_sweep_phi90.json"),
    fetchJson("data/validation_summary.json"),
    fetchJson("data/app_manifest.json"),
  ]);

  return { cases, sweep, validation, manifest };
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Could not load ${path}`);
  }
  return response.json();
}
