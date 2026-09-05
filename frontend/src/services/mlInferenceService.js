const API_BASE_URL = import.meta.env.VITE_ML_MODEL_API_URL || 'http://localhost:8000';
const USE_REAL_MODEL = import.meta.env.VITE_USE_REAL_MODEL === 'true';

// Per-region physical profile — mirrors mockThresholdConfig.js's calibration
// logic (Himalayan/hill regions more precip/terrain-sensitive; coastal/plains
// regions more wind/drainage-sensitive). Kept separate from the threshold
// evaluator since this drives simulated *scores*, not pass/fail rules.
const regionProfiles = {
  'IN-MH-MUM': { terrainFactor: 0.9, drainageFactor: 1.3, baseHazard: 'flashFlood' },
  'IN-HP-SOL': { terrainFactor: 1.4, drainageFactor: 0.8, baseHazard: 'flashFlood' },
  'IN-HP-SHM': { terrainFactor: 1.3, drainageFactor: 0.8, baseHazard: 'cloudburst' },
  'IN-UK-RUD': { terrainFactor: 1.5, drainageFactor: 0.7, baseHazard: 'cloudburst' },
  'IN-AS-GUW': { terrainFactor: 0.8, drainageFactor: 1.0, baseHazard: 'thunderstorm' },
  'IN-WB-DAR': { terrainFactor: 1.35, drainageFactor: 0.85, baseHazard: 'cloudburst' },
  'IN-TN-CHE': { terrainFactor: 0.7, drainageFactor: 1.2, baseHazard: 'flashFlood' },
  'IN-MH-PUN': { terrainFactor: 0.75, drainageFactor: 0.9, baseHazard: 'thunderstorm' },
};

function getProfile(regionId) {
  return regionProfiles[regionId] || { terrainFactor: 1.0, drainageFactor: 1.0, baseHazard: 'thunderstorm' };
}

/**
 * Executes severe weather ML model inference for a region.
 *
 * payload: { regionId, regionName, lat, lng, timestamp,
 *            telemetry: { precipitation, windSpeed, iwv, cape, cin, cttDrop } }
 *
 * When VITE_USE_REAL_MODEL=true, this posts the exact same payload shape to
 * a real backend at VITE_ML_MODEL_API_URL — no caller changes needed to swap.
 */
export const predictSevereWeather = async (payload = {}) => {
  if (!USE_REAL_MODEL) {
    await new Promise((resolve) => setTimeout(resolve, 250));

    const { regionId = 'IN-HP-SHM', regionName = 'Shimla', telemetry = {} } = payload;
    const {
      precipitation = 65,
      windSpeed = 40,
      iwv = 45,
      cape = 1800,
      cin = -15,
      cttDrop = 8,
    } = telemetry;

    const { terrainFactor, drainageFactor, baseHazard } = getProfile(regionId);

    // Moisture + instability + kinematics, weighted per the problem
    // statement's three predictive ingredients, then terrain/drainage
    // scaled for the flash-flood translation step.
    const moistureScore = iwv * 0.5 + precipitation * 0.3;
    const instabilityScore = Math.max(0, cape / 40 - Math.abs(cin) * 0.3);
    const kinematicsScore = windSpeed * 0.3 + cttDrop * 1.5;

    const rawRisk = Math.min(
      99,
      Math.max(5, Math.round((moistureScore + instabilityScore + kinematicsScore) * 0.55 * terrainFactor))
    );

    const hourlyTrend = [1,2, 3, 4, 5, 6].map((hour) => {
      const curve = [0.9, 1, 1.1, 1.15, 1.0, 0.85][hour - 1];
      return { hour, risk: Math.min(99, Math.max(5, Math.round(rawRisk * curve))) };
    });

    const featureImportance = [
      { feature: 'Integrated Water Vapor (IWV)', impact: Math.round(iwv * 0.4) },
      { feature: 'CAPE (Instability)', impact: Math.round(cape / 45) },
      { feature: 'Precipitation Rate', impact: Math.round(precipitation * 0.35) },
      { feature: 'CTT Drop Rate', impact: Math.round(cttDrop * 1.6) },
      { feature: 'Wind Shear Speed', impact: Math.round(windSpeed * 0.25) },
      { feature: 'Terrain Slope', impact: Math.round(terrainFactor * 15) },
      { feature: 'Drainage Density', impact: Math.round(drainageFactor * -10) },
    ].sort((a, b) => b.impact - a.impact);

    // 5x5 attention grid, weighted toward center, deterministic-ish spread
    const baseWeight = rawRisk / 100;
    const attentionGrid = Array.from({ length: 5 }, (_, r) =>
      Array.from({ length: 5 }, (_, c) => {
        const distFromCenter = Math.abs(r - 2) + Math.abs(c - 2);
        const val = baseWeight * (1 - distFromCenter * 0.18);
        return parseFloat(Math.min(1, Math.max(0.05, val)).toFixed(2));
      })
    );

    return {
      regionId,
      regionName,
      timestamp: payload.timestamp || new Date().toISOString(),
      riskScore: rawRisk,
      hazardType: baseHazard,
      confidence: 0.86 + terrainFactor * 0.03,
      hourlyTrend,
      metrics: {
        precipitationRate: `${precipitation} mm/h`,
        windSpeed: `${windSpeed} km/h`,
        iwvMoisture: `${iwv} kg/m²`,
        cape: `${cape} J/kg`,
        cin: `${cin} J/kg`,
        cttDropRate: `-${cttDrop}°C/30min`,
      },
      featureImportance,
      attentionGrid,
      xaiExplanation: `${regionName}'s 0–6h nowcast projects peak ${baseHazard} intensity between +2h and +3h, driven primarily by moisture convergence (IWV ${iwv} kg/m²) and instability (CAPE ${cape} J/kg), with terrain slope amplifying downstream flood translation.`,
    };
  }

  // ---- Real backend swap point ----
  // Same payload shape posted as-is; response is expected to match the
  // mock shape above (riskScore, hazardType, confidence, hourlyTrend[],
  // metrics{}, featureImportance[], attentionGrid, xaiExplanation).
  const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Inference HTTP Error ${response.status}: ${response.statusText}`);
  }

  return await response.json();
};

export default { predictSevereWeather };