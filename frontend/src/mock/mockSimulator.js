import { mockTelemetry } from './mockTelemetry';

// Grid steps are deltas FROM each region's current baseline (pulled from
// mockTelemetry.current), not absolute values — this keeps the grid generic
// while the UI/hook layer converts to/from absolute km/h and mm/h.
const deltaWindSteps = [-30, -15, 0, 15, 30];   // km/h change from baseline
const deltaPrecipSteps = [-40, -20, 0, 20, 40]; // mm/h change from baseline

// Per-region physical sensitivity profile.
// windSensitivity: risk points per km/h of wind speed change — higher for
// plains/coastal regions where convective/thunderstorm risk is wind-driven.
// precipSensitivity: risk points per mm/h of rainfall change — higher for
// Himalayan/hill regions where orographic lift amplifies flash flood risk.
const regionSensitivity = {
  'IN-MH-MUM': { windSensitivity: 0.6, precipSensitivity: 0.5 },
  'IN-HP-SOL': { windSensitivity: 0.3, precipSensitivity: 0.7 },
  'IN-HP-SHM': { windSensitivity: 0.3, precipSensitivity: 0.65 },
  'IN-UK-RUD': { windSensitivity: 0.3, precipSensitivity: 0.75 },
  'IN-AS-GUW': { windSensitivity: 0.9, precipSensitivity: 0.4 },
  'IN-WB-DAR': { windSensitivity: 0.35, precipSensitivity: 0.68 },
  'IN-TN-CHE': { windSensitivity: 0.5, precipSensitivity: 0.45 },
  'IN-MH-PUN': { windSensitivity: 0.5, precipSensitivity: 0.35 },
};

// Base risk per region — reuses each region's live risk score as the
// "current model output" the simulator perturbs away from.
const regionBaseRisk = {
  'IN-MH-MUM': 94,
  'IN-HP-SOL': 85,
  'IN-HP-SHM': 82,
  'IN-UK-RUD': 84,
  'IN-AS-GUW': 65,
  'IN-WB-DAR': 78,
  'IN-TN-CHE': 22,
  'IN-MH-PUN': 45,
};

function computeRisk(baseRisk, deltaWind, deltaPrecip, windSensitivity, precipSensitivity) {
  const score = baseRisk + deltaWind * windSensitivity + deltaPrecip * precipSensitivity;
  return Math.min(100, Math.max(0, Math.round(score)));
}

function buildGrid(regionId) {
  const { windSensitivity, precipSensitivity } = regionSensitivity[regionId];
  const baseRisk = regionBaseRisk[regionId];

  const grid = [];
  for (const deltaWind of deltaWindSteps) {
    for (const deltaPrecip of deltaPrecipSteps) {
      grid.push({
        deltaWind,
        deltaPrecip,
        riskScore: computeRisk(baseRisk, deltaWind, deltaPrecip, windSensitivity, precipSensitivity),
      });
    }
  }
  return grid;
}

// Baseline absolute values per region, sourced directly from mockTelemetry
// so the Simulator's starting point matches what Live Map already shows.
function getBaseline(regionId) {
  const telemetry = mockTelemetry[regionId]?.current;
  return {
    windSpeed: telemetry?.windGust ?? 30,
    precipRate: telemetry?.rainfallRate ?? 40,
  };
}

export const mockSimulator = {
  deltaWindSteps,
  deltaPrecipSteps,

  // Absolute slider bounds shown in the UI — generous enough to cover
  // realistic extremes above/below any region's baseline.
  windSpeedRange: { min: 0, max: 150, step: 5 },
  precipRange: { min: 0, max: 200, step: 5 },

  // Keyed by regionId — each region has its own baseline, sensitivity
  // profile, and precomputed 5x5 (wind x precip) risk grid.
  regions: Object.fromEntries(
    Object.keys(regionSensitivity).map((regionId) => [
      regionId,
      {
        baseRisk: regionBaseRisk[regionId],
        ...regionSensitivity[regionId],
        baseline: getBaseline(regionId),
        grid: buildGrid(regionId),
      },
    ])
  ),

  // Quick-select scenarios — expressed as absolute deltas applied on top
  // of whatever region is currently selected.
  presets: [
    {
      id: 'severe-storm',
      label: 'Severe Storm Surge',
      description: 'Sharp increase in both wind speed and rainfall intensity — worst-case convective event.',
      deltaWind: 30,
      deltaPrecip: 40,
    },
    {
      id: 'cloudburst-spike',
      label: 'Cloudburst Spike',
      description: 'Rainfall intensity surges with only modest wind change — typical cloudburst signature.',
      deltaWind: 5,
      deltaPrecip: 40,
    },
    {
      id: 'high-wind-dry',
      label: 'High Wind, Low Rain',
      description: 'Strong gusty winds with little precipitation — dry thunderstorm / squall-line pattern.',
      deltaWind: 30,
      deltaPrecip: -20,
    },
    {
      id: 'calm-baseline',
      label: 'Calm Conditions',
      description: 'Reduced wind and rainfall — a subsiding, lower-risk scenario.',
      deltaWind: -20,
      deltaPrecip: -30,
    },
  ],
};