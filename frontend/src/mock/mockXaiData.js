// ---- Deterministic helpers (no Math.random — stable across demo reloads) ----
function seededNoise(lat, lng, salt = 0) {
  const n = Math.sin(lat * 12.9898 + lng * 78.233 + salt * 37.719) * 43758.5453;
  return n - Math.floor(n); // 0..1
}

// Generates a small local grid of attention points around a region center,
// weight decaying outward from the peak — feeds Deck.gl HeatmapLayer directly.
function generateAttentionGrid(centerLat, centerLng, peakWeight = 0.95) {
  const points = [];
  const steps = [-0.2, -0.1, 0, 0.1, 0.2]; // ~±22km spread at this latitude scale
  steps.forEach((dLat) => {
    steps.forEach((dLng) => {
      const lat = centerLat + dLat;
      const lng = centerLng + dLng;
      const dist = Math.sqrt(dLat ** 2 + dLng ** 2);
      const falloff = Math.exp(-(dist ** 2) / (2 * 0.12 ** 2));
      const jitter = 0.9 + seededNoise(lat, lng) * 0.2; // slight organic variation
      const attentionWeight = Math.min(1, Math.round(peakWeight * falloff * jitter * 100) / 100);
      if (attentionWeight > 0.05) {
        points.push({ position: [lng, lat], attentionWeight });
      }
    });
  });
  return points;
}

// Generates a 15-minute, 4-frame thermal IR cooling sequence.
function generateIrStrip(regionId, startCtt, totalDelta) {
  const offsets = [-15, -10, -5, 0]; // minutes relative to now
  return offsets.map((minuteOffset, i) => {
    const progress = i / (offsets.length - 1);
    const cttValue = Math.round((startCtt + totalDelta * progress) * 10) / 10;
    return {
      minuteOffset,
      cttValue,
      imagePath: `/satellite-rasters/${regionId}_ir_t${minuteOffset}.png`,
    };
  });
}

export const mockXaiData = {
  'IN-MH-MUM': {
    predictionId: 'XAI-001',
    hazardType: 'flashFlood',
    confidence: 94,
    generatedAt: '2026-08-28T10:42:00Z',
    plainLanguageSummary:
      'Extreme moisture convergence over Mumbai Metro combined with high drainage density strain suggests severe flash flood risk within 2 hours. Urban runoff capacity is likely to be exceeded.',
    featureContribution: [
      { feature: 'IWV Spike', weight: 44 },
      { feature: 'CTT Drop', weight: 27 },
      { feature: 'CAPE', weight: 14 },
      { feature: 'Terrain Slope', weight: 9 },
      { feature: 'Drainage Dens.', weight: 6 },
    ],
    whatChanged: {
      iwv: { value: 62, delta: 14, unit: 'mm' },
      ctt: { value: -65, delta: -6, unit: '°C' },
      riskScore: { value: 94, delta: 16 },
    },
    calibration: [
      { predictedConfidence: 20, historicalAccuracy: 22 },
      { predictedConfidence: 45, historicalAccuracy: 48 },
      { predictedConfidence: 65, historicalAccuracy: 70 },
      { predictedConfidence: 94, historicalAccuracy: 88 },
    ],
    attentionZone: { centerLat: 19.076, centerLng: 72.8777, radiusKm: 30 },
    attentionGrid: generateAttentionGrid(19.076, 72.8777, 0.97),
    satelliteIrStrip: generateIrStrip('IN-MH-MUM', -50, -15),
  },

  'IN-HP-SOL': {
    predictionId: 'XAI-003',
    hazardType: 'flashFlood',
    confidence: 89,
    generatedAt: '2026-08-28T14:02:45Z',
    plainLanguageSummary:
      'High moisture convergence and orographic lift over Solan District point to imminent flash flood risk near the Giri River basin within 2 hours.',
    featureContribution: [
      { feature: 'IWV Spike', weight: 40 },
      { feature: 'CTT Drop', weight: 25 },
      { feature: 'CAPE', weight: 18 },
      { feature: 'Terrain Slope', weight: 12 },
      { feature: 'Drainage Dens.', weight: 5 },
    ],
    whatChanged: {
      iwv: { value: 54, delta: 9, unit: 'mm' },
      ctt: { value: -65, delta: -5, unit: '°C' },
      riskScore: { value: 89, delta: 12 },
    },
    calibration: [
      { predictedConfidence: 30, historicalAccuracy: 33 },
      { predictedConfidence: 55, historicalAccuracy: 58 },
      { predictedConfidence: 70, historicalAccuracy: 66 },
      { predictedConfidence: 89, historicalAccuracy: 85 },
    ],
    attentionZone: { centerLat: 30.907, centerLng: 77.0999, radiusKm: 22 },
    attentionGrid: generateAttentionGrid(30.907, 77.0999, 0.91),
    satelliteIrStrip: generateIrStrip('IN-HP-SOL', -55, -10),
  },

  'IN-HP-SHM': {
    predictionId: 'XAI-005',
    hazardType: 'cloudburst',
    confidence: 82,
    generatedAt: '2026-08-28T10:40:00Z',
    plainLanguageSummary:
      'Rapid cloud development observed over Shimla. Rising CAPE and steepening cloud-top cooling suggest cloudburst formation is likely within 2-3 hours.',
    featureContribution: [
      { feature: 'IWV Spike', weight: 38 },
      { feature: 'CTT Drop', weight: 29 },
      { feature: 'CAPE', weight: 17 },
      { feature: 'Terrain Slope', weight: 11 },
      { feature: 'Drainage Dens.', weight: 5 },
    ],
    whatChanged: {
      iwv: { value: 49, delta: 8, unit: 'mm' },
      ctt: { value: -58, delta: -5, unit: '°C' },
      riskScore: { value: 82, delta: 11 },
    },
    calibration: [
      { predictedConfidence: 26, historicalAccuracy: 29 },
      { predictedConfidence: 50, historicalAccuracy: 53 },
      { predictedConfidence: 82, historicalAccuracy: 77 },
    ],
    attentionZone: { centerLat: 31.1048, centerLng: 77.1734, radiusKm: 20 },
    attentionGrid: generateAttentionGrid(31.1048, 77.1734, 0.88),
    satelliteIrStrip: generateIrStrip('IN-HP-SHM', -48, -10),
  },

  'IN-UK-RUD': {
    predictionId: 'XAI-002',
    hazardType: 'cloudburst',
    confidence: 84,
    generatedAt: '2026-08-28T14:20:00Z',
    plainLanguageSummary:
      'Rapid cloud development and steep terrain slope suggest high cloudburst risk in the next 2 hours. The interaction between intense convective available potential energy (CAPE) and local orographic lift is creating highly unstable conditions.',
    featureContribution: [
      { feature: 'IWV Spike', weight: 42 },
      { feature: 'CTT Drop', weight: 30 },
      { feature: 'CAPE', weight: 15 },
      { feature: 'Terrain Slope', weight: 10 },
      { feature: 'Drainage Dens.', weight: 3 },
    ],
    whatChanged: {
      iwv: { value: 42, delta: 12, unit: 'mm' },
      ctt: { value: -45, delta: -8, unit: '°C' },
      riskScore: { value: 84, delta: 22 },
    },
    calibration: [
      { predictedConfidence: 25, historicalAccuracy: 30 },
      { predictedConfidence: 50, historicalAccuracy: 52 },
      { predictedConfidence: 74, historicalAccuracy: 68 },
      { predictedConfidence: 84, historicalAccuracy: 80 },
    ],
    attentionZone: { centerLat: 30.2849, centerLng: 78.9814, radiusKm: 25 },
    attentionGrid: generateAttentionGrid(30.2849, 78.9814, 0.93),
    satelliteIrStrip: generateIrStrip('IN-UK-RUD', -37, -8),
  },

  'IN-AS-GUW': {
    predictionId: 'XAI-006',
    hazardType: 'thunderstorm',
    confidence: 65,
    generatedAt: '2026-08-28T10:35:00Z',
    plainLanguageSummary:
      'Rising CAPE with eroding Convective Inhibition over Guwahati indicates storm cell formation is likely within the next 3 hours, though moisture levels remain moderate.',
    featureContribution: [
      { feature: 'CAPE', weight: 36 },
      { feature: 'IWV Spike', weight: 24 },
      { feature: 'CTT Drop', weight: 22 },
      { feature: 'Terrain Slope', weight: 12 },
      { feature: 'Drainage Dens.', weight: 6 },
    ],
    whatChanged: {
      iwv: { value: 38, delta: 5, unit: 'mm' },
      ctt: { value: -40, delta: -4, unit: '°C' },
      riskScore: { value: 65, delta: 9 },
    },
    calibration: [
      { predictedConfidence: 20, historicalAccuracy: 24 },
      { predictedConfidence: 45, historicalAccuracy: 47 },
      { predictedConfidence: 65, historicalAccuracy: 61 },
    ],
    attentionZone: { centerLat: 26.1445, centerLng: 91.7362, radiusKm: 28 },
    attentionGrid: generateAttentionGrid(26.1445, 91.7362, 0.7),
    satelliteIrStrip: generateIrStrip('IN-AS-GUW', -32, -8),
  },

  'IN-WB-DAR': {
    predictionId: 'XAI-004',
    hazardType: 'flashFlood',
    confidence: 78,
    generatedAt: '2026-08-28T09:35:00Z',
    plainLanguageSummary:
      'Steep terrain slope in Darjeeling combined with sustained moisture accumulation is increasing flash flood likelihood over the next 3-4 hours.',
    featureContribution: [
      { feature: 'IWV Spike', weight: 35 },
      { feature: 'Terrain Slope', weight: 28 },
      { feature: 'CTT Drop', weight: 20 },
      { feature: 'CAPE', weight: 12 },
      { feature: 'Drainage Dens.', weight: 5 },
    ],
    whatChanged: {
      iwv: { value: 46, delta: 7, unit: 'mm' },
      ctt: { value: -48, delta: -4, unit: '°C' },
      riskScore: { value: 78, delta: 10 },
    },
    calibration: [
      { predictedConfidence: 28, historicalAccuracy: 31 },
      { predictedConfidence: 52, historicalAccuracy: 55 },
      { predictedConfidence: 78, historicalAccuracy: 74 },
    ],
    attentionZone: { centerLat: 27.041, centerLng: 88.2663, radiusKm: 18 },
    attentionGrid: generateAttentionGrid(27.041, 88.2663, 0.84),
    satelliteIrStrip: generateIrStrip('IN-WB-DAR', -40, -8),
  },

  'IN-TN-CHE': {
    predictionId: 'XAI-007',
    hazardType: 'flashFlood',
    confidence: 22,
    generatedAt: '2026-08-28T10:15:00Z',
    plainLanguageSummary:
      'Localized drainage strain is possible in low-lying areas of Chennai, but overall atmospheric instability and moisture levels remain low, keeping risk minimal.',
    featureContribution: [
      { feature: 'Drainage Dens.', weight: 34 },
      { feature: 'IWV Spike', weight: 26 },
      { feature: 'Terrain Slope', weight: 20 },
      { feature: 'CTT Drop', weight: 12 },
      { feature: 'CAPE', weight: 8 },
    ],
    whatChanged: {
      iwv: { value: 30, delta: 2, unit: 'mm' },
      ctt: { value: -22, delta: -1, unit: '°C' },
      riskScore: { value: 22, delta: 3 },
    },
    calibration: [
      { predictedConfidence: 15, historicalAccuracy: 18 },
      { predictedConfidence: 22, historicalAccuracy: 20 },
    ],
    attentionZone: { centerLat: 13.0827, centerLng: 80.2707, radiusKm: 15 },
    attentionGrid: generateAttentionGrid(13.0827, 80.2707, 0.3),
    satelliteIrStrip: generateIrStrip('IN-TN-CHE', -20, -2),
  },

  'IN-MH-PUN': {
    predictionId: 'XAI-008',
    hazardType: 'thunderstorm',
    confidence: 45,
    generatedAt: '2026-08-28T09:00:00Z',
    plainLanguageSummary:
      'A previously developing storm cell over Pune Outskirts has dissipated before reaching populated zones. Residual instability is low and no further action is required.',
    featureContribution: [
      { feature: 'CAPE', weight: 30 },
      { feature: 'IWV Spike', weight: 25 },
      { feature: 'CTT Drop', weight: 20 },
      { feature: 'Terrain Slope', weight: 15 },
      { feature: 'Drainage Dens.', weight: 10 },
    ],
    whatChanged: {
      iwv: { value: 27, delta: -3, unit: 'mm' },
      ctt: { value: -15, delta: 5, unit: '°C' },
      riskScore: { value: 45, delta: -18 },
    },
    calibration: [
      { predictedConfidence: 20, historicalAccuracy: 23 },
      { predictedConfidence: 45, historicalAccuracy: 40 },
    ],
    attentionZone: { centerLat: 18.5204, centerLng: 73.8567, radiusKm: 12 },
    attentionGrid: generateAttentionGrid(18.5204, 73.8567, 0.4),
    satelliteIrStrip: generateIrStrip('IN-MH-PUN', -12, 5), // warming = dissipating storm
  },
};