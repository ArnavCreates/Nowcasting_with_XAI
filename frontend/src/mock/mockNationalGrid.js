import { mockAlerts } from './mockAlerts';

// Bounding box roughly covering mainland India
const INDIA_BOUNDS = { latMin: 6, latMax: 38, lngMin: 68, lngMax: 98 };
const GRID_RESOLUTION_DEG = 0.5; // ~32 x 30 grid, light enough for Deck.gl HeatmapLayer

// Deterministic pseudo-noise (no Math.random — keeps demo output stable across reloads)
function seededNoise(lat, lng) {
  const n = Math.sin(lat * 12.9898 + lng * 78.233) * 43758.5453;
  return n - Math.floor(n); // 0..1
}

// Haversine distance in km (self-contained here; geoUtils.js can re-export/reuse later)
function haversineKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// Build hotspots for a given hazard directly from mockAlerts — single source of truth
function getHotspots(hazardType) {
  return mockAlerts
    .filter((a) => a.hazardType === hazardType && a.status !== 'resolved')
    .map((a) => ({
      lat: a.lat,
      lng: a.lng,
      intensity: a.riskScore,
      // higher-risk regions bleed influence further outward
      sigmaKm: 60 + (a.riskScore / 100) * 140,
    }));
}

// Gaussian decay contribution from one hotspot at a given point
function gaussianContribution(distKm, intensity, sigmaKm) {
  return intensity * Math.exp(-(distKm ** 2) / (2 * sigmaKm ** 2));
}

function generateGridPoints() {
  const points = [];
  for (let lat = INDIA_BOUNDS.latMin; lat <= INDIA_BOUNDS.latMax; lat += GRID_RESOLUTION_DEG) {
    for (let lng = INDIA_BOUNDS.lngMin; lng <= INDIA_BOUNDS.lngMax; lng += GRID_RESOLUTION_DEG) {
      points.push({ lat, lng });
    }
  }
  return points;
}

function computeHazardGrid(hazardType) {
  const hotspots = getHotspots(hazardType);
  const points = generateGridPoints();

  return points.map(({ lat, lng }) => {
    let value = 0;
    for (const h of hotspots) {
      const dist = haversineKm(lat, lng, h.lat, h.lng);
      value += gaussianContribution(dist, h.intensity, h.sigmaKm);
    }
    // ambient embient baseline + tiny deterministic texture so flat zero-zones aren't perfectly uniform
    const ambient = seededNoise(lat, lng) * 1.5;
    const total = Math.min(100, value + ambient);

    return { lat, lng, value: Math.round(total * 10) / 10 };
  });
}

// Eagerly computed once at module load — cheap (~960 points x 3 hazards)
export const nationalHazardGrids = {
  thunderstorm: computeHazardGrid('thunderstorm'),
  cloudburst: computeHazardGrid('cloudburst'),
  flashFlood: computeHazardGrid('flashFlood'),
};

// Composite grid for the "All" tab — each point tagged with whichever hazard is dominant there
export const nationalCompositeGrid = (() => {
  const { thunderstorm, cloudburst, flashFlood } = nationalHazardGrids;
  return thunderstorm.map((_, i) => {
    const candidates = [
      { hazardType: 'thunderstorm', value: thunderstorm[i].value },
      { hazardType: 'cloudburst', value: cloudburst[i].value },
      { hazardType: 'flashFlood', value: flashFlood[i].value },
    ];
    const dominant = candidates.reduce((a, b) => (b.value > a.value ? b : a));
    return {
      lat: thunderstorm[i].lat,
      lng: thunderstorm[i].lng,
      value: dominant.value,
      hazardType: dominant.hazardType,
    };
  });
})();

// Grid metadata — needed later for contour generation (d3-contour expects a flat row-major array + dims)
export const gridMeta = {
  bounds: INDIA_BOUNDS,
  resolutionDeg: GRID_RESOLUTION_DEG,
  cols: Math.floor((INDIA_BOUNDS.lngMax - INDIA_BOUNDS.lngMin) / GRID_RESOLUTION_DEG) + 1,
  rows: Math.floor((INDIA_BOUNDS.latMax - INDIA_BOUNDS.latMin) / GRID_RESOLUTION_DEG) + 1,
};

// Flat value array per hazard, row-major, for contour libraries (d3-contour, Deck.gl ContourLayer)
export function getFlatValues(hazardType) {
  const grid =
    hazardType === 'all' ? nationalCompositeGrid : nationalHazardGrids[hazardType];
  return grid.map((p) => p.value);
}