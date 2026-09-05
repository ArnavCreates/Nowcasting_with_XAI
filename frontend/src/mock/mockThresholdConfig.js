// Static per-region threshold limits. Calibrated for microclimates: steep
// Himalayan regions trigger on lower moisture/rain accumulation than flat
// coastal or plateau regions. Pure data only — no styling, no evaluation logic.
// See utils/thresholdEvaluator.js for how these get compared against live telemetry.

export const regionalThresholdConfigs = {
  'IN-MH-MUM': {
    regionName: 'Mumbai Metro',
    flashFlood: { iwv: 55, cttDrop: 8 },
    cloudburst: { rainfallRate: 100, area: 20 },
    highWind: { speed: 65 },
  },
  'IN-HP-SOL': {
    regionName: 'Solan District',
    flashFlood: { iwv: 38, cttDrop: 10 },
    cloudburst: { rainfallRate: 75, area: 10 },
    highWind: { speed: 52 },
  },
  'IN-HP-SHM': {
    regionName: 'Shimla',
    flashFlood: { iwv: 40, cttDrop: 9 },
    cloudburst: { rainfallRate: 80, area: 12 },
    highWind: { speed: 55 },
  },
  'IN-UK-RUD': {
    regionName: 'Rudraprayag',
    flashFlood: { iwv: 36, cttDrop: 10 },
    cloudburst: { rainfallRate: 70, area: 10 },
    highWind: { speed: 50 },
  },
  'IN-AS-GUW': {
    regionName: 'Guwahati',
    flashFlood: { iwv: 50, cttDrop: 7 },
    cloudburst: { rainfallRate: 90, area: 15 },
    highWind: { speed: 60 },
  },
  'IN-WB-DAR': {
    regionName: 'Darjeeling',
    flashFlood: { iwv: 42, cttDrop: 9 },
    cloudburst: { rainfallRate: 80, area: 12 },
    highWind: { speed: 55 },
  },
  'IN-TN-CHE': {
    regionName: 'Chennai',
    flashFlood: { iwv: 54, cttDrop: 7 },
    cloudburst: { rainfallRate: 95, area: 20 },
    highWind: { speed: 65 },
  },
  'IN-MH-PUN': {
    regionName: 'Pune Outskirts',
    flashFlood: { iwv: 48, cttDrop: 8 },
    cloudburst: { rainfallRate: 85, area: 18 },
    highWind: { speed: 60 },
  },
};

// Rule metadata shared across all regions — label/icon/tier don't vary by region,
// only the numeric limits above do.
export const thresholdRuleMeta = {
  'flash-flood-critical': { label: 'Flash Flood Critical', tier: 'critical', icon: 'droplet' },
  'cloudburst-warning': { label: 'Cloudburst Warning', tier: 'warning', icon: 'cloud-lightning' },
  'high-wind-watch': { label: 'High Wind Watch', tier: 'info', icon: 'wind' },
};