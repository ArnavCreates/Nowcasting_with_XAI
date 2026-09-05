// Static satellite input sequence — INSAT-3DR High-resolution Enhanced
// Microwave (HEM) rainfall rate imagery, sourced via MOSDAC, matching the
// problem statement's stated satellite observation input (Water Vapor /
// Thermal IR channels feeding IWV and rainfall-rate estimation).
//
// This is a shared reference sequence (not per-region generated) since it's
// real archival imagery demonstrating the actual satellite product the
// model ingests — a real backend would instead serve live, region-specific
// frames through the same shape below.

export const mockSatelliteFrames = [
  {
    id: 'frame-1',
    timestamp: '2018-06-21T07:15:00Z',
    label: '01:15',
    imagePath: '/satellite-frames/western-ghats1-0715.png',
    caption: 'An extreme convective core (>110 mm/hr) is localized at 10.5°N, 75.6°E (Central Kerala / Western Ghats). A secondary, weaker cell (30–40 mm/hr) is visible further north at 11.8°N.',
  },
  {
    id: 'frame-2',
    timestamp: '2018-06-21T07:45:00Z',
    label: '01:45',
    imagePath: '/satellite-frames/western-ghats1-0745.png',
    caption: '30 minutes later, the primary core at 10.5°N has remained completely stationary over the topography, intensifying to dark red (>120 mm/hr). Simultaneously, the northern cell at 11.8°N has rapidly exploded in intensity, reaching 90–110 mm/hr.',
  },
  // {
  //   id: 'frame-3',
  //   timestamp: '2018-06-21T08:15:00Z',
  //   label: '02:15',
  //   imagePath: '/satellite-frames/western-ghats1-0815.png',
  //   caption: 'Peak cloudburst signature — multiple high-intensity cells merging.',
  // },
];

export const satelliteSource = {
  product: 'INSAT-3DR HEM Rainfall',
  event: 'Western Ghats Cloudburst',
  scaleUnit: 'mm/hr',
  scaleMax: 120,
};