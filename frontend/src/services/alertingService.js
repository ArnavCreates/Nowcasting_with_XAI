// src/services/alertingService.js
//
// Bridges the frontend to the Automated Alerting API (Flask backend — see
// the /alert_api project). Follows the exact same USE_REAL_* toggle pattern
// already established in mlInferenceService.js / webSocketService.js:
// defaults to the existing mockAlerts dataset so the UI keeps working with
// zero setup, and switches to the live Flask API when
// VITE_USE_REAL_ALERTS=true is set in .env.
//
// To enable, add to your .env (or .env.local):
//   VITE_USE_REAL_ALERTS=true
//   VITE_ALERTING_API_URL=http://localhost:8000
//
// If the flag is unset, VITE_ALERTING_API_URL defaults to
// http://localhost:8000 anyway (matching `python main.py`'s default port).

import { mockAlerts } from '../mock/mockAlerts';

const API_BASE_URL = import.meta.env.VITE_ALERTING_API_URL || 'http://localhost:8000';
export const USE_REAL_ALERTS = import.meta.env.VITE_USE_REAL_ALERTS === 'true';

// ---------------------------------------------------------------------------
// Field mapping: Flask API (snake_case) -> frontend shape (camelCase, as
// used throughout mockAlerts.js / AlertCard.jsx / AlertDetails.jsx)
// ---------------------------------------------------------------------------

const HAZARD_MAP = {
  thunderstorm: 'thunderstorm',
  cloudburst: 'cloudburst',
  flash_flood: 'flashFlood',
};

// AlertCard's statusClasses only styles active/acknowledged/resolved — map
// the API's extra escalated/false_alarm states onto the closest existing
// visual bucket so nothing ever renders with an undefined className.
// (escalated is still an open/urgent alert -> active red styling;
//  false_alarm is a closed/non-actionable alert -> resolved green styling)
const STATUS_MAP = {
  active: 'active',
  acknowledged: 'acknowledged',
  escalated: 'active',
  resolved: 'resolved',
  false_alarm: 'resolved',
};

function mapApiAlertToFrontend(apiAlert) {
  const deliveredChannels = new Set(
    (apiAlert.deliveries || [])
      .filter((d) => d.status === 'sent')
      .map((d) => d.channel)
  );

  return {
    id: apiAlert.id,
    regionId: apiAlert.region_id,
    regionName: apiAlert.region_name,
    state: apiAlert.state,
    lat: apiAlert.lat,
    lng: apiAlert.lon,
    hazardType: HAZARD_MAP[apiAlert.hazard_type] || apiAlert.hazard_type,
    severity: apiAlert.severity,
    // The API only tracks one confidence figure (confidence_pct); the
    // frontend's mock data has separate riskScore/confidence fields that
    // happen to always match in mockAlerts.js, so this preserves that.
    riskScore: Math.round(apiAlert.confidence_pct),
    confidence: Math.round(apiAlert.confidence_pct),
    status: STATUS_MAP[apiAlert.status] || apiAlert.status,
    // No numeric threshold-config ID comes from the API — pass the
    // human-readable rule name through instead. AlertDetails.jsx already
    // falls back to displaying the raw string when it's not a recognized
    // key in thresholdRuleMeta, so this renders correctly with zero
    // changes needed there (see: `thresholdRuleMeta[id]?.label ?? id`).
    triggeredThresholdId: apiAlert.triggered_rule || null,
    timestamp: apiAlert.created_at,
    description: apiAlert.summary,
    delivery: {
      sms: deliveredChannels.has('sms'),
      push: deliveredChannels.has('push'),
      siren: deliveredChannels.has('siren'),
    },
    statusTimeline: (apiAlert.timeline || []).map((event) => ({
      label: event.label,
      time: event.timestamp,
    })),
    // Kept for anything that wants the raw record later (e.g. wiring the
    // Live Map's estimated-impact countdown to estimated_impact_minutes).
    _raw: apiAlert,
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Fetches the current alert list from the live API. Falls back to
 * mockAlerts if real alerts are disabled, OR if the live request fails —
 * so a backend outage never blanks the Alert Ticker mid-demo.
 *
 * @param {string} [regionId] - When provided, scopes the fetch to just
 *   that region's alerts (used by the Live Map's single-city view). When
 *   omitted, returns alerts for every region (used by the Alert Ticker
 *   and the Live Map's "All India" national view).
 */
export async function fetchAlerts(regionId) {
  if (!USE_REAL_ALERTS) {
    return regionId ? mockAlerts.filter((a) => a.regionId === regionId) : mockAlerts;
  }

  try {
    const url = regionId
      ? `${API_BASE_URL}/alerts?region_id=${encodeURIComponent(regionId)}`
      : `${API_BASE_URL}/alerts`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Alerts HTTP ${response.status}`);
    const data = await response.json();
    return data.map(mapApiAlertToFrontend);
  } catch (err) {
    console.error('Live alert fetch failed, falling back to mock alerts:', err);
    return regionId ? mockAlerts.filter((a) => a.regionId === regionId) : mockAlerts;
  }
}

export async function acknowledgeAlert(alertId, actor, note) {
  if (!USE_REAL_ALERTS) {
    console.warn('acknowledgeAlert called in mock mode — no backend to update.');
    return null;
  }
  const response = await fetch(`${API_BASE_URL}/alerts/${alertId}/acknowledge`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ actor, note }),
  });
  if (!response.ok) throw new Error(`Acknowledge HTTP ${response.status}`);
  return mapApiAlertToFrontend(await response.json());
}

export async function escalateAlert(alertId, actor, escalateTo, note) {
  if (!USE_REAL_ALERTS) {
    console.warn('escalateAlert called in mock mode — no backend to update.');
    return null;
  }
  const response = await fetch(`${API_BASE_URL}/alerts/${alertId}/escalate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ actor, escalate_to: escalateTo, note }),
  });
  if (!response.ok) throw new Error(`Escalate HTTP ${response.status}`);
  return mapApiAlertToFrontend(await response.json());
}

export async function resolveAlert(alertId, actor, outcome = 'resolved', note) {
  if (!USE_REAL_ALERTS) {
    console.warn('resolveAlert called in mock mode — no backend to update.');
    return null;
  }
  const response = await fetch(`${API_BASE_URL}/alerts/${alertId}/resolve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ actor, outcome, note }),
  });
  if (!response.ok) throw new Error(`Resolve HTTP ${response.status}`);
  return mapApiAlertToFrontend(await response.json());
}

export default { fetchAlerts, acknowledgeAlert, escalateAlert, resolveAlert, USE_REAL_ALERTS };