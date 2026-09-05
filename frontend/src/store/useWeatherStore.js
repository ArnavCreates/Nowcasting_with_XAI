import { create } from 'zustand';
import { mockAlerts } from '../mock/mockAlerts';
import { predictSevereWeather } from '../services/mlInferenceService';
import { websocketService } from '../services/webSocketService';

// Single source of truth for region metadata — derived from mockAlerts so
// there is exactly one region list shared by every page (Live Map, Risk
// Analysis, Simulator, XAI Reports, Alert Ticker, Historical Data).
function formatRegionObject(regionIdOrObj) {
  if (!regionIdOrObj) return null;

  const regionId = typeof regionIdOrObj === 'string' ? regionIdOrObj : regionIdOrObj.regionId;
  const alert = mockAlerts.find((a) => a.regionId === regionId);
  if (!alert) return null;

  return {
    id: alert.regionId,
    name: alert.regionName,
    state: alert.state,
    lat: alert.lat,
    lng: alert.lng,
    hazardType: alert.hazardType,
    riskScore: alert.riskScore,
    severity: alert.severity,
  };
}

function getDefaultRegion() {
  const activeAlerts = mockAlerts.filter((a) => a.status === 'active');
  const highest = [...activeAlerts].sort((a, b) => b.riskScore - a.riskScore)[0];
  return formatRegionObject((highest || mockAlerts[0]).regionId);
}

export const useWeatherStore = create((set, get) => ({
  // ---- Region selection — single shared source across all 6 pages ----
  regions: mockAlerts.map((a) => formatRegionObject(a.regionId)),
  selectedRegion: getDefaultRegion(),

  setSelectedRegion: (regionIdOrObj) =>
    set({ selectedRegion: formatRegionObject(regionIdOrObj) }),

  // ---- Live Map forecast lead-time scrubber (+2h to +6h, per problem statement) ----
  forecastHorizon: { min: 1, max: 6, value: 1 },

  setForecastHorizonValue: (value) =>
    set((state) => ({
      forecastHorizon: {
        ...state.forecastHorizon,
        value: Math.min(Math.max(value, state.forecastHorizon.min), state.forecastHorizon.max),
      },
    })),

  // ---- Live Map hazard layer + render style ----
  mapLayers: {
    activeHazard: 'all', // 'all' | 'thunderstorm' | 'cloudburst' | 'flashFlood'
    renderStyle: 'heatmap', // 'heatmap' | 'contour'
  },

  setActiveHazard: (activeHazard) =>
    set((state) => ({ mapLayers: { ...state.mapLayers, activeHazard } })),

  setRenderStyle: (renderStyle) =>
    set((state) => ({ mapLayers: { ...state.mapLayers, renderStyle } })),

  // ---- Live model/inference output (mock now, real backend later — see
  // mlInferenceService.js for the swap point) ----
  currentRiskData: null,
  isLoading: false,
  error: null,
  isConnectedToWS: false,
  streamPaused: false,
  setStreamPaused: (paused) => set({ streamPaused: paused }),

  // ---- What-If Simulator parameters — full predictive matrix, matches the
  // problem statement's three ingredients (moisture, instability, kinematics) ----
  simulatorParameters: {
    precipitation: 60, // mm/h
    windSpeed: 40,      // km/h
    iwv: 45,            // kg/m²
    cape: 1800,         // J/kg
    cin: -15,           // J/kg
    cttDrop: 8,          // °C/30min
  },

  setSimulatorParameters: (params) =>
    set((state) => ({ simulatorParameters: { ...state.simulatorParameters, ...params } })),

  // Seeds the simulator to the currently selected region's real telemetry —
  // called on region change so the sliders start from a realistic baseline.
  syncSimulatorToRegionTelemetry: (telemetry) => {
    if (!telemetry) return;
    set((state) => ({
      simulatorParameters: {
        ...state.simulatorParameters,
        precipitation: telemetry.rainfallRate ?? state.simulatorParameters.precipitation,
        windSpeed: telemetry.windGust ?? state.simulatorParameters.windSpeed,
        iwv: telemetry.iwv ?? state.simulatorParameters.iwv,
        cape: telemetry.cape ?? state.simulatorParameters.cape,
        cin: telemetry.cin ?? state.simulatorParameters.cin,
        cttDrop: telemetry.cttDrop30m ?? state.simulatorParameters.cttDrop,
      },
    }));
  },

  // ---- Core inference call — this is the single place a real backend
  // connects. Swap happens entirely inside mlInferenceService.js via
  // VITE_USE_REAL_MODEL; nothing here needs to change when that flips. ----
  fetchRiskAnalysis: async (telemetryOverride) => {
    const region = get().selectedRegion;
    if (!region) return;

    const telemetry = telemetryOverride || get().simulatorParameters;

    set({ isLoading: true, error: null });
    try {
      const payload = {
        regionId: region.id,
        regionName: region.name,
        lat: region.lat,
        lng: region.lng,
        timestamp: new Date().toISOString(),
        telemetry,
      };

      const result = await predictSevereWeather(payload);
      set({ currentRiskData: result, isLoading: false });
    } catch (err) {
      set({ error: err.message || 'Failed to fetch risk analysis', isLoading: false });
    }
  },

  // ---- Live nowcast WebSocket stream (mock-generated now, real socket
  // later — toggle lives in webSocketService.js) ----
  connectWebSocket: () => {
    const region = get().selectedRegion;
    if (!region) return;
    websocketService.disconnect();

    websocketService.connect(
      region.id,
      region,
      (frame) => set({currentRiskData: frame}),
      (status) => set({ isConnectedToWS: status })
    );
  },

  disconnectWebSocket: () => {
    websocketService.disconnect();
    set({ isConnectedToWS: false });
  },

  // ---- Alert Ticker filters ----
  alertFilters: {
    hazardType: 'all',
    severity: 'all',
    status: 'active',
    searchQuery: '',
  },

  setAlertFilters: (filters) =>
    set((state) => ({ alertFilters: { ...state.alertFilters, ...filters } })),

  // ---- Historical Replay playback ----
  replayState: {
    isPlaying: false,
    speed: 1,
    currentEventId: null,
    currentFrameIndex: 0,
  },

  setReplayState: (partial) =>
    set((state) => ({ replayState: { ...state.replayState, ...partial } })),
}));

export default useWeatherStore;