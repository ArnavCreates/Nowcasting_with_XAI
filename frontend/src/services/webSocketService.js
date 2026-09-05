import { predictSevereWeather } from './mlInferenceService';

const WS_URL = import.meta.env.VITE_ML_MODEL_WS_URL || 'ws://localhost:8000/ws/nowcast';
const USE_REAL_MODEL = import.meta.env.VITE_USE_REAL_MODEL === 'true';

class NowcastWebSocketService {
  constructor() {
    this.socket = null;
    this.mockInterval = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  /**
   * Connects to the live nowcast stream (or starts a mock stream).
   * Every frame — mock or real — is expected to match the same shape
   * returned by mlInferenceService.predictSevereWeather (riskScore,
   * hazardType, confidence, hourlyTrend[], metrics{}, featureImportance[],
   * attentionGrid, xaiExplanation), so consuming pages never need to branch
   * on where the data came from.
   */
  connect(regionId, regionMeta, onMessage, onStatusChange) {
    if (!USE_REAL_MODEL) {
      onStatusChange(true);

      // Emit a fresh, full-shaped frame every few seconds by re-running the
      // same inference function with slightly jittered telemetry — this is
      // what makes the "live" updates visibly change over time in the demo,
      // while staying shape-identical to a real backend push.
      this.mockInterval = setInterval(async () => {
        const jitter = (base, spread) => Math.max(0, Math.round(base + (Math.random() - 0.5) * spread));
        const telemetry = {
          precipitation: jitter(60, 20),
          windSpeed: jitter(40, 15),
          iwv: jitter(45, 10),
          cape: jitter(1800, 400),
          cin: -jitter(15, 8),
          cttDrop: jitter(8, 4),
        };

        const frame = await predictSevereWeather({
          regionId,
          regionName: regionMeta?.name,
          timestamp: new Date().toISOString(),
          telemetry,
        });

        onMessage(frame);
      }, 3000);
      return;
    }

    // Production WebSocket connection — real backend is expected to push
    // messages matching the same response contract as predictSevereWeather.
    try {
      this.socket = new WebSocket(`${WS_URL}?regionId=${regionId}`);

      this.socket.onopen = () => {
        this.reconnectAttempts = 0;
        onStatusChange(true);
      };

      this.socket.onmessage = (event) => {
        try {
          onMessage(JSON.parse(event.data));
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      this.socket.onerror = (error) => console.error('WebSocket Error:', error);

      this.socket.onclose = () => {
        onStatusChange(false);
        this.handleReconnect(regionId, regionMeta, onMessage, onStatusChange);
      };
    } catch (err) {
      console.error('WebSocket Initialization Error:', err);
      onStatusChange(false);
    }
  }

  handleReconnect(regionId, regionMeta, onMessage, onStatusChange) {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts += 1;
      const timeout = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
      setTimeout(() => this.connect(regionId, regionMeta, onMessage, onStatusChange), timeout);
    }
  }

  disconnect() {
    if (this.mockInterval) clearInterval(this.mockInterval);
    if (this.socket) {
      this.socket.onclose = null; // prevent reconnect loop on intentional close
      this.socket.close();
    }
  }
}

export const websocketService = new NowcastWebSocketService();