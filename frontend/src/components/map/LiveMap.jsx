import React, { useState } from 'react';
import { 
  Layers, 
  MapPin, 
  Radio, 
  Eye, 
  EyeOff, 
  Compass, 
  Zap, 
  AlertTriangle 
} from 'lucide-react';

export const LiveMap = ({ riskData, isConnected }) => {
  const [showAttentionGrid, setShowAttentionGrid] = useState(true);
  const [showRadarOverlay, setShowRadarOverlay] = useState(true);

  const riskScore = riskData?.riskScore ?? 0;
  const hazardType = riskData?.hazardType || 'Monitoring...';
  const attentionGrid = riskData?.attentionGrid || Array.from({ length: 5 }, () =>
    Array.from({ length: 5 }, () => 0.15)
  );

  return (
    <div className="relative w-full h-130 bg-slate-950 rounded-xl border border-slate-800 overflow-hidden group">
      {/* Blueprint Grid & Tactical Radar Animation */}
      <div className="absolute inset-0 bg-[radial-gradient(#1e293b_1px,transparent_1px)] bg-size-[16px_16px] opacity-60" />

      {showRadarOverlay && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          {/* Radar Sweep Animation */}
          <div className="w-112.5 h-112.5 rounded-full border border-cyan-500/20 relative animate-[spin_8s_linear_infinite]">
            <div className="absolute top-1/2 left-1/2 w-1/2 h-1/2 bg-linear-to-tr from-cyan-500/20 to-transparent origin-top-left -rotate-45 rounded-tl-full" />
          </div>
          <div className="w-75 h-75 rounded-full border border-cyan-500/10 absolute" />
          <div className="w-37.5 h-37.5 rounded-full border border-cyan-500/10 absolute" />
        </div>
      )}

      {/* Spatial Attention Matrix Overlay Layer (5x5 Transformer Heatmap) */}
      {showAttentionGrid && (
        <div className="absolute inset-0 flex items-center justify-center p-12 pointer-events-none">
          <div className="grid grid-cols-5 gap-2 w-full max-w-105 aspect-square opacity-70">
            {attentionGrid.flatMap((row, rIdx) =>
              row.map((weight, cIdx) => (
                <div
                  key={`${rIdx}-${cIdx}`}
                  className="rounded-lg transition-all duration-700 border border-cyan-400/20 flex items-end justify-end p-1"
                  style={{
                    backgroundColor: `rgba(6, 182, 212, ${Math.max(0.05, weight * 0.85)})`,
                    boxShadow: weight > 0.6 ? '0 0 15px rgba(6, 182, 212, 0.5)' : 'none',
                  }}
                >
                  <span className="text-[8px] font-mono text-cyan-200 opacity-80">
                    {weight.toFixed(2)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Dynamic Hazard Target Reticle */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 flex flex-col items-center">
        <div className="relative flex items-center justify-center">
          <span className={`absolute h-10 w-10 rounded-full animate-ping ${
            riskScore > 70 ? 'bg-rose-500/40' : riskScore > 40 ? 'bg-amber-500/40' : 'bg-cyan-500/40'
          }`} />
          <div className={`p-2 rounded-full border shadow-lg backdrop-blur-md ${
            riskScore > 70 
              ? 'bg-rose-950/80 border-rose-500 text-rose-400' 
              : riskScore > 40 
              ? 'bg-amber-950/80 border-amber-500 text-amber-400' 
              : 'bg-cyan-950/80 border-cyan-500 text-cyan-400'
          }`}>
            <MapPin className="w-5 h-5 animate-bounce" />
          </div>
        </div>

        <div className="mt-2 px-3 py-1 bg-slate-900/90 border border-slate-700 rounded-lg backdrop-blur-md text-center shadow-xl">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
            Primary Cell Target
          </div>
          <div className="text-xs font-bold text-slate-100">{hazardType}</div>
        </div>
      </div>

      {/* Floating Layer Controls (Top Left) */}
      <div className="absolute top-4 left-4 flex flex-col gap-2 z-10">
        <button
          onClick={() => setShowAttentionGrid(!showAttentionGrid)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-all ${
            showAttentionGrid
              ? 'bg-cyan-950/80 border-cyan-700 text-cyan-300'
              : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          {showAttentionGrid ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          <span>Transformer XAI Grid</span>
        </button>

        <button
          onClick={() => setShowRadarOverlay(!showRadarOverlay)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium border backdrop-blur-md transition-all ${
            showRadarOverlay
              ? 'bg-cyan-950/80 border-cyan-700 text-cyan-300'
              : 'bg-slate-900/80 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Radar Sweep Layer</span>
        </button>
      </div>

      {/* Map Status Overlay (Bottom Left) */}
      <div className="absolute bottom-4 left-4 p-3 bg-slate-900/90 border border-slate-800 rounded-lg backdrop-blur-md text-xs space-y-1">
        <div className="flex items-center gap-2">
          <Compass className="w-3.5 h-3.5 text-cyan-400" />
          <span className="text-slate-300 font-mono">Center: 28.6139° N, 77.2090° E</span>
        </div>
        <div className="text-[10px] text-slate-500">
          Spatial Resolution: 1km x 1km Doppler Grid
        </div>
      </div>

      {/* Live Connection Badge (Top Right) */}
      <div className="absolute top-4 right-4 z-10">
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border backdrop-blur-md text-xs font-medium ${
          isConnected
            ? 'bg-emerald-950/80 border-emerald-800 text-emerald-400'
            : 'bg-rose-950/80 border-rose-800 text-rose-400'
        }`}>
          <Radio className={`w-3.5 h-3.5 ${isConnected ? 'animate-pulse' : ''}`} />
          <span>{isConnected ? 'LIVE NOWCAST SOCKET' : 'SOCKET DISCONNECTED'}</span>
        </div>
      </div>
    </div>
  );
};

export default LiveMap;