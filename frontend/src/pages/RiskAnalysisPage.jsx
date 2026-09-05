import React, { useEffect } from 'react';
import { useWeatherStore } from '../store/useWeatherStore';
import { ShieldAlert, Activity, Zap, CloudRain, Wind, Info, Layers, Clock } from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LabelList
} from 'recharts';

export const RiskAnalysisPage = () => {
  const selectedRegion = useWeatherStore((state) => state.selectedRegion);
  const currentRiskData = useWeatherStore((state) => state.currentRiskData);
  const isLoading = useWeatherStore((state) => state.isLoading);
  const fetchRiskAnalysis = useWeatherStore((state) => state.fetchRiskAnalysis);

  useEffect(() => {
    if (selectedRegion) fetchRiskAnalysis();
  }, [selectedRegion?.id]);

  if (!selectedRegion) {
    return <div className="p-6 text-sm text-slate-400">Select a region to view risk analysis.</div>;
  }

  const riskData = currentRiskData || { 
    riskScore: selectedRegion.riskScore, 
    hazardType: selectedRegion.hazardType, 
    confidence: 0.85 
  };
  const hourlyTrend = riskData.hourlyTrend || [];
  const featureImportance = riskData.featureImportance || [];
  const metrics = riskData.metrics || {};

  // Theme helper providing Tailwind classes and explicit Hex codes for Recharts
  const getRiskTheme = (score) => {
    if (score > 75) return { text: 'text-rose-600', badge: 'bg-rose-500', hex: '#f43f5e' };
    if (score > 45) return { text: 'text-amber-600', badge: 'bg-amber-500', hex: '#f59e0b' };
    return { text: 'text-emerald-600', badge: 'bg-emerald-500', hex: '#10b981' };
  };

  const theme = getRiskTheme(riskData.riskScore);

  // Recharts formatted dataset
  const chartData = hourlyTrend.map((pt) => ({
    time: `+${pt.hour}h`,
    risk: pt.risk,
  }));

  // Sleek Glassmorphism Tooltip for Recharts
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white/95 backdrop-blur-md p-3 rounded-xl shadow-xl border border-slate-200/80 text-xs space-y-1">
          <p className="font-bold text-slate-400 uppercase tracking-wider text-[10px]">
            Forecast {label}
          </p>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: theme.hex }} />
            <span className="font-black text-slate-900 text-sm">
              {payload[0].value} <span className="text-slate-400 font-medium text-xs">/ 100 Risk</span>
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6 bg-slate-50 min-h-screen">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">XAI Nowcasting Engine</span>
            {isLoading && <span className="text-xs text-amber-600 font-semibold animate-pulse">• Running ML Inference...</span>}
          </div>
          <h1 className="text-2xl font-black text-slate-900">{selectedRegion.name} — Severe Weather Nowcast</h1>
          <p className="text-xs text-slate-500">
            {selectedRegion.state} • +1h to +6h Window • Model Confidence: {((riskData.confidence ?? 0.85) * 100).toFixed(1)}%
          </p>
        </div>
        <span className={`px-3.5 py-1.5 rounded-xl text-xs font-bold text-white ${theme.badge} shadow-sm uppercase`}>
          {selectedRegion.severity} severity
        </span>
      </div>

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-slate-400 uppercase">Overall Risk Score</span>
            <ShieldAlert className={`w-5 h-5 ${theme.text}`} />
          </div>
          <div className="mt-4">
            <div className="flex items-baseline gap-1">
              <span className={`text-4xl font-black font-mono ${theme.text}`}>{riskData.riskScore}</span>
              <span className="text-slate-400 font-bold text-sm">/ 100</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 mt-3 overflow-hidden">
              <div className={`h-full ${theme.badge}`} style={{ width: `${riskData.riskScore}%` }} />
            </div>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-slate-400 uppercase">Primary Hazard</span>
            <CloudRain className="w-5 h-5 text-blue-500" />
          </div>
          <div className="mt-4">
            <span className="text-xl font-black text-slate-900 capitalize">{riskData.hazardType}</span>
            <p className="text-[11px] text-slate-500 mt-1">Precipitation: <strong className="text-slate-700">{metrics.precipitationRate ?? '—'}</strong></p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-slate-400 uppercase">IWV Moisture</span>
            <Activity className="w-5 h-5 text-amber-500" />
          </div>
          <div className="mt-4">
            <span className="text-xl font-black text-slate-900 font-mono">{metrics.iwvMoisture ?? '—'}</span>
            <p className="text-[11px] text-slate-500 mt-1">CAPE: <strong className="text-slate-700">{metrics.cape ?? '—'}</strong></p>
          </div>
        </div>

        <div className="bg-white p-5 rounded-2xl border border-slate-200/80 shadow-sm">
          <div className="flex justify-between items-start">
            <span className="text-xs font-bold text-slate-400 uppercase">Wind / CTT</span>
            <Wind className="w-5 h-5 text-teal-500" />
          </div>
          <div className="mt-4">
            <span className="text-xl font-black text-slate-900 font-mono">{metrics.windSpeed ?? '—'}</span>
            <p className="text-[11px] text-slate-500 mt-1">CTT Drop: <strong className="text-slate-700">{metrics.cttDropRate ?? '—'}</strong></p>
          </div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recharts Vibrant Area Chart */}
        <div className="lg:col-span-2 bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
          <div className="flex justify-between items-center border-b border-slate-100 pb-3">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-blue-600" />
              <h2 className="text-sm font-bold text-slate-900">+1h to +6h Nowcast Window</h2>
            </div>
            <span className="text-[11px] font-bold text-slate-500 bg-slate-100 px-2.5 py-1 rounded-lg">
              Dynamic Risk Trajectory
            </span>
          </div>

          <div className="h-64 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 25, right: 20, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="nowcastRiskGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={theme.hex} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={theme.hex} stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis
                  dataKey="time"
                  stroke="#64748b"
                  fontSize={12}
                  fontWeight={600}
                  tickLine={false}
                  axisLine={false}
                  dy={10}
                />
                <YAxis
                  domain={[0, 100]}
                  stroke="#94a3b8"
                  fontSize={11}
                  tickLine={false}
                  axisLine={false}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="risk"
                  stroke={theme.hex}
                  strokeWidth={3.5}
                  fillOpacity={1}
                  fill="url(#nowcastRiskGrad)"
                  dot={{
                    r: 5,
                    fill: '#ffffff',
                    stroke: theme.hex,
                    strokeWidth: 3,
                  }}
                  activeDot={{
                    r: 7,
                    fill: theme.hex,
                    stroke: '#ffffff',
                    strokeWidth: 3,
                  }}
                >
                  <LabelList
                    dataKey="risk"
                    position="top"
                    offset={12}
                    fill="#334155"
                    fontSize={12}
                    fontWeight={800}
                  />
                </Area>
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Feature Impact List */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Zap className="w-5 h-5 text-amber-500" />
            <h2 className="text-sm font-bold text-slate-900">Feature Impact</h2>
          </div>
          <div className="space-y-3.5 pt-1">
            {featureImportance.map((item, idx) => (
              <div key={idx} className="space-y-1 text-xs">
                <div className="flex justify-between font-semibold text-slate-700">
                  <span>{item.feature}</span>
                  <span className={item.impact > 0 ? 'text-rose-600 font-bold' : 'text-emerald-600 font-bold'}>
                    {item.impact > 0 ? `+${item.impact}` : item.impact}
                  </span>
                </div>
                <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${item.impact > 0 ? 'bg-rose-500' : 'bg-emerald-500'}`}
                    style={{ width: `${Math.min(100, Math.abs(item.impact) * 2)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Insights Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bg-slate-900 text-white p-6 rounded-2xl shadow-md space-y-3">
          <div className="flex items-center gap-2 text-blue-400">
            <Info className="w-5 h-5" />
            <h3 className="text-sm font-bold uppercase tracking-wider">XAI Diagnostic Summary</h3>
          </div>
          <p className="text-xs leading-relaxed text-slate-300">{riskData.xaiExplanation}</p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200/80 shadow-sm space-y-3">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Layers className="w-5 h-5 text-blue-600" />
            <h3 className="text-sm font-bold text-slate-900">Telemetry Feeds</h3>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(metrics).map(([key, val]) => (
              <div key={key} className="p-2.5 bg-slate-50 rounded-xl border border-slate-100">
                <span className="text-[10px] text-slate-400 block font-medium capitalize">
                  {key.replace(/([A-Z])/g, ' $1')}
                </span>
                <span className="font-bold text-slate-800">{val}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskAnalysisPage;