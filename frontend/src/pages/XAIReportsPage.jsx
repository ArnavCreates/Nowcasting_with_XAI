import React, { useEffect } from 'react';
import { useWeatherStore } from '../store/useWeatherStore';
import { Cpu, MapPin, BarChart3, FileText } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell } from 'recharts';
import SatelliteInputStrip from '../components/xai/SatelliteInputStrip';

const FEATURE_COLORS = ['#e11d48', '#9333ea', '#0284c7', '#06b6d4', '#f59e0b', '#16a34a', '#64748b'];

export const XAIReportsPage = () => {
  const selectedRegion = useWeatherStore((state) => state.selectedRegion);
  const currentRiskData = useWeatherStore((state) => state.currentRiskData);
  const fetchRiskAnalysis = useWeatherStore((state) => state.fetchRiskAnalysis);

  useEffect(() => {
    if (selectedRegion) fetchRiskAnalysis();
  }, [selectedRegion?.id]);

  if (!selectedRegion) {
    return <div className="p-6 text-sm text-slate-400">Select a region to view XAI diagnostics.</div>;
  }

  const shapData = (currentRiskData?.featureImportance || []).map((f, i) => ({
    feature: f.feature,
    importance: Math.min(100, Math.abs(f.impact) * 2),
    color: FEATURE_COLORS[i % FEATURE_COLORS.length],
  }));

  const summaryBlock = currentRiskData?.xaiExplanation && (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-purple-600">
        <FileText className="w-4 h-4" />
        <h3 className="text-xs font-bold uppercase tracking-wider">Summary</h3>
      </div>
      <p className="text-sm leading-relaxed text-slate-700 bg-purple-50/60 rounded-xl p-4 border border-purple-100">
        Spatiotemporal Attention (STA): Flags critical flash flood risks at 10.5°N as the stationary 120 mm/hr storm core persists over identical river catchments.<br/>
        Integrated Gradients (IG): Detects rapid cell intensification at 11.8°N, tripling in intensity (cyan to deep red) within 30 minutes.<br/>
        DGMR Projection: Projects a 94% probability of sustained cloudbursts (less than 100 mm/hr) over the Kerala Western Ghats for the next 2 hours.<br/>
      </p>
    </div>
  );

  return (
    <div className="p-6 space-y-6 bg-slate-50 min-h-screen text-slate-900">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-slate-200 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-purple-50 border border-purple-200 rounded-2xl text-purple-600 shadow-sm">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">XAI Model Diagnostics</h1>
            <p className="text-xs text-slate-500">Satellite input, reasoning, and feature contribution</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3.5 py-2 bg-white border border-slate-200 rounded-xl shadow-sm text-xs font-semibold text-slate-700">
          <MapPin className="w-4 h-4 text-purple-600" />
          <span>Target: <strong>{selectedRegion.name}, {selectedRegion.state}</strong></span>
        </div>
      </div>

      {/* 1. Satellite Input + Summary, side by side */}
      <SatelliteInputStrip summary={summaryBlock} />

      {/* 2. Feature Contribution — full width, below */}
      <div className="p-6 bg-white rounded-2xl border border-slate-200 shadow-sm space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
          <BarChart3 className="w-5 h-5 text-purple-600" />
          <h2 className="font-bold text-slate-800 text-sm uppercase tracking-wider">Feature Contribution</h2>
        </div>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart layout="vertical" data={shapData} margin={{ top: 10, right: 30, left: 40, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <XAxis type="number" stroke="#64748b" fontSize={11} domain={[0, 100]} unit="%" />
              <YAxis dataKey="feature" type="category" stroke="#64748b" fontSize={10} width={140} tickLine={false} />
              <Tooltip contentStyle={{ backgroundColor: '#fff', borderRadius: '12px', border: '1px solid #e2e8f0' }} />
              <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
                {shapData.map((entry, index) => <Cell key={index} fill={entry.color} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default XAIReportsPage;