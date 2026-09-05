import React from 'react';
import { 
  ResponsiveContainer, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Cell, 
  CartesianGrid 
} from 'recharts';
import { ShieldAlert, Activity, CheckCircle2, AlertTriangle, Cpu, Zap } from 'lucide-react';

export const RiskComparisionCanvas = ({ riskData, isLoading }) => {
  const riskScore = riskData?.riskScore ?? 0;
  const hazardType = riskData?.hazardType || 'Low Risk';
  const confidence = riskData?.confidence ? (riskData.confidence * 100).toFixed(0) : '--';

  const chartData = [
    { metric: 'Baseline', score: 25 },
    { metric: 'Simulated', score: riskScore },
  ];

  return (
    <div className="p-6 bg-white text-slate-900 rounded-2xl border border-slate-200 shadow-sm space-y-6 relative overflow-hidden">
      {isLoading && (
        <div className="absolute inset-0 bg-white/80 backdrop-blur-sm flex items-center justify-center z-10">
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-blue-600 animate-spin" />
            <span className="text-sm font-semibold text-blue-900">
              Running Neural Inference...
            </span>
          </div>
        </div>
      )}

      <div className="flex justify-between items-center border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-blue-600" />
          <h3 className="font-bold text-sm text-slate-800 uppercase tracking-wider">Simulation Impact Visualizer</h3>
        </div>
        <span className="flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 bg-blue-50 text-blue-700 border border-blue-200 rounded-full">
          <Zap className="w-3.5 h-3.5 text-amber-500" />
          Model Confidence: {confidence}%
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Baseline Card */}
        <div className="p-4 bg-slate-50 rounded-xl border border-slate-200 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
              Baseline Output
            </span>
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
          </div>
          <div className="text-3xl font-bold text-slate-800">25 / 100</div>
          <div className="text-xs text-emerald-600 font-semibold">Normal Operating Limits</div>
        </div>

        {/* Simulated Output Card */}
        <div className={`p-4 rounded-xl border space-y-2 ${
          riskScore > 70 
            ? 'bg-rose-50 border-rose-200 text-rose-900' 
            : riskScore > 40 
            ? 'bg-amber-50 border-amber-200 text-amber-900' 
            : 'bg-emerald-50 border-emerald-200 text-emerald-900'
        }`}>
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold uppercase tracking-wider opacity-80">
              Simulated Forecast
            </span>
            {riskScore > 70 ? (
              <ShieldAlert className="w-4 h-4 text-rose-600" />
            ) : (
              <AlertTriangle className="w-4 h-4 text-amber-600" />
            )}
          </div>
          <div className="text-3xl font-black">{riskScore} / 100</div>
          <div className="text-xs font-bold">{hazardType}</div>
        </div>
      </div>

      {/* Recharts Visual Comparison */}
      <div className="space-y-2 pt-2">
        <div className="text-xs text-slate-500 font-semibold">Baseline vs. Simulation Variance</div>
        <div className="h-52 w-full bg-slate-50 p-3 rounded-xl border border-slate-200">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="metric" stroke="#64748b" fontSize={12} tickLine={false} />
              <YAxis domain={[0, 100]} stroke="#64748b" fontSize={12} tickLine={false} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#ffffff', borderColor: '#cbd5e1', borderRadius: '0.75rem', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}
                itemStyle={{ color: '#2563eb' }}
              />
              <Bar dataKey="score" radius={[6, 6, 0, 0]}>
                <Cell fill="#3b82f6" />
                <Cell fill={riskScore > 70 ? '#f43f5e' : riskScore > 40 ? '#f59e0b' : '#10b981'} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default RiskComparisionCanvas;