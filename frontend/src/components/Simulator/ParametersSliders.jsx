import React, { useEffect } from 'react';
import { useWeatherStore } from '../../store/useWeatherStore';
import { CloudRain, Wind, Droplets, Thermometer, Gauge, Zap } from 'lucide-react';

export const ParameterSliders = () => {
  const { simulatorParameters, setSimulatorParameters, fetchRiskAnalysis, isLoading } = useWeatherStore();

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchRiskAnalysis(simulatorParameters);
    }, 300);
    return () => clearTimeout(timer);
  }, [simulatorParameters, fetchRiskAnalysis]);

  const handleChange = (key, value) => setSimulatorParameters({ [key]: Number(value) });

  return (
    <div className="p-5 bg-white text-slate-900 rounded-2xl border border-slate-200 shadow-sm space-y-5">
      <div className="flex justify-between items-center border-b border-slate-100 pb-3">
        <h3 className="font-bold text-sm text-slate-800 uppercase tracking-wider">Simulation Controls</h3>
        {isLoading && <span className="text-xs text-amber-600 font-semibold animate-pulse">Executing Model...</span>}
      </div>

      <div className="space-y-4">
        <Slider icon={<CloudRain className="w-4 h-4 text-blue-600" />} label="Precipitation Rate" unit="mm/hr" min={0} max={150} value={simulatorParameters?.precipitation ?? 0} onChange={(v) => handleChange('precipitation', v)} />
        <Slider icon={<Wind className="w-4 h-4 text-blue-600" />} label="Wind Speed" unit="km/h" min={0} max={180} value={simulatorParameters?.windSpeed ?? 0} onChange={(v) => handleChange('windSpeed', v)} />
        <Slider icon={<Droplets className="w-4 h-4 text-blue-600" />} label="Integrated Water Vapor (IWV)" unit="kg/m²" min={0} max={100} value={simulatorParameters?.iwv ?? 0} onChange={(v) => handleChange('iwv', v)} />
        <Slider icon={<Zap className="w-4 h-4 text-blue-600" />} label="CAPE (Instability)" unit="J/kg" min={0} max={4000} value={simulatorParameters?.cape ?? 0} onChange={(v) => handleChange('cape', v)} />
        <Slider icon={<Gauge className="w-4 h-4 text-blue-600" />} label="CIN (Convective Inhibition)" unit="J/kg" min={-50} max={0} value={simulatorParameters?.cin ?? 0} onChange={(v) => handleChange('cin', v)} />
        <Slider icon={<Thermometer className="w-4 h-4 text-blue-600" />} label="Cloud Top Temp Drop" unit="°C/30min" min={0} max={20} value={simulatorParameters?.cttDrop ?? 0} onChange={(v) => handleChange('cttDrop', v)} />
      </div>
    </div>
  );
};

function Slider({ icon, label, unit, min, max, value, onChange }) {
  return (
    <div>
      <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1.5">
        <span className="flex items-center gap-1.5">{icon} {label}</span>
        <span className="font-mono text-blue-600">{value} {unit}</span>
      </div>
      <input 
        type="range" 
        min={min} 
        max={max} 
        value={value} 
        onChange={(e) => onChange(e.target.value)} 
        className="w-full accent-blue-600 bg-slate-100 h-2 rounded-lg appearance-none cursor-pointer" 
      />
    </div>
  );
}

export default ParameterSliders;