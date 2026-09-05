import React, { useEffect } from 'react';
import { useWeatherStore } from '../store/useWeatherStore';
import { mockTelemetry } from '../mock/mockTelemetry';
import { Sliders, MapPin } from 'lucide-react';
import ParameterSliders from '../components/Simulator/ParametersSliders.jsx';
import RiskComparisionCanvas from '../components/Simulator/RiskComparisionCanvas';
import SimulatorChat from '../components/Simulator/SimulatorChat';

export const SimulatorPage = () => {
  const selectedRegion = useWeatherStore((state) => state.selectedRegion);
  const currentRiskData = useWeatherStore((state) => state.currentRiskData);
  const isLoading = useWeatherStore((state) => state.isLoading);
  const syncSimulatorToRegionTelemetry = useWeatherStore((state) => state.syncSimulatorToRegionTelemetry);
  const setStreamPaused = useWeatherStore((state) => state.setStreamPaused);

  useEffect(() => {
    setStreamPaused(true);
    return () => setStreamPaused(false);
  }, [setStreamPaused]);

  useEffect(() => {
    if (selectedRegion && mockTelemetry[selectedRegion.id]?.current) {
      syncSimulatorToRegionTelemetry(mockTelemetry[selectedRegion.id].current);
    }
  }, [selectedRegion, syncSimulatorToRegionTelemetry]);

  if (!selectedRegion) {
    return (
      <div className="p-6 text-sm text-slate-500 bg-slate-50 min-h-screen">
        Select a region to run the simulator.
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 bg-slate-50 min-h-screen text-slate-900">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 border-b border-slate-200 pb-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-blue-50 border border-blue-200 rounded-2xl text-blue-600 shadow-sm">
            <Sliders className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-slate-900">What-If Scenario Simulator</h1>
            <p className="text-xs text-slate-500">Adjust telemetry to see simulated model output, sent back for a new risk calculation.</p>
          </div>
        </div>
        <div className="flex items-center gap-2 px-3.5 py-2 bg-white border border-slate-200 rounded-xl shadow-sm text-xs font-semibold text-slate-700">
          <MapPin className="w-4 h-4 text-blue-600" />
          <span>Target: <strong>{selectedRegion.name}, {selectedRegion.state}</strong></span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-6">
          <ParameterSliders />
          <SimulatorChat riskData={currentRiskData} />
        </div>
        <RiskComparisionCanvas riskData={currentRiskData} isLoading={isLoading} />
      </div>
    </div>
  );
};

export default SimulatorPage;