import React from 'react';
import { Satellite } from 'lucide-react';
import { mockSatelliteFrames, satelliteSource } from '../../mock/mockSatelliteFrames';

export default function SatelliteInputStrip({ summary }) {
  return (
    <div className="p-6 bg-white rounded-2xl border border-slate-200 shadow-sm space-y-4">
      <div className="flex items-center justify-between border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Satellite className="w-5 h-5 text-sky-600" />
          <h2 className="font-bold text-slate-800 text-sm uppercase tracking-wider">Satellite Input</h2>
        </div>
        <span className="text-xs font-mono font-bold px-2.5 py-1 bg-sky-50 text-sky-700 rounded-lg border border-sky-200">
          {satelliteSource.product}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_1fr] gap-4">
        {/* Two satellite frames, stacked vertically on the left */}
        <div className="lg:col-span-2 grid grid-cols-1 sm:grid-cols-2 gap-4">
          {mockSatelliteFrames.map((frame) => (
            <div key={frame.id} className="rounded-xl overflow-hidden border border-slate-200 bg-slate-50">
              <img
                src={frame.imagePath}
                alt={`${satelliteSource.event} — ${frame.label}`}
                className="w-full h-auto object-contain"
              />
              <div className="p-3 border-t border-slate-200">
                <p className="text-xs font-bold text-slate-800">{frame.label}</p>
                <p className="text-[11px] text-slate-500 mt-0.5">{frame.caption}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Summary/reasoning panel, beside the images */}
        <div className="flex flex-col justify-center">{summary}</div>
      </div>

      <div className="pt-1 text-[11px] text-slate-400">
        Scale: 0–{satelliteSource.scaleMax} {satelliteSource.scaleUnit} · {satelliteSource.event}
      </div>
    </div>
  );
}