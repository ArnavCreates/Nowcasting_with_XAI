import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

const hazardLabels = {
  thunderstorm: 'Thunderstorm',
  cloudburst: 'Cloudburst',
  flashFlood: 'Flash Flood',
};

export default function HistoricalReplay({ event, onClose }) {
  if (!event) return null;

  const peakRisk = Math.max(
    ...event.replay.series.map((point) => point.riskScore)
  );

  const peakIWV = Math.max(
    ...event.replay.series.map((point) => point.iwv)
  );

  const lowestCTT = Math.min(
    ...event.replay.series.map((point) => point.ctt)
  );

  const chartData = event.replay.series.map((point) => ({
    time: point.tOffset === 0
      ? 'Event'
      : `${point.tOffset > 0 ? '+' : ''}${point.tOffset}h`,
    iwv: point.iwv,
    ctt: point.ctt,
    risk: Math.round(point.riskScore * 100),
  }));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">

      <div className="w-full max-w-5xl rounded-2xl bg-white shadow-2xl">

        {/* Header */}
        <div className="flex items-start justify-between border-b p-5">

          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Historical Replay
            </p>

            <h2 className="mt-1 text-xl font-bold text-slate-900">
              {event.id} — {event.region}
            </h2>

            <p className="mt-1 text-sm text-slate-500">
              {hazardLabels[event.hazardType] || event.hazardType}{' '}
              • {event.severity}
            </p>
          </div>

          <button
            onClick={onClose}
            className="rounded-lg px-3 py-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900"
          >
            ✕
          </button>

        </div>


        {/* Event summary */}
        <div className="grid gap-3 border-b bg-slate-50 p-5 sm:grid-cols-3">

          <div>
            <p className="text-xs text-slate-400">
              Lead Time
            </p>

            <p className="mt-1 font-semibold text-slate-800">
              {event.leadTime > 0
                ? `${event.leadTime} hrs`
                : 'No lead time'}
            </p>
          </div>


          <div>
            <p className="text-xs text-slate-400">
              Outcome
            </p>

            <p className="mt-1 font-semibold capitalize text-slate-800">
              {event.outcome === 'falseAlarm'
                ? 'False Alarm'
                : event.outcome}
            </p>
          </div>


          <div>
            <p className="text-xs text-slate-400">
              Peak Prediction
            </p>

            <p className="mt-1 font-semibold text-slate-800">
              {event.replay.peakOffset}
            </p>
          </div>

        </div>
        {/* Signal Summary */}
        <div className="grid grid-cols-1 gap-3 border-b p-5 sm:grid-cols-3">

          <div className="rounded-lg border border-slate-200 bg-white p-3">
            <p className="text-xs text-slate-400">
              Peak Risk
            </p>

            <p className="mt-1 text-lg font-bold text-slate-900">
              {Math.round(peakRisk * 100)}%
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-3">
            <p className="text-xs text-slate-400">
              Peak IWV
            </p>

            <p className="mt-1 text-lg font-bold text-slate-900">
              {peakIWV}
            </p>
          </div>

          <div className="rounded-lg border border-slate-200 bg-white p-3">
            <p className="text-xs text-slate-400">
              Lowest CTT
            </p>

            <p className="mt-1 text-lg font-bold text-slate-900">
              {lowestCTT}°C
            </p>
          </div>

        </div>
        <div className="px-5 pt-4">
          <div className="rounded-lg bg-slate-50 p-4">

            <p className="text-sm font-semibold text-slate-800">
              Model Signal Interpretation
            </p>

            <p className="mt-1 text-sm leading-relaxed text-slate-500">
              The event replay shows how atmospheric moisture and cloud-top
              temperature evolved as the model risk increased. Peak risk reached{' '}
              <span className="font-semibold text-slate-700">
                {Math.round(peakRisk * 100)}%
              </span>{' '}
              with IWV reaching{' '}
              <span className="font-semibold text-slate-700">
                {peakIWV}
              </span>{' '}
              and CTT falling to{' '}
              <span className="font-semibold text-slate-700">
                {lowestCTT}°C
              </span>.
            </p>

          </div>
        </div>


        {/* Chart */}
        <div className="p-5">

          <h3 className="text-sm font-semibold text-slate-800">
            Atmospheric Signal Replay
          </h3>

          <p className="mt-1 text-xs text-slate-500">
            Evolution of IWV, cloud-top temperature and model risk around the event.
          </p>

          <div className="mt-5 h-87.5 w-full">

            <ResponsiveContainer width="100%" height="100%">

              <LineChart
                data={chartData}
                margin={{
                  top: 10,
                  right: 20,
                  left: 0,
                  bottom: 5,
                }}
              >

                <CartesianGrid
                  strokeDasharray="3 3"
                  vertical={false}
                />

                <XAxis
                  dataKey="time"
                  tick={{ fontSize: 12 }}
                />

                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 12 }}
                />

                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={[0, 100]}
                  tick={{ fontSize: 12 }}
                />

                <Tooltip />

                <Legend />

                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="iwv"
                  name="IWV"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />

                <Line
                  yAxisId="left"
                  type="monotone"
                  dataKey="ctt"
                  name="Cloud Top Temperature"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />

                <Line
                  yAxisId="right"
                  type="monotone"
                  dataKey="risk"
                  name="Risk Score (%)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />

              </LineChart>

            </ResponsiveContainer>

          </div>

        </div>

      </div>

    </div>
  );
}