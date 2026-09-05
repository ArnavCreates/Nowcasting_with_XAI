import React from 'react';
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Clock3,
  Siren,
  Smartphone,
} from 'lucide-react';

const hazardLabels = {
  thunderstorm: 'Thunderstorm',
  cloudburst: 'Cloudburst',
  flashFlood: 'Flash Flood',
};

const severityLabels = {
  severe: 'Severe',
  high: 'High',
  moderate: 'Moderate',
  low: 'Low',
};

const severityClasses = {
  severe: 'border-red-200 bg-red-50 text-red-700',
  high: 'border-orange-200 bg-orange-50 text-orange-700',
  moderate: 'border-yellow-200 bg-yellow-50 text-yellow-700',
  low: 'border-green-200 bg-green-50 text-green-700',
};

const statusClasses = {
  active: 'bg-red-50 text-red-700',
  acknowledged: 'bg-blue-50 text-blue-700',
  resolved: 'bg-green-50 text-green-700',
};

function formatTimestamp(timestamp) {
  return new Date(timestamp).toLocaleString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function AlertCard({ alert, onClick }) {
  return (
    <div
      onClick={onClick}
      className="cursor-pointer rounded-lg border bg-white p-5 shadow-sm transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <AlertTriangle
              size={18}
              className={
                alert.severity === 'severe'
                  ? 'text-red-600'
                  : alert.severity === 'high'
                    ? 'text-orange-600'
                    : alert.severity === 'moderate'
                      ? 'text-yellow-600'
                      : 'text-green-600'
              }
            />

            <span
              className={`rounded-md border px-2 py-1 text-xs font-semibold uppercase ${severityClasses[alert.severity]}`}
            >
              {severityLabels[alert.severity]}
            </span>

            <span
              className={`rounded-md px-2 py-1 text-xs font-medium capitalize ${statusClasses[alert.status]}`}
            >
              {alert.status}
            </span>
          </div>

          <h3 className="mt-3 text-lg font-semibold">
            {hazardLabels[alert.hazardType]}
          </h3>

          <p className="mt-1 text-sm text-slate-600">
            {alert.regionName}, {alert.state}
          </p>
        </div>

        <div className="text-right">
          <p className="text-xs text-slate-500">Alert ID</p>
          <p className="font-mono text-sm font-semibold">{alert.id}</p>
        </div>
      </div>

      <p className="mt-4 text-sm leading-6 text-slate-600">
        {alert.description}
      </p>

      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        <div className="rounded-md bg-slate-50 p-3">
          <p className="text-xs text-slate-500">Risk Score</p>
          <p className="mt-1 text-2xl font-bold text-slate-900">
            {alert.riskScore}%
          </p>
        </div>

        <div className="rounded-md bg-slate-50 p-3">
          <p className="text-xs text-slate-500">Model Confidence</p>
          <p className="mt-1 text-2xl font-bold text-slate-900">
            {alert.confidence}%
          </p>
        </div>
      </div>

      <div className="mt-5 border-t pt-4">
        <p className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-500">
          Notification Delivery
        </p>

        <div className="flex flex-wrap gap-4 text-sm">
          <span className="flex items-center gap-2">
            <Smartphone size={16} />
            SMS
            {alert.delivery.sms ? (
              <CheckCircle2 size={15} className="text-green-600" />
            ) : (
              <span className="text-slate-400">—</span>
            )}
          </span>

          <span className="flex items-center gap-2">
            <Bell size={16} />
            Push
            {alert.delivery.push ? (
              <CheckCircle2 size={15} className="text-green-600" />
            ) : (
              <span className="text-slate-400">—</span>
            )}
          </span>

          <span className="flex items-center gap-2">
            <Siren size={16} />
            Siren
            {alert.delivery.siren ? (
              <CheckCircle2 size={15} className="text-green-600" />
            ) : (
              <span className="text-slate-400">—</span>
            )}
          </span>
        </div>
      </div>

      <div className="mt-5 border-t pt-4">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Clock3 size={15} />
          Issued {formatTimestamp(alert.timestamp)}
        </div>
      </div>
    </div>
  );
}