import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  CheckCircle2,
  Clock3,
  MapPin,
  MessageSquare,
  Radio,
  X,
} from 'lucide-react';
import { thresholdRuleMeta } from '../../mock/mockThresholdConfig';

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
  severe: 'bg-red-50 text-red-700 border-red-200',
  high: 'bg-orange-50 text-orange-700 border-orange-200',
  moderate: 'bg-yellow-50 text-yellow-700 border-yellow-200',
  low: 'bg-green-50 text-green-700 border-green-200',
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

function DeliveryItem({ icon: Icon, label, enabled }) {
  return (
    <div className="flex items-center gap-2 rounded-lg border bg-slate-50 px-3 py-2 text-sm">
      <Icon size={16} className="text-slate-500" />

      <span>{label}</span>

      {enabled ? (
        <CheckCircle2 size={15} className="ml-auto text-green-600" />
      ) : (
        <span className="ml-auto text-xs text-slate-400">
          Not sent
        </span>
      )}
    </div>
  );
}

export default function AlertDetails({ alert, onClose }) {
  return (
    <AnimatePresence>
      {alert && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="max-h-[90vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white shadow-2xl"
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ duration: 0.18 }}
            onClick={(event) => event.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-start justify-between border-b p-5">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`rounded-md border px-2 py-1 text-xs font-semibold uppercase ${severityClasses[alert.severity]}`}
                  >
                    {severityLabels[alert.severity]}
                  </span>

                  <span className="font-mono text-xs text-slate-400">
                    {alert.id}
                  </span>
                </div>

                <h3 className="mt-3 text-xl font-semibold text-slate-900">
                  {hazardLabels[alert.hazardType]}
                </h3>

                <div className="mt-1 flex items-center gap-1.5 text-sm text-slate-500">
                  <MapPin size={15} />
                  {alert.regionName}, {alert.state}
                </div>
              </div>

              <button
                type="button"
                onClick={onClose}
                className="rounded-md p-2 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700"
                aria-label="Close alert details"
              >
                <X size={20} />
              </button>
            </div>

            <div className="space-y-6 p-5">
              {/* Risk metrics */}
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border bg-slate-50 p-4">
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Risk Score
                  </p>

                  <div className="mt-2 flex items-end justify-between">
                    <p className="text-3xl font-bold text-slate-900">
                      {alert.riskScore}%
                    </p>

                    <span className="text-xs text-slate-500">
                      Current risk
                    </span>
                  </div>

                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                    <motion.div
                      className="h-full rounded-full bg-red-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${alert.riskScore}%` }}
                      transition={{ duration: 0.6, delay: 0.15 }}
                    />
                  </div>
                </div>

                <div className="rounded-lg border bg-slate-50 p-4">
                  <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
                    Model Confidence
                  </p>

                  <div className="mt-2 flex items-end justify-between">
                    <p className="text-3xl font-bold text-slate-900">
                      {alert.confidence}%
                    </p>

                    <span className="text-xs text-slate-500">
                      Prediction confidence
                    </span>
                  </div>

                  <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                    <motion.div
                      className="h-full rounded-full bg-sky-500"
                      initial={{ width: 0 }}
                      animate={{ width: `${alert.confidence}%` }}
                      transition={{ duration: 0.6, delay: 0.2 }}
                    />
                  </div>
                </div>
              </div>

              {/* Alert information */}
              <section>
                <h4 className="text-sm font-semibold text-slate-900">
                  Alert Information
                </h4>

                <div className="mt-3 grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
                  <div>
                    <span className="text-slate-500">Hazard</span>
                    <p className="mt-0.5 font-medium">
                      {hazardLabels[alert.hazardType]}
                    </p>
                  </div>

                  <div>
                    <span className="text-slate-500">Status</span>
                    <p className="mt-0.5 font-medium capitalize">
                      {alert.status}
                    </p>
                  </div>

                  <div>
                    <span className="text-slate-500">Triggered Threshold</span>
                    <p className="mt-0.5 font-medium">
                      {alert.triggeredThresholdId
                        ? thresholdRuleMeta[alert.triggeredThresholdId]?.label ?? alert.triggeredThresholdId
                        : 'None'}
                    </p>
                  </div>

                  <div>
                    <span className="text-slate-500">Issued</span>
                    <p className="mt-0.5 font-medium">
                      {formatTimestamp(alert.timestamp)}
                    </p>
                  </div>
                </div>
              </section>

              {/* Description */}
              <section>
                <h4 className="text-sm font-semibold text-slate-900">
                  Meteorological Assessment
                </h4>

                <p className="mt-2 rounded-lg border bg-slate-50 p-4 text-sm leading-6 text-slate-600">
                  {alert.description}
                </p>
              </section>

              {/* Delivery */}
              <section>
                <h4 className="text-sm font-semibold text-slate-900">
                  Notification Delivery
                </h4>

                <div className="mt-3 grid gap-2 sm:grid-cols-3">
                  <DeliveryItem
                    icon={MessageSquare}
                    label="SMS"
                    enabled={alert.delivery.sms}
                  />

                  <DeliveryItem
                    icon={Bell}
                    label="Push"
                    enabled={alert.delivery.push}
                  />

                  <DeliveryItem
                    icon={Radio}
                    label="Siren"
                    enabled={alert.delivery.siren}
                  />
                </div>
              </section>

              {/* Timeline */}
              <section>
                <h4 className="text-sm font-semibold text-slate-900">
                  Response Timeline
                </h4>

                <div className="mt-4">
                  {alert.statusTimeline.map((item, index) => {
                    const isLast =
                      index === alert.statusTimeline.length - 1;

                    return (
                      <div
                        key={`${alert.id}-timeline-${index}`}
                        className="flex gap-3"
                      >
                        <div className="flex flex-col items-center">
                          <div
                            className={`mt-1 h-3 w-3 rounded-full ${
                              item.time
                                ? 'bg-sky-500'
                                : 'border-2 border-slate-300 bg-white'
                            }`}
                          />

                          {!isLast && (
                            <div className="mt-1 h-full min-h-8 w-px bg-slate-200" />
                          )}
                        </div>

                        <div className="pb-5">
                          <p className="text-sm font-medium text-slate-800">
                            {item.label}
                          </p>

                          <div className="mt-1 flex items-center gap-1.5 text-xs text-slate-500">
                            <Clock3 size={13} />

                            {item.time
                              ? formatTimestamp(item.time)
                              : 'Awaiting update'}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </section>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}