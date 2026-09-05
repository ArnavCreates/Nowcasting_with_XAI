import React, { useState } from 'react';
import { mockHistorical } from '../mock/mockHistorical';
import MonthlyEventChart from '../components/charts/MonthlyEventChart';
import SeasonalHeatmap from '../components/charts/SeasonalHeatmap';
import HistoricalReplay from '../components/replay/HistoricalReplay';

const hazardLabels = {
  thunderstorm: 'Thunderstorm',
  cloudburst: 'Cloudburst',
  flashFlood: 'Flash Flood',
};

export default function HistoricalPage() {
  const [selectedEvent, setSelectedEvent] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [hazardFilter, setHazardFilter] = useState('all');
  const [outcomeFilter, setOutcomeFilter] = useState('all');

  const filteredEvents = mockHistorical.events.filter((event) => {
    const search = searchTerm.toLowerCase();

    const matchesSearch =
      event.id.toLowerCase().includes(search) ||
      event.region.toLowerCase().includes(search) ||
      event.state.toLowerCase().includes(search);

    const matchesHazard =
      hazardFilter === 'all' ||
      event.hazardType.toLowerCase() === hazardFilter.toLowerCase();

    const matchesOutcome =
      outcomeFilter === 'all' ||
      event.outcome === outcomeFilter;

    return matchesSearch && matchesHazard && matchesOutcome;
  });

  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-50/70 p-6">
      {/* Background Visual Layer: Technical Blueprint Dot Matrix & Depth Shadow */}
      <div
        className="pointer-events-none absolute inset-0 z-0 opacity-40"
        style={{
          backgroundImage: `radial-gradient(#94a3b8 1px, transparent 1px)`,
          backgroundSize: '24px 24px',
        }}
      />
      <div className="pointer-events-none absolute top-0 left-0 right-0 z-0 h-48 bg-gradient-to-b from-slate-200/50 via-slate-100/20 to-transparent" />

      {/* Main Content Layer */}
      <div className="relative z-10 space-y-6">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Historical Data</h2>
          <p className="mt-1 text-sm text-slate-500">
            Historical event analysis and model performance.
          </p>
        </div>

        {/* KPI Cards Grid */}
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {/* Model Precision */}
          <div className="rounded-xl border border-slate-200/80 bg-white/90 p-4 shadow-sm backdrop-blur-sm transition-shadow hover:shadow-md">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Model Precision
            </p>
            <div className="mt-2 flex items-end justify-between">
              <p className="text-2xl font-bold text-slate-900">
                {mockHistorical.kpis.modelPrecision.value}%
              </p>
              <span className="text-xs font-semibold text-emerald-600">
                +{mockHistorical.kpis.modelPrecision.yoyChange}% YoY
              </span>
            </div>
          </div>

          {/* Recall Rate */}
          <div className="rounded-xl border border-slate-200/80 bg-white/90 p-4 shadow-sm backdrop-blur-sm transition-shadow hover:shadow-md">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Recall Rate
            </p>
            <div className="mt-2 flex items-end justify-between">
              <p className="text-2xl font-bold text-slate-900">
                {mockHistorical.kpis.recallRate.value}%
              </p>
              <span className="text-xs font-semibold text-emerald-600">
                +{mockHistorical.kpis.recallRate.yoyChange}% YoY
              </span>
            </div>
          </div>

          {/* Average Lead Time */}
          <div className="rounded-xl border border-slate-200/80 bg-white/90 p-4 shadow-sm backdrop-blur-sm transition-shadow hover:shadow-md">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Avg Lead Time
            </p>
            <div className="mt-2 flex items-end justify-between">
              <p className="text-2xl font-bold text-slate-900">
                {mockHistorical.kpis.avgLeadTime.value}
                {mockHistorical.kpis.avgLeadTime.unit}
              </p>
              <span className="text-xs font-semibold text-emerald-600">
                +{mockHistorical.kpis.avgLeadTime.yoyChange} hrs YoY
              </span>
            </div>
          </div>

          {/* False Alarm Rate */}
          <div className="rounded-xl border border-slate-200/80 bg-white/90 p-4 shadow-sm backdrop-blur-sm transition-shadow hover:shadow-md">
            <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
              False Alarm Rate
            </p>
            <div className="mt-2 flex items-end justify-between">
              <p className="text-2xl font-bold text-slate-900">
                {mockHistorical.kpis.falseAlarmRate.value}%
              </p>
              <span className="text-xs font-semibold text-emerald-600">
                {mockHistorical.kpis.falseAlarmRate.yoyChange}% YoY
              </span>
            </div>
          </div>
        </div>

        {/* Monthly Event Archive Chart */}
        <div className="rounded-xl border border-slate-200/80 bg-white/90 p-5 shadow-sm backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-slate-900">Event Archive</h3>
          <p className="mt-1 text-sm text-slate-500">
            Monthly frequency of recorded severe-weather events.
          </p>
          <div className="mt-5">
            <MonthlyEventChart data={mockHistorical.monthlyFrequency} />
          </div>
        </div>

        {/* Seasonal Patterns Heatmap */}
        <div className="rounded-xl border border-slate-200/80 bg-white/90 p-5 shadow-sm backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-slate-900">Seasonal Patterns</h3>
          <p className="mt-1 text-sm text-slate-500">
            Historical intensity of weather hazards across different terrain types and months.
          </p>
          <div className="mt-5">
            <SeasonalHeatmap data={mockHistorical.seasonalIntensity} />
          </div>
        </div>

        {/* Historical Events Data Table */}
        <div className="rounded-xl border border-slate-200/80 bg-white/90 p-5 shadow-sm backdrop-blur-sm">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Historical Events</h3>
              <p className="text-sm text-slate-500">
                Archived severe-weather events and model outcomes.
              </p>
            </div>
            <span className="text-xs font-medium text-slate-400">
              Showing {filteredEvents.length} of {mockHistorical.events.length}
            </span>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-[1fr_180px_180px_auto]">
            {/* Search */}
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search event, region or state..."
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-slate-400 focus:ring-2 focus:ring-slate-100"
            />

            {/* Hazard filter */}
            <select
              value={hazardFilter}
              onChange={(e) => setHazardFilter(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 outline-none transition focus:border-slate-400"
            >
              <option value="all">All Hazards</option>
              <option value="thunderstorm">Thunderstorm</option>
              <option value="cloudburst">Cloudburst</option>
              <option value="flashFlood">Flash Flood</option>
            </select>

            {/* Outcome filter */}
            <select
              value={outcomeFilter}
              onChange={(e) => setOutcomeFilter(e.target.value)}
              className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-700 outline-none transition focus:border-slate-400"
            >
              <option value="all">All Outcomes</option>
              <option value="confirmed">Confirmed</option>
              <option value="falseAlarm">False Alarm</option>
              <option value="missed">Missed</option>
            </select>

            {/* Clear filters */}
            <button
              onClick={() => {
                setSearchTerm('');
                setHazardFilter('all');
                setOutcomeFilter('all');
              }}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition hover:bg-slate-50"
            >
              Clear
            </button>
          </div>

          {/* Table */}
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-225 text-left">
              <thead>
                <tr className="border-b border-slate-200 text-xs uppercase tracking-wide text-slate-500">
                  <th className="px-3 py-3 font-medium">Event</th>
                  <th className="px-3 py-3 font-medium">Date</th>
                  <th className="px-3 py-3 font-medium">Region</th>
                  <th className="px-3 py-3 font-medium">Hazard</th>
                  <th className="px-3 py-3 font-medium">Severity</th>
                  <th className="px-3 py-3 font-medium">Lead Time</th>
                  <th className="px-3 py-3 font-medium">Outcome</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100">
                {filteredEvents.length === 0 ? (
                  <tr>
                    <td colSpan="7" className="px-3 py-12 text-center">
                      <p className="text-sm font-medium text-slate-700">
                        No historical events found
                      </p>
                      <p className="mt-1 text-xs text-slate-400">
                        Try changing your search or filters.
                      </p>
                    </td>
                  </tr>
                ) : (
                  filteredEvents.map((event) => (
                    <tr
                      key={event.id}
                      onClick={() => setSelectedEvent(event)}
                      className="cursor-pointer transition-colors hover:bg-slate-50/80"
                    >
                      {/* Event ID */}
                      <td className="px-3 py-4">
                        <span className="font-medium text-slate-900">{event.id}</span>
                      </td>

                      {/* Date */}
                      <td className="px-3 py-4 text-sm text-slate-600">
                        {new Date(event.date).toLocaleDateString('en-IN', {
                          day: '2-digit',
                          month: 'short',
                          year: 'numeric',
                        })}
                      </td>

                      {/* Region */}
                      <td className="px-3 py-4">
                        <p className="text-sm font-medium text-slate-800">{event.region}</p>
                        <p className="text-xs text-slate-400">{event.state}</p>
                      </td>

                      {/* Hazard */}
                      <td className="px-3 py-4">
                        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700">
                          {hazardLabels[event.hazardType] || event.hazardType}
                        </span>
                      </td>

                      {/* Severity */}
                      <td className="px-3 py-4">
                        <span
                          className={`rounded-full px-2.5 py-1 text-xs font-semibold capitalize ${
                            event.severity === 'severe'
                              ? 'bg-red-100 text-red-700'
                              : event.severity === 'high'
                              ? 'bg-orange-100 text-orange-700'
                              : event.severity === 'moderate'
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-slate-100 text-slate-600'
                          }`}
                        >
                          {event.severity}
                        </span>
                      </td>

                      {/* Lead Time */}
                      <td className="px-3 py-4 text-sm font-medium text-slate-700">
                        {event.leadTime > 0 ? `${event.leadTime} hrs` : '—'}
                      </td>

                      {/* Outcome */}
                      <td className="px-3 py-4">
                        <span
                          className={`rounded-full px-2.5 py-1 text-xs font-semibold ${
                            event.outcome === 'confirmed'
                              ? 'bg-emerald-100 text-emerald-700'
                              : event.outcome === 'falseAlarm'
                              ? 'bg-amber-100 text-amber-700'
                              : 'bg-red-100 text-red-700'
                          }`}
                        >
                          {event.outcome === 'falseAlarm'
                            ? 'False Alarm'
                            : event.outcome === 'confirmed'
                            ? 'Confirmed'
                            : 'Missed'}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        {selectedEvent && (
          <HistoricalReplay
            event={selectedEvent}
            onClose={() => setSelectedEvent(null)}
          />
        )}
      </div>
    </div>
  );
}