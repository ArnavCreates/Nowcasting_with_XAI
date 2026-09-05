import { Bell, Settings } from 'lucide-react';
import { useWeatherStore } from '../../store/useWeatherStore';

export default function Header({ activeAlertCount = 0 }) {
  const isConnectedToWS = useWeatherStore((state) => state.isConnectedToWS);

  return (
    <header
      className="flex h-16 shrink-0 items-center justify-between border-b border-slate-200 bg-white px-6 relative"
      style={{
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        backgroundImage: 'radial-gradient(rgba(14, 165, 233, 0.2) 1.5px, transparent 1.5px)',
        backgroundSize: '20px 20px',
      }}
    >
      <div className="flex items-center gap-4">
        <input
          type="text"
          placeholder="Search region, coordinates..."
          className="w-72 rounded-md border border-slate-200 bg-slate-50 px-3 py-1.5 text-sm text-slate-700 placeholder:text-slate-400 focus:border-sky-400 focus:outline-none focus:ring-1 focus:ring-sky-400"
        />
      </div>

      <div className="flex items-center gap-4">
        {/* Live-connection badge */}
        <div
          className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-[11px] font-bold transition-colors ${
            isConnectedToWS
              ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
              : 'border-slate-200 bg-slate-100 text-slate-400'
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isConnectedToWS ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'
            }`}
          />
          {isConnectedToWS ? 'Live Nowcast' : 'Disconnected'}
        </div>

        <button
          aria-label="Notifications"
          className="relative rounded-md p-2 text-slate-500 hover:bg-slate-50"
        >
          <Bell className="h-5 w-5" />
          {activeAlertCount > 0 && (
            <>
              <span className="absolute -right-0.5 -top-0.5 flex h-4 w-4 rounded-full bg-red-400 opacity-75 animate-ping"></span>
              <span className="absolute -right-0.5 -top-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-red-600 text-[10px] font-bold text-white">
                {activeAlertCount}
              </span>
            </>
          )}
        </button>
        <button
          aria-label="Settings"
          className="rounded-md p-2 text-slate-500 hover:bg-slate-50"
        >
          <Settings className="h-5 w-5" />
        </button>
      </div>
    </header>
  );
}