import { AlertTriangle, X } from 'lucide-react';
import { useState } from 'react';

export default function EmergencyAlertBanner({ alert }) {
  const [dismissed, setDismissed] = useState(false);

  if (!alert || dismissed) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex items-center justify-between gap-4 border-t border-red-200 bg-red-50 px-6 py-3 shadow-[0_-2px_8px_rgba(0,0,0,0.06)]">
      <div className="flex items-center gap-3">
        <AlertTriangle className="h-5 w-5 shrink-0 text-red-600" />
        <p className="text-sm font-medium text-red-800">
          <span className="font-bold uppercase tracking-wide">
            {alert.severity} Alert:
          </span>{' '}
          {alert.message}
        </p>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => setDismissed(true)}
          className="rounded-md bg-red-600 px-4 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700"
        >
          Acknowledge
        </button>
        <button
          onClick={() => setDismissed(true)}
          aria-label="Dismiss"
          className="rounded-md p-1.5 text-red-500 transition-colors hover:bg-red-100"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}