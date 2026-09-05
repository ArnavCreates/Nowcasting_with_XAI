import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar.jsx';
import Header from './Header.jsx';
import EmergencyAlertBanner from './EmergencyAlertBanner.jsx';

export default function AppLayout() {
  const activeAlert = {
    severity: 'Severe',
    message: 'Flash Flood Risk in Solan District (85% Probability)',
  };

  return (
    <div className="flex h-screen bg-slate-50">
      <Sidebar />
      <div className="flex flex-1 flex-col overflow-hidden">
        <Header activeAlertCount={1} />
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
      <EmergencyAlertBanner alert={activeAlert} />
    </div>
  );
}