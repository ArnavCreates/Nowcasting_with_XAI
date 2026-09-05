import { NavLink } from 'react-router-dom';
import {
  Map,
  BarChart3,
  Bell,
  Brain,
  SlidersHorizontal,
  History,
  Settings,
  HelpCircle,
} from 'lucide-react';
import logo from "../../assets/IndraAI-Logo.png"

const navItems = [
  { to: '/live-map', label: 'Live Map', icon: Map },
  { to: '/risk-analysis', label: 'Risk Analysis', icon: BarChart3 },
  { to: '/alert-ticker', label: 'Alert Ticker', icon: Bell },
  { to: '/xai-reports', label: 'XAI Reports', icon: Brain },
  { to: '/simulator', label: 'Simulator', icon: SlidersHorizontal },
  { to: '/historical', label: 'Historical Data', icon: History },
];

export default function Sidebar() {
  return (
    <aside className="flex h-screen w-64 shrink-0 flex-col border-r border-slate-200 bg-white text-slate-700 shadow-sm relative overflow-hidden"
           style={{
              backgroundColor: '#ffffff',
              backgroundImage: 'radial-gradient(rgba(13, 148, 136, 0.2) 1.5px, transparent 1.5px)',
              backgroundSize: '20px 20px'
            }}
    >
      <div className="flex items-center gap-3">
        <a href="/" className="flex items-center transition-opacity hover:opacity-90">
          <img
            src={logo}
            alt="IndraAI Logo"
            className="h-35 w-auto object-contain"
          />
        </a>
      </div>

      <nav className="flex-1 space-y-1.5 overflow-y-auto px-3 py-4">
        {navItems.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              `flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium transition-all duration-200 ease-out transform ${
                isActive
                  ? 'bg-linear-to-r from-teal-50 to-cyan-50/60 text-teal-700 font-semibold border-l-4 border-teal-500 shadow-sm scale-[1.02]'
                  : 'text-slate-600 hover:bg-teal-50/50 hover:text-teal-700 hover:translate-x-1 hover:scale-[1.01] hover:shadow-[0_2px_8px_rgba(20,184,166,0.08)]'
              }`
            }
          >
            <Icon className="h-4 w-4 transition-transform duration-300 group-hover:scale-110" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-1 border-t border-slate-200 px-3 py-4">
        <NavLink
          to="/settings"
          className="flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-900"
        >
          <Settings className="h-4 w-4" />
          Settings
        </NavLink>
        <NavLink
          to="/support"
          className="flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-sm font-medium text-slate-600 transition-colors hover:bg-slate-50 hover:text-slate-900"
        >
          <HelpCircle className="h-4 w-4" />
          Support
        </NavLink>
      </div>

      <div className="flex items-center gap-3 border-t border-slate-200 px-5 py-4 bg-slate-50/50">
        <div className="h-8 w-8 rounded-full bg-teal-100 border border-teal-200 flex items-center justify-center text-teal-700 font-bold text-xs shadow-sm">
          CC
        </div>
        <div>
          <p className="text-sm font-medium text-slate-900">Command Center</p>
          <p className="text-xs text-slate-500">Disaster Mgmt India</p>
        </div>
      </div>
    </aside>
  );
}