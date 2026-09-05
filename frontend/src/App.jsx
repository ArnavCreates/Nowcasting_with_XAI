import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppLayout from './components/layout/AppLayout';
import LiveMapPage from './pages/LiveMapPage';
import RiskAnalysisPage from './pages/RiskAnalysisPage';
import AlertTickerPage from './pages/AlertTickerPage';
import XAIReportsPage from './pages/XAIReportsPage';
import SimulatorPage from './pages/SimulatorPage';
import HistoricalPage from './pages/HistoricalPage';
import './App.css';

// Initialize React Query client with dashboard defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route index element={<Navigate to="/live-map" replace />} />
            <Route path="/live-map" element={<LiveMapPage />} />
            <Route path="/risk-analysis" element={<RiskAnalysisPage />} />
            <Route path="/alert-ticker" element={<AlertTickerPage />} />
            <Route path="/xai-reports" element={<XAIReportsPage />} />
            <Route path="/simulator" element={<SimulatorPage />} />
            <Route path="/historical" element={<HistoricalPage />} />
            {/* Catch-all fallback */}
            <Route path="*" element={<Navigate to="/live-map" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}