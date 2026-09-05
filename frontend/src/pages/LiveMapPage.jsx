import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { useWeatherStore } from '../store/useWeatherStore';
import { nationalHazardGrids, nationalCompositeGrid } from '../mock/mockNationalGrid';
import { getRegionThresholds } from '../utils/thresholdEvaluator';
import { Globe, RotateCcw, Flame, Layers } from 'lucide-react';

const INDIA_CENTER = [78.9629, 22.5937];
const INDIA_ZOOM = 4.5;

const OSM_MAP_STYLE = {
  version: 8,
  sources: {
    'osm-tiles': {
      type: 'raster',
      tiles: [
        'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png',
      ],
      tileSize: 256,
      attribution: '&copy; OpenStreetMap contributors',
    },
  },
  layers: [{ id: 'osm-tiles-layer', type: 'raster', source: 'osm-tiles', minzoom: 0, maxzoom: 19 }],
};

const HAZARD_TABS = [
  { id: 'all', label: 'All' },
  { id: 'thunderstorm', label: 'Thunderstorms' },
  { id: 'cloudburst', label: 'Cloudbursts' },
  { id: 'flashFlood', label: 'Flash Floods' },
];

function gridToGeoJSON(grid) {
  return {
    type: 'FeatureCollection',
    features: grid.map((p) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [p.lng, p.lat] },
      properties: { value: p.value },
    })),
  };
}

export const LiveMapPage = () => {
  const regions = useWeatherStore((state) => state.regions);
  const selectedRegion = useWeatherStore((state) => state.selectedRegion);
  const setSelectedRegion = useWeatherStore((state) => state.setSelectedRegion);
  const forecastHorizon = useWeatherStore((state) => state.forecastHorizon);
  const setForecastHorizonValue = useWeatherStore((state) => state.setForecastHorizonValue);
  const mapLayers = useWeatherStore((state) => state.mapLayers);
  const setActiveHazard = useWeatherStore((state) => state.setActiveHazard);
  const currentRiskData = useWeatherStore((state) => state.currentRiskData);
  const fetchRiskAnalysis = useWeatherStore((state) => state.fetchRiskAnalysis);


  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const markersRef = useRef([]);
  const [showHeatmap, setShowHeatmap] = useState(true);

  // Fetch inference for the selected region whenever it (or the lead-time
  // scrubber) changes — this is the live call, mock now, real later.
  useEffect(() => {
    if (selectedRegion) fetchRiskAnalysis();
  }, [selectedRegion?.id, forecastHorizon.value]);

  const activeThresholds = selectedRegion ? getRegionThresholds(selectedRegion.id) : null;

  // Initialize map once
  useEffect(() => {
    if (!mapContainerRef.current || mapInstanceRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: OSM_MAP_STYLE,
      center: INDIA_CENTER,
      zoom: INDIA_ZOOM,
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
    mapInstanceRef.current = map;
    map.on('load', () => map.resize());

    return () => {
      mapInstanceRef.current?.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // National hazard heatmap — swaps source data based on the active hazard tab
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const activeGrid =
      !mapLayers.activeHazard || mapLayers.activeHazard === 'all'
        ? nationalCompositeGrid
        : nationalHazardGrids[mapLayers.activeHazard] || nationalCompositeGrid;

    const geojsonData = gridToGeoJSON(activeGrid);

    const applyLayers = () => {
      if (map.getSource('national-hazard-source')) {
        map.getSource('national-hazard-source').setData(geojsonData);
      } else {
        map.addSource('national-hazard-source', { type: 'geojson', data: geojsonData });
        map.addLayer({
          id: 'national-hazard-heatmap',
          type: 'heatmap',
          source: 'national-hazard-source',
          paint: {
            'heatmap-weight': ['interpolate', ['linear'], ['get', 'value'], 0, 0, 100, 1],
            'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
            'heatmap-color': [
              'interpolate', ['linear'], ['heatmap-density'],
              0, 'rgba(0, 0, 0, 0)',
              0.2, 'rgba(16, 185, 129, 0.7)',
              0.5, 'rgba(245, 158, 11, 0.85)',
              0.8, 'rgba(239, 68, 68, 0.95)',
              1.0, 'rgba(159, 18, 57, 1)',
            ],
            'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 2, 25, 6, 55, 10, 90],
            'heatmap-opacity': showHeatmap ? 0.85 : 0,
          },
        });
      }
    };

    if (map.isStyleLoaded()) applyLayers();
    else map.once('load', applyLayers);
  }, [mapLayers.activeHazard, showHeatmap]);

  // Region markers — one per region in the unified store list
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const renderMarkers = () => {
      markersRef.current.forEach((m) => m.remove());
      markersRef.current = [];

      regions.forEach((region) => {
        const isSelected = selectedRegion?.id === region.id;
        const color = region.riskScore > 70 ? '#e11d48' : region.riskScore > 40 ? '#d97706' : '#10b981';
        const size = isSelected ? 38 : 28;

        const el = document.createElement('div');
        el.style.cssText = `position: relative; display: flex; align-items: center; justify-content: center; cursor: pointer; width: ${size}px; height: ${size}px;`;

        if (isSelected) {
          const pulse = document.createElement('div');
          pulse.style.cssText = `position: absolute; width: ${size + 16}px; height: ${size + 16}px; background-color: ${color}44; border-radius: 50%; animation: ping 1.5s cubic-bezier(0,0,0.2,1) infinite;`;
          el.appendChild(pulse);
        }

        const icon = document.createElement('div');
        icon.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="${color}" stroke="#fff" stroke-width="2" style="filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.3));"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3" fill="#fff"/></svg>`;
        el.appendChild(icon);

        el.addEventListener('click', (e) => {
          e.stopPropagation();
          setSelectedRegion(region.id);
        });

        markersRef.current.push(new maplibregl.Marker({ element: el }).setLngLat([region.lng, region.lat]).addTo(map));
      });
    };

    if (map.isStyleLoaded()) renderMarkers();
    else map.once('load', renderMarkers);
  }, [regions, selectedRegion]);

  // Fly to selected region
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !selectedRegion) return;
    map.flyTo({ center: [selectedRegion.lng, selectedRegion.lat], zoom: 8.5, essential: true, speed: 1.2 });
  }, [selectedRegion?.id]);

  return (
    <div className="relative w-full h-full min-h-[calc(100vh-4rem)] bg-slate-100 overflow-hidden">
      <div ref={mapContainerRef} className="absolute inset-0 w-full h-full z-0" />

      {/* Top bar: hazard tabs + lead-time scrubber */}
      <div className="absolute top-4 left-4 right-4 z-10 flex flex-col gap-3 pointer-events-none">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-1 bg-white/95 backdrop-blur-md p-1.5 rounded-2xl border border-slate-200/80 shadow-lg pointer-events-auto">
            {HAZARD_TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveHazard(tab.id)}
                className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all ${
                  (mapLayers.activeHazard || 'all') === tab.id
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 bg-white/95 backdrop-blur-md px-3.5 py-2.5 rounded-2xl border border-slate-200/80 shadow-lg pointer-events-auto">
            <span className="text-[11px] font-bold text-slate-500">Lead-Time:</span>
            <input
              type="range"
              min={forecastHorizon.min}
              max={forecastHorizon.max}
              step={1}
              value={forecastHorizon.value}
              onChange={(e) => setForecastHorizonValue(Number(e.target.value))}
              className="w-24 accent-blue-600 cursor-pointer"
            />
            <span className="text-xs font-mono font-bold text-blue-700">+{forecastHorizon.value}h</span>
          </div>
        </div>

        <div className="flex items-center justify-between gap-3">
          <div className="bg-white/95 backdrop-blur-md px-4 py-3 rounded-2xl border border-slate-200/80 shadow-lg pointer-events-auto flex items-center gap-3">
            <div className="p-2 bg-blue-50 border border-blue-200 rounded-xl text-blue-600">
              <Globe className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-900">National Hazard Overview — India</h1>
              <p className="text-[11px] text-slate-500">2–6 hour nowcast lead time · live target zone analytics</p>
            </div>
          </div>

          <div className="flex items-center gap-2 pointer-events-auto">
            <button
              onClick={() => setShowHeatmap(!showHeatmap)}
              className={`flex items-center gap-2 px-3.5 py-2.5 rounded-xl border text-xs font-bold shadow-lg transition-all active:scale-95 ${
                showHeatmap ? 'bg-rose-500 text-white border-rose-600' : 'bg-white/95 text-slate-700 border-slate-200'
              }`}
            >
              <Flame className={`w-4 h-4 ${showHeatmap ? 'text-white' : 'text-rose-500'}`} />
              <span>{showHeatmap ? 'Heatmap Active' : 'Show Heatmap'}</span>
            </button>
            <button
              onClick={() => {
                setSelectedRegion(null);
                mapInstanceRef.current?.flyTo({ center: INDIA_CENTER, zoom: INDIA_ZOOM });
              }}
              className="flex items-center gap-2 px-3.5 py-2.5 bg-white/95 backdrop-blur-md hover:bg-slate-100 text-slate-700 rounded-xl border border-slate-200/90 shadow-lg text-xs font-bold transition-all active:scale-95"
            >
              <RotateCcw className="w-4 h-4 text-blue-600" />
              <span>Reset Overview</span>
            </button>
          </div>
        </div>
      </div>

      {/* Selected region detail card */}
      {selectedRegion && (
        <div className="absolute bottom-6 left-6 z-10 w-84 bg-white/95 backdrop-blur-md p-5 rounded-2xl border border-slate-200 shadow-xl space-y-3">
          <div className="flex justify-between items-start border-b border-slate-100 pb-2">
            <div>
              <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider">Selected Region</span>
              <h2 className="text-lg font-extrabold text-slate-900">{selectedRegion.name}</h2>
              <p className="text-xs text-slate-500">{selectedRegion.state}</p>
            </div>
            <div className="text-right">
              <span className="text-2xl font-black text-rose-600 font-mono">
                {currentRiskData?.riskScore ?? selectedRegion.riskScore}
              </span>
              <span className="text-[10px] text-slate-400 block font-semibold">/ 100 Risk (+{forecastHorizon.value}h)</span>
            </div>
          </div>

          {activeThresholds && (
            <div className="space-y-1.5">
              {activeThresholds.thresholds.map((rule) => (
                <div
                  key={rule.id}
                  className={`flex items-center justify-between rounded-lg px-2.5 py-1.5 text-[11px] ${
                    rule.isMet ? 'bg-red-50 text-red-700' : 'bg-slate-50 text-slate-500'
                  }`}
                >
                  <span className="font-semibold">{rule.label}</span>
                  <span className="font-bold uppercase">{rule.statusText}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {/* RIGHT: Telemetry Feeds */}
      {selectedRegion && currentRiskData?.metrics && (
        <div className="absolute bottom-1 right-1 z-10 w-72 bg-white/95 backdrop-blur-md p-5 rounded-2xl border border-slate-200 shadow-xl space-y-3">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-2">
            <Layers className="w-4 h-4 text-blue-600" />
            <h3 className="text-sm font-bold text-slate-900">Telemetry Feeds</h3>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(currentRiskData.metrics).map(([key, val]) => (
              <div key={key} className="p-2.5 bg-slate-50 rounded-lg border border-slate-100">
                <span className="text-[10px] text-slate-400 block font-medium capitalize">
                  {key.replace(/([A-Z])/g, ' $1')}
                </span>
                <span className="font-bold text-slate-800">{val}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default LiveMapPage;

// import React, { useEffect, useRef, useState } from 'react';
// import * as maplibregl from 'maplibre-gl';
// import 'maplibre-gl/dist/maplibre-gl.css';
// import { useQuery } from '@tanstack/react-query';
// import { useWeatherStore } from '../store/useWeatherStore';
// import { nationalHazardGrids, nationalCompositeGrid } from '../mock/mockNationalGrid';
// import { getRegionThresholds } from '../utils/thresholdEvaluator';
// import { fetchAlerts } from '../services/alertingService';
// import AlertDetails from '../components/alerts/AlertDetails';
// import { Globe, RotateCcw, Flame, Layers, AlertTriangle } from 'lucide-react';

// const INDIA_CENTER = [78.9629, 22.5937];
// const INDIA_ZOOM = 4.5;

// const OSM_MAP_STYLE = {
//   version: 8,
//   sources: {
//     'osm-tiles': {
//       type: 'raster',
//       tiles: [
//         'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png',
//         'https://b.tile.openstreetmap.org/{z}/{x}/{y}.png',
//         'https://c.tile.openstreetmap.org/{z}/{x}/{y}.png',
//       ],
//       tileSize: 256,
//       attribution: '&copy; OpenStreetMap contributors',
//     },
//   },
//   layers: [{ id: 'osm-tiles-layer', type: 'raster', source: 'osm-tiles', minzoom: 0, maxzoom: 19 }],
// };

// const HAZARD_TABS = [
//   { id: 'all', label: 'All' },
//   { id: 'thunderstorm', label: 'Thunderstorms' },
//   { id: 'cloudburst', label: 'Cloudbursts' },
//   { id: 'flashFlood', label: 'Flash Floods' },
// ];

// const SEVERITY_DOT = {
//   severe: 'bg-red-500',
//   high: 'bg-orange-500',
//   moderate: 'bg-yellow-500',
//   low: 'bg-emerald-500',
// };

// const HAZARD_LABEL = {
//   thunderstorm: 'Thunderstorm',
//   cloudburst: 'Cloudburst',
//   flashFlood: 'Flash Flood',
// };

// function gridToGeoJSON(grid) {
//   return {
//     type: 'FeatureCollection',
//     features: grid.map((p) => ({
//       type: 'Feature',
//       geometry: { type: 'Point', coordinates: [p.lng, p.lat] },
//       properties: { value: p.value },
//     })),
//   };
// }

// // Compact alert row for the map-overlay panel — deliberately smaller than
// // the full AlertCard used on the Alert Ticker page, since this sits inside
// // a floating panel over the map rather than a full-width page list.
// function CompactAlertRow({ alert, onClick }) {
//   return (
//     <button
//       onClick={onClick}
//       className="w-full border-b border-slate-100 px-4 py-2.5 text-left transition-colors last:border-b-0 hover:bg-slate-50"
//     >
//       <div className="flex items-center gap-2">
//         <span className={`h-2 w-2 shrink-0 rounded-full ${SEVERITY_DOT[alert.severity] || 'bg-slate-400'}`} />
//         <span className="text-xs font-bold text-slate-800">
//           {HAZARD_LABEL[alert.hazardType] || alert.hazardType}
//         </span>
//         <span className="ml-auto shrink-0 font-mono text-[10px] text-slate-400">
//           {alert.confidence}%
//         </span>
//       </div>
//       <p className="mt-0.5 truncate text-[11px] font-medium text-slate-600">
//         {alert.regionName}, {alert.state}
//       </p>
//     </button>
//   );
// }

// export const LiveMapPage = () => {
//   const regions = useWeatherStore((state) => state.regions);
//   const selectedRegion = useWeatherStore((state) => state.selectedRegion);
//   const setSelectedRegion = useWeatherStore((state) => state.setSelectedRegion);
//   const forecastHorizon = useWeatherStore((state) => state.forecastHorizon);
//   const setForecastHorizonValue = useWeatherStore((state) => state.setForecastHorizonValue);
//   const mapLayers = useWeatherStore((state) => state.mapLayers);
//   const setActiveHazard = useWeatherStore((state) => state.setActiveHazard);
//   const currentRiskData = useWeatherStore((state) => state.currentRiskData);
//   const fetchRiskAnalysis = useWeatherStore((state) => state.fetchRiskAnalysis);

//   const mapContainerRef = useRef(null);
//   const mapInstanceRef = useRef(null);
//   const markersRef = useRef([]);
//   const [showHeatmap, setShowHeatmap] = useState(true);
//   const [selectedMapAlert, setSelectedMapAlert] = useState(null);

//   // Region-aware alert feed for the floating panel below:
//   //  - selectedRegion set  -> only that region's alerts (city view)
//   //  - selectedRegion null -> every region's alerts (India-wide view,
//   //    reached via the "Reset Overview" button, which already clears
//   //    selectedRegion to null)
//   // Polls every 5s so switching between city/national view and watching
//   // an alert escalate both work live, without a page reload.
//   const {
//     data: mapAlerts = [],
//     isFetching: alertsFetching,
//   } = useQuery({
//     queryKey: ['map-alerts', selectedRegion?.id ?? 'all'],
//     queryFn: () => fetchAlerts(selectedRegion?.id),
//     refetchInterval: 5000,
//     refetchIntervalInBackground: true,
//   });

//   // Fetch inference for the selected region whenever it (or the lead-time
//   // scrubber) changes — this is the live call, mock now, real later.
//   useEffect(() => {
//     if (selectedRegion) fetchRiskAnalysis();
//   }, [selectedRegion?.id, forecastHorizon.value]);

//   const activeThresholds = selectedRegion ? getRegionThresholds(selectedRegion.id) : null;

//   // Initialize map once
//   useEffect(() => {
//     if (!mapContainerRef.current || mapInstanceRef.current) return;

//     const map = new maplibregl.Map({
//       container: mapContainerRef.current,
//       style: OSM_MAP_STYLE,
//       center: INDIA_CENTER,
//       zoom: INDIA_ZOOM,
//     });

//     map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right');
//     mapInstanceRef.current = map;
//     map.on('load', () => map.resize());

//     return () => {
//       mapInstanceRef.current?.remove();
//       mapInstanceRef.current = null;
//     };
//   }, []);

//   // National hazard heatmap — swaps source data based on the active hazard tab
//   useEffect(() => {
//     const map = mapInstanceRef.current;
//     if (!map) return;

//     const activeGrid =
//       !mapLayers.activeHazard || mapLayers.activeHazard === 'all'
//         ? nationalCompositeGrid
//         : nationalHazardGrids[mapLayers.activeHazard] || nationalCompositeGrid;

//     const geojsonData = gridToGeoJSON(activeGrid);

//     const applyLayers = () => {
//       if (map.getSource('national-hazard-source')) {
//         map.getSource('national-hazard-source').setData(geojsonData);
//       } else {
//         map.addSource('national-hazard-source', { type: 'geojson', data: geojsonData });
//         map.addLayer({
//           id: 'national-hazard-heatmap',
//           type: 'heatmap',
//           source: 'national-hazard-source',
//           paint: {
//             'heatmap-weight': ['interpolate', ['linear'], ['get', 'value'], 0, 0, 100, 1],
//             'heatmap-intensity': ['interpolate', ['linear'], ['zoom'], 0, 1, 9, 3],
//             'heatmap-color': [
//               'interpolate', ['linear'], ['heatmap-density'],
//               0, 'rgba(0, 0, 0, 0)',
//               0.2, 'rgba(16, 185, 129, 0.7)',
//               0.5, 'rgba(245, 158, 11, 0.85)',
//               0.8, 'rgba(239, 68, 68, 0.95)',
//               1.0, 'rgba(159, 18, 57, 1)',
//             ],
//             'heatmap-radius': ['interpolate', ['linear'], ['zoom'], 2, 25, 6, 55, 10, 90],
//             'heatmap-opacity': showHeatmap ? 0.85 : 0,
//           },
//         });
//       }
//     };

//     if (map.isStyleLoaded()) applyLayers();
//     else map.once('load', applyLayers);
//   }, [mapLayers.activeHazard, showHeatmap]);

//   // Region markers — one per region in the unified store list
//   useEffect(() => {
//     const map = mapInstanceRef.current;
//     if (!map) return;

//     const renderMarkers = () => {
//       markersRef.current.forEach((m) => m.remove());
//       markersRef.current = [];

//       regions.forEach((region) => {
//         const isSelected = selectedRegion?.id === region.id;
//         const color = region.riskScore > 70 ? '#e11d48' : region.riskScore > 40 ? '#d97706' : '#10b981';
//         const size = isSelected ? 38 : 28;

//         const el = document.createElement('div');
//         el.style.cssText = `position: relative; display: flex; align-items: center; justify-content: center; cursor: pointer; width: ${size}px; height: ${size}px;`;

//         if (isSelected) {
//           const pulse = document.createElement('div');
//           pulse.style.cssText = `position: absolute; width: ${size + 16}px; height: ${size + 16}px; background-color: ${color}44; border-radius: 50%; animation: ping 1.5s cubic-bezier(0,0,0.2,1) infinite;`;
//           el.appendChild(pulse);
//         }

//         const icon = document.createElement('div');
//         icon.innerHTML = `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="${color}" stroke="#fff" stroke-width="2" style="filter: drop-shadow(0px 4px 6px rgba(0,0,0,0.3));"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3" fill="#fff"/></svg>`;
//         el.appendChild(icon);

//         el.addEventListener('click', (e) => {
//           e.stopPropagation();
//           setSelectedRegion(region.id);
//         });

//         markersRef.current.push(new maplibregl.Marker({ element: el }).setLngLat([region.lng, region.lat]).addTo(map));
//       });
//     };

//     if (map.isStyleLoaded()) renderMarkers();
//     else map.once('load', renderMarkers);
//   }, [regions, selectedRegion]);

//   // Fly to selected region
//   useEffect(() => {
//     const map = mapInstanceRef.current;
//     if (!map || !selectedRegion) return;
//     map.flyTo({ center: [selectedRegion.lng, selectedRegion.lat], zoom: 8.5, essential: true, speed: 1.2 });
//   }, [selectedRegion?.id]);

//   return (
//     <div className="relative w-full h-full min-h-[calc(100vh-4rem)] bg-slate-100 overflow-hidden">
//       <div ref={mapContainerRef} className="absolute inset-0 w-full h-full z-0" />

//       {/* Top bar: hazard tabs + lead-time scrubber */}
//       <div className="absolute top-4 left-4 right-4 z-10 flex flex-col gap-3 pointer-events-none">
//         <div className="flex flex-wrap items-center justify-between gap-3">
//           <div className="flex items-center gap-1 bg-white/95 backdrop-blur-md p-1.5 rounded-2xl border border-slate-200/80 shadow-lg pointer-events-auto">
//             {HAZARD_TABS.map((tab) => (
//               <button
//                 key={tab.id}
//                 onClick={() => setActiveHazard(tab.id)}
//                 className={`px-3.5 py-2 rounded-xl text-xs font-bold transition-all ${
//                   (mapLayers.activeHazard || 'all') === tab.id
//                     ? 'bg-blue-600 text-white shadow-sm'
//                     : 'text-slate-600 hover:bg-slate-100'
//                 }`}
//               >
//                 {tab.label}
//               </button>
//             ))}
//           </div>

//           <div className="flex items-center gap-2 bg-white/95 backdrop-blur-md px-3.5 py-2.5 rounded-2xl border border-slate-200/80 shadow-lg pointer-events-auto">
//             <span className="text-[11px] font-bold text-slate-500">Lead-Time:</span>
//             <input
//               type="range"
//               min={forecastHorizon.min}
//               max={forecastHorizon.max}
//               step={1}
//               value={forecastHorizon.value}
//               onChange={(e) => setForecastHorizonValue(Number(e.target.value))}
//               className="w-24 accent-blue-600 cursor-pointer"
//             />
//             <span className="text-xs font-mono font-bold text-blue-700">+{forecastHorizon.value}h</span>
//           </div>
//         </div>

//         <div className="flex items-center justify-between gap-3">
//           <div className="bg-white/95 backdrop-blur-md px-4 py-3 rounded-2xl border border-slate-200/80 shadow-lg pointer-events-auto flex items-center gap-3">
//             <div className="p-2 bg-blue-50 border border-blue-200 rounded-xl text-blue-600">
//               <Globe className="w-5 h-5" />
//             </div>
//             <div>
//               <h1 className="text-sm font-bold text-slate-900">National Hazard Overview — India</h1>
//               <p className="text-[11px] text-slate-500">2–6 hour nowcast lead time · live target zone analytics</p>
//             </div>
//           </div>

//           <div className="flex items-center gap-2 pointer-events-auto">
//             <button
//               onClick={() => setShowHeatmap(!showHeatmap)}
//               className={`flex items-center gap-2 px-3.5 py-2.5 rounded-xl border text-xs font-bold shadow-lg transition-all active:scale-95 ${
//                 showHeatmap ? 'bg-rose-500 text-white border-rose-600' : 'bg-white/95 text-slate-700 border-slate-200'
//               }`}
//             >
//               <Flame className={`w-4 h-4 ${showHeatmap ? 'text-white' : 'text-rose-500'}`} />
//               <span>{showHeatmap ? 'Heatmap Active' : 'Show Heatmap'}</span>
//             </button>
//             <button
//               onClick={() => {
//                 setSelectedRegion(null);
//                 mapInstanceRef.current?.flyTo({ center: INDIA_CENTER, zoom: INDIA_ZOOM });
//               }}
//               className="flex items-center gap-2 px-3.5 py-2.5 bg-white/95 backdrop-blur-md hover:bg-slate-100 text-slate-700 rounded-xl border border-slate-200/90 shadow-lg text-xs font-bold transition-all active:scale-95"
//             >
//               <RotateCcw className="w-4 h-4 text-blue-600" />
//               <span>Reset Overview</span>
//             </button>
//           </div>
//         </div>
//       </div>

//       <div className="absolute top-32 right-4 z-10 bg-white/95 backdrop-blur-md px-3.5 py-2.5 rounded-xl border border-slate-200/90 shadow-lg space-y-1.5 text-xs">
//         <div className="flex items-center gap-1.5 text-[11px] font-bold text-slate-700">
//           <Layers className="w-3.5 h-3.5 text-blue-600" />
//           <span>Risk Heat Intensity</span>
//         </div>
//         <div className="h-2 w-36 rounded-full bg-linear-to-r from-emerald-400 via-amber-400 to-rose-600" />
//         <div className="flex justify-between text-[9px] font-bold text-slate-400">
//           <span>Low (0)</span><span>Moderate</span><span>Severe (100)</span>
//         </div>
//       </div>

//       {/* Region-aware alerts panel: this region's alerts when a city is
//           selected, ALL regions' alerts when in India-wide view. */}
//       <div className="absolute top-32 left-4 z-10 max-h-[45vh] w-72 overflow-y-auto rounded-2xl border border-slate-200/90 bg-white/95 shadow-lg backdrop-blur-md">
//         <div className="sticky top-0 flex items-center justify-between rounded-t-2xl border-b border-slate-100 bg-white/95 px-4 py-3 backdrop-blur-md">
//           <div className="flex items-center gap-1.5">
//             <AlertTriangle className="h-3.5 w-3.5 text-rose-500" />
//             <h3 className="text-xs font-bold uppercase tracking-wide text-slate-800">
//               {selectedRegion ? `Alerts — ${selectedRegion.name}` : 'All Active Alerts — India'}
//             </h3>
//           </div>
//           <span
//             className={`h-1.5 w-1.5 rounded-full ${alertsFetching ? 'animate-pulse bg-emerald-500' : 'bg-slate-300'}`}
//             title={alertsFetching ? 'Refreshing…' : 'Up to date'}
//           />
//         </div>

//         {mapAlerts.length === 0 ? (
//           <p className="px-4 py-6 text-center text-xs text-slate-400">
//             {selectedRegion
//               ? 'No active alerts for this region.'
//               : 'No active alerts nationwide.'}
//           </p>
//         ) : (
//           mapAlerts.map((alert) => (
//             <CompactAlertRow
//               key={alert.id}
//               alert={alert}
//               onClick={() => setSelectedMapAlert(alert)}
//             />
//           ))
//         )}
//       </div>

//       {/* Selected region detail card */}
//       {selectedRegion && (
//         <div className="absolute bottom-6 left-6 z-10 w-84 bg-white/95 backdrop-blur-md p-5 rounded-2xl border border-slate-200 shadow-xl space-y-3">
//           <div className="flex justify-between items-start border-b border-slate-100 pb-2">
//             <div>
//               <span className="text-[10px] font-bold text-blue-600 uppercase tracking-wider">Selected Region</span>
//               <h2 className="text-lg font-extrabold text-slate-900">{selectedRegion.name}</h2>
//               <p className="text-xs text-slate-500">{selectedRegion.state}</p>
//             </div>
//             <div className="text-right">
//               <span className="text-2xl font-black text-rose-600 font-mono">
//                 {currentRiskData?.riskScore ?? selectedRegion.riskScore}
//               </span>
//               <span className="text-[10px] text-slate-400 block font-semibold">/ 100 Risk (+{forecastHorizon.value}h)</span>
//             </div>
//           </div>

//           {activeThresholds && (
//             <div className="space-y-1.5">
//               {activeThresholds.thresholds.map((rule) => (
//                 <div
//                   key={rule.id}
//                   className={`flex items-center justify-between rounded-lg px-2.5 py-1.5 text-[11px] ${
//                     rule.isMet ? 'bg-red-50 text-red-700' : 'bg-slate-50 text-slate-500'
//                   }`}
//                 >
//                   <span className="font-semibold">{rule.label}</span>
//                   <span className="font-bold uppercase">{rule.statusText}</span>
//                 </div>
//               ))}
//             </div>
//           )}
//         </div>
//       )}

//       {/* Clicking a compact alert row opens the same full detail modal
//           used on the Alert Ticker page — consistent UX, zero new component. */}
//       <AlertDetails alert={selectedMapAlert} onClose={() => setSelectedMapAlert(null)} />
//     </div>
//   );
// };

// export default LiveMapPage;
