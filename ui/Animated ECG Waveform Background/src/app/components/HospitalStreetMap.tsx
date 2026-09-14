import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { ChevronLeft, ChevronRight, LocateFixed } from "lucide-react";
import "leaflet/dist/leaflet.css";

import type { OfflineHospital } from "../data/chennaiHospitals";

interface MapLocation {
  latitude: number;
  longitude: number;
  accuracy: number;
}

interface HospitalStreetMapProps {
  hospitals: OfflineHospital[];
  location: MapLocation | null;
  selectedHospitalId: string;
  routingEnabled: boolean;
  onSelectHospital: (hospital: OfflineHospital) => void;
  onExpandedChange?: (expanded: boolean) => void;
}

interface RouteSummary {
  distanceKm: number;
  durationMinutes: number;
}

const hospitalIcon = (selected: boolean) => L.divIcon({
  className: "hospital-map-marker",
  html: `<div style="display:flex;align-items:center;justify-content:center;width:${selected ? 38 : 30}px;height:${selected ? 38 : 30}px;border-radius:999px;background:${selected ? "#dc2626" : "white"};box-shadow:0 3px 12px rgba(0,0,0,.28);border:${selected ? "3px" : "2px"} solid white;font-size:${selected ? 21 : 17}px">🏥</div>`,
  iconSize: [selected ? 38 : 30, selected ? 38 : 30],
  iconAnchor: [selected ? 19 : 15, selected ? 19 : 15],
});

const userIcon = L.divIcon({
  className: "hospital-map-marker",
  html: '<div style="display:flex;align-items:center;justify-content:center;width:40px;height:40px;border-radius:999px;background:white;box-shadow:0 3px 14px rgba(0,0,0,.32);border:3px solid #dc2626;font-size:21px">❤️</div>',
  iconSize: [40, 40],
  iconAnchor: [20, 20],
});

export default function HospitalStreetMap({ hospitals, location, selectedHospitalId, routingEnabled, onSelectHospital, onExpandedChange }: HospitalStreetMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const hospitalMarkersRef = useRef<Map<string, L.Marker>>(new Map());
  const userMarkerRef = useRef<L.Marker | null>(null);
  const accuracyCircleRef = useRef<L.Circle | null>(null);
  const routeLayerRef = useRef<L.GeoJSON | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [routeSummary, setRouteSummary] = useState<RouteSummary | null>(null);
  const [routeStatus, setRouteStatus] = useState("Select a hospital to view its details.");

  const toggleExpanded = () => {
    setExpanded(value => {
      const nextValue = !value;
      onExpandedChange?.(nextValue);
      return nextValue;
    });
  };

  const recenterMap = () => {
    const map = mapRef.current;
    if (!map) return;
    if (location) {
      map.setView([location.latitude, location.longitude], 13);
      return;
    }
    map.setView([13.0604, 80.2496], 11);
  };

  const selectAdjacentHospital = (direction: -1 | 1) => {
    if (!hospitals.length) return;
    const currentIndex = hospitals.findIndex(hospital => hospital.id === selectedHospitalId);
    const nextIndex = currentIndex < 0
      ? (direction === 1 ? 0 : hospitals.length - 1)
      : (currentIndex + direction + hospitals.length) % hospitals.length;
    onSelectHospital(hospitals[nextIndex]);
  };

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, { zoomControl: true, attributionControl: true }).setView([13.0604, 80.2496], 11);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (routingEnabled && !tileLayerRef.current) {
      tileLayerRef.current = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
    } else if (!routingEnabled && tileLayerRef.current) {
      tileLayerRef.current.remove();
      tileLayerRef.current = null;
    }
  }, [routingEnabled]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    hospitalMarkersRef.current.forEach(marker => marker.remove());
    hospitalMarkersRef.current.clear();
    hospitals.forEach(hospital => {
      const marker = L.marker([hospital.latitude, hospital.longitude], { icon: hospitalIcon(hospital.id === selectedHospitalId) })
        .addTo(map)
        .bindTooltip(hospital.name, { direction: "top", offset: [0, -14] })
        .on("click", () => onSelectHospital(hospital));
      hospitalMarkersRef.current.set(hospital.id, marker);
    });
  }, [hospitals, onSelectHospital, selectedHospitalId]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !location) return;
    userMarkerRef.current?.remove();
    accuracyCircleRef.current?.remove();
    userMarkerRef.current = L.marker([location.latitude, location.longitude], { icon: userIcon, zIndexOffset: 1000 })
      .addTo(map)
      .bindTooltip("Your device location", { direction: "top", offset: [0, -18] });
    accuracyCircleRef.current = L.circle([location.latitude, location.longitude], {
      radius: Math.max(location.accuracy, 10),
      color: "#dc2626",
      fillColor: "#fecaca",
      fillOpacity: 0.16,
      weight: 1,
    }).addTo(map);
    map.setView([location.latitude, location.longitude], 13);
  }, [location]);

  useEffect(() => {
    window.setTimeout(() => mapRef.current?.invalidateSize(), 80);
  }, [expanded]);

  useEffect(() => {
    const map = mapRef.current;
    const hospital = hospitals.find(item => item.id === selectedHospitalId);
    routeLayerRef.current?.remove();
    routeLayerRef.current = null;
    setRouteSummary(null);
    if (!map || !hospital) {
      setRouteStatus("Select a hospital to view its details.");
      return;
    }
    if (!location) {
      map.setView([hospital.latitude, hospital.longitude], 14);
      setRouteStatus("Hospital selected. Use My Location to calculate a route.");
      return;
    }
    if (!routingEnabled) {
      const bounds = L.latLngBounds([[location.latitude, location.longitude], [hospital.latitude, hospital.longitude]]);
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
      setRouteStatus("Enable online street routing to calculate the driving route.");
      return;
    }

    const controller = new AbortController();
    setRouteStatus("Calculating road route…");
    const routeUrl = `https://router.project-osrm.org/route/v1/driving/${location.longitude},${location.latitude};${hospital.longitude},${hospital.latitude}?overview=full&geometries=geojson&steps=false`;
    fetch(routeUrl, { signal: controller.signal })
      .then(response => {
        if (!response.ok) throw new Error(`Routing service returned ${response.status}`);
        return response.json();
      })
      .then(data => {
        const route = data?.routes?.[0];
        if (!route?.geometry) throw new Error("No driving route was found");
        const layer = L.geoJSON(route.geometry, { style: { color: "#dc2626", weight: 5, opacity: 0.85 } }).addTo(map);
        routeLayerRef.current = layer;
        map.fitBounds(layer.getBounds(), { padding: [45, 45] });
        const summary = { distanceKm: route.distance / 1000, durationMinutes: route.duration / 60 };
        setRouteSummary(summary);
        setRouteStatus(`Route to ${hospital.name}`);
      })
      .catch(error => {
        if (error.name !== "AbortError") setRouteStatus("Online route unavailable. Check connectivity or open the location in your maps app.");
      });
    return () => controller.abort();
  }, [hospitals, location, routingEnabled, selectedHospitalId]);

  return <div className={expanded ? "fixed inset-0 z-[9999] flex flex-col bg-white p-3 sm:p-5" : "overflow-hidden rounded-xl border border-neutral-200 bg-white"}>
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-neutral-200 px-3 py-2">
      <div>
        <h4 className="text-xs font-bold text-neutral-900">Chennai street and hospital map</h4>
        <p className="text-[9px] text-neutral-500">❤️ Your GPS location · 🏥 Hospital · online road route</p>
      </div>
      <div className="flex items-center gap-2">
        <button type="button" onClick={recenterMap} title={location ? "Recenter on my location" : "Recenter on Chennai"} aria-label={location ? "Recenter map on my location" : "Recenter map on Chennai"} className="rounded-lg border border-neutral-300 bg-white p-2 text-neutral-700 hover:bg-neutral-50">
          <LocateFixed className="h-4 w-4" />
        </button>
        <button type="button" onClick={toggleExpanded} className="rounded-lg border border-neutral-300 bg-white px-3 py-2 text-[10px] font-bold text-neutral-700 hover:bg-neutral-50">
          {expanded ? "Exit full screen" : "Full screen"}
        </button>
      </div>
    </div>
    <div className={`relative ${expanded ? "min-h-0 flex-1" : "h-80"}`}>
      <div ref={containerRef} className="absolute inset-0 z-0" aria-label="Interactive Chennai hospital street map" />
      <button type="button" onClick={() => selectAdjacentHospital(-1)} title="Previous hospital" aria-label="Show previous hospital" className="absolute left-3 top-1/2 z-[600] -translate-y-1/2 rounded-full border border-neutral-300 bg-white/95 p-2.5 text-neutral-800 shadow-lg hover:bg-white disabled:opacity-40" disabled={!hospitals.length}>
        <ChevronLeft className="h-5 w-5" />
      </button>
      <button type="button" onClick={() => selectAdjacentHospital(1)} title="Next hospital" aria-label="Show next hospital" className="absolute right-3 top-1/2 z-[600] -translate-y-1/2 rounded-full border border-neutral-300 bg-white/95 p-2.5 text-neutral-800 shadow-lg hover:bg-white disabled:opacity-40" disabled={!hospitals.length}>
        <ChevronRight className="h-5 w-5" />
      </button>
      {!routingEnabled && <div className="pointer-events-none absolute inset-x-4 bottom-4 z-[500] rounded-xl border border-blue-200 bg-blue-50/95 px-3 py-2 text-center text-[10px] font-semibold text-blue-900 shadow-lg">Enable online streets and routes to load Chennai roads and calculate driving directions.</div>}
      <div className="pointer-events-none absolute inset-x-3 top-3 z-[500] flex justify-center">
        <div className="rounded-xl border border-neutral-200 bg-white/95 px-3 py-2 text-center text-[10px] font-semibold text-neutral-700 shadow-lg backdrop-blur">
          <div>{routeStatus}</div>
          {routeSummary && <div className="mt-0.5 font-black text-red-700">{routeSummary.distanceKm.toFixed(1)} km by road · about {Math.round(routeSummary.durationMinutes)} min</div>}
        </div>
      </div>
    </div>
    <div className="flex flex-wrap items-center justify-between gap-2 border-t border-neutral-200 bg-neutral-50 px-3 py-2 text-[9px] text-neutral-500">
      <span>Travel estimates depend on current routing data and are not emergency-response guarantees.</span>
      <span>{routingEnabled ? "Online routing enabled" : "Street map only"}</span>
    </div>
  </div>;
}
