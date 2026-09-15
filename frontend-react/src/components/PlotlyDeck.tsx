import React, { useState, useMemo, useEffect } from 'react';
import Plotly from 'plotly.js-dist-min';
import createPlotlyComponent from 'react-plotly.js/factory';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const Plot = createPlotlyComponent(Plotly);

const API_BASE = typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://127.0.0.1:8000';

type PlotKind = '1d' | '2d' | '3d';

// Backend stamps layout.meta.plot_kind on every figure; fall back to the old
// content sniffing only for figures generated before the tag existed.
const plotKindOf = (spec: any): PlotKind => {
  const tagged = (spec?.layout as any)?.meta?.plot_kind;
  if (tagged === '1d' || tagged === '2d' || tagged === '3d') return tagged;
  const is3D = spec?.data?.some((d: any) => d.type === 'scatter3d' || d.type === 'surface' || d.type === 'mesh3d');
  if (is3D) return '3d';
  const title = spec?.layout?.title?.text ?? spec?.layout?.title ?? '';
  return String(title).includes('Crossplot') || String(title).includes('Picket') ? '2d' : '1d';
};

// Single POST-JSON helper for all tool endpoints (replaces 7 copy-pasted fetches).
const postTool = async <T,>(path: string, body: unknown): Promise<T | null> => {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch (e) {
    console.error(`POST ${path} failed:`, e);
    return null;
  }
};

// Shared Plotly chrome: white canvas, app font, no fixed size (fills container).
const toLightLayout = (
  spec: any,
  uirevision: string,
  opts: { margin?: any; legend?: any; plotBg?: boolean; hovermode?: string } = {}
): any => {
  const lightLayout: any = {
    ...spec.layout,
    autosize: true,
    uirevision,
    paper_bgcolor: '#FFFFFF',
    font: { family: 'Inter, system-ui, sans-serif', color: '#334155', size: 11 },
    ...(opts.margin !== undefined ? { margin: opts.margin } : {}),
    ...(opts.legend !== undefined ? { legend: opts.legend } : {}),
  };
  if (opts.plotBg) lightLayout.plot_bgcolor = '#FFFFFF';
  if (opts.hovermode) lightLayout.hovermode = opts.hovermode;
  delete lightLayout.width;
  delete lightLayout.height;
  return lightLayout;
};
import { 
  LineChart, 
  ScatterChart, 
  BarChart2, 
  BookOpen, 
  Download,
  Flame,
  Layers,
  ChevronRight,
  ChevronLeft,
  Sliders,
  RotateCcw,
  Box,
  Compass,
  CheckCircle2
} from 'lucide-react';
import { NetPayKPIs, SweetspotScanResult } from '../types';

interface PlotlyDeckProps {
  activeFigureJson: string | null;
  netPayData: NetPayKPIs | null;
  sweetspotsData: SweetspotScanResult | null;
  citations: Array<{ well_name: string; formation_tops: string; lithology_notes: string }>;
  selectedWell: string;
  onSelectWell?: (well: string) => void;
}

export const PlotlyDeck: React.FC<PlotlyDeckProps> = ({
  activeFigureJson,
  netPayData,
  sweetspotsData,
  citations,
  selectedWell,
  onSelectWell
}) => {
  const [activeTab, setActiveTab] = useState<'1d' | '3d' | 'sweetspots' | '2d' | 'kpis' | 'citations'>('1d');
  
  interface WellVisualCache {
    '1d'?: string | null;
    '2d'?: string | null;
    '3d'?: string | null;
    sweetspots?: SweetspotScanResult | null;
    kpis?: NetPayKPIs | null;
    citations?: Array<{ well_name: string; formation_tops: string; lithology_notes: string }> | null;
  }

  // Unified per-well cache so all tabs and graphs synchronize cleanly on well switch
  const [wellDataCache, setWellDataCache] = useState<{ [wellId: string]: WellVisualCache }>({});

  const [loading3d, setLoading3d] = useState<boolean>(false);
  const [loadingTab, setLoadingTab] = useState<boolean>(false);
  const [current3dMode, setCurrent3dMode] = useState<'trajectory' | 'cube'>('cube');
  const [trajColorBy, setTrajColorBy] = useState<'sweetspots' | 'rdeep' | 'gr' | 'tvdss'>('sweetspots');
  const [depthMin, setDepthMin] = useState<number>(1800);
  const [depthMax, setDepthMax] = useState<number>(2000);
  const [depthMarker, setDepthMarker] = useState<number | null>(null);
  const [showCorrelationLine, setShowCorrelationLine] = useState<boolean>(true);
  const [showHorizon, setShowHorizon] = useState<boolean>(true);
  const [highlightedSweetspot, setHighlightedSweetspot] = useState<{
    label: string;
    top: number;
    base: number;
  } | null>(null);

  // Sync default depth ranges and correlation line marker on well change, and load active tab
  useEffect(() => {
    if (selectedWell === 'Well1') {
      setDepthMin(1850);
      setDepthMax(1950);
      setDepthMarker(1908.0);
    } else {
      setDepthMin(3650);
      setDepthMax(3750);
      setDepthMarker(3685.0);
    }
    ensureTabData(activeTab, selectedWell);
  }, [selectedWell]);

  // React to sweet spots emitted from chat
  useEffect(() => {
    if (sweetspotsData && sweetspotsData.sweetspots && sweetspotsData.sweetspots.length > 0) {
      setWellDataCache(prev => ({
        ...prev,
        [selectedWell]: { ...(prev[selectedWell] || {}), sweetspots: sweetspotsData }
      }));
      setActiveTab('sweetspots');
    }
  }, [sweetspotsData]);

  // React to figure emitted from chat (kind comes stamped from the backend)
  useEffect(() => {
    if (activeFigureJson) {
      try {
        const kind = plotKindOf(JSON.parse(activeFigureJson));
        setWellDataCache(prev => ({
          ...prev,
          [selectedWell]: { ...(prev[selectedWell] || {}), [kind]: activeFigureJson }
        }));
        setActiveTab(kind);
      } catch {
        setActiveTab('1d');
      }
    }
  }, [activeFigureJson]);

  // Universal well-data loader for active tab
  const loadTabForWell = async (tab: typeof activeTab, well: string) => {
    if (tab === '3d') {
      loadDirect3D(current3dMode, well);
      return;
    }

    setLoadingTab(true);
    const top = well === 'Well1' ? 1850 : 3650;
    const bot = well === 'Well1' ? 1950 : 3750;

    try {
      if (tab === '1d') {
        const data = await postTool<{ figure_json?: string }>('/api/tools/plot_1d', { well_id: well, top_depth: top, bottom_depth: bot });
        if (data?.figure_json) {
          setWellDataCache(prev => ({
            ...prev,
            [well]: { ...(prev[well] || {}), '1d': data.figure_json as string }
          }));
        }
      } else if (tab === '2d') {
        const data = await postTool<{ figure_json?: string }>('/api/tools/crossplot', {
          well_id: well,
          x_curve: 'NEUT',
          y_curve: 'DENB',
          z_curve: 'GR',
          top_depth: top,
          bottom_depth: bot
        });
        if (data?.figure_json) {
          setWellDataCache(prev => ({
            ...prev,
            [well]: { ...(prev[well] || {}), '2d': data.figure_json as string }
          }));
        }
      } else if (tab === 'sweetspots') {
        const data = await postTool<SweetspotScanResult>('/api/tools/sweetspots', { well_id: well, min_thickness: 1.5 });
        if (data) {
          setWellDataCache(prev => ({
            ...prev,
            [well]: { ...(prev[well] || {}), sweetspots: data }
          }));
        }
      } else if (tab === 'kpis') {
        const data = await postTool<NetPayKPIs>('/api/tools/net_pay', {
          well_id: well,
          top_depth: top,
          bottom_depth: bot,
          vsh_cutoff: 0.3,
          phi_cutoff: 0.1,
          sw_cutoff: 0.5
        });
        if (data) {
          setWellDataCache(prev => ({
            ...prev,
            [well]: { ...(prev[well] || {}), kpis: data }
          }));
        }
      } else if (tab === 'citations') {
        const res = await fetch(`${API_BASE}/api/tools/catalog?q=formation&well=${well}`);
        if (res.ok) {
          const data = await res.json();
          setWellDataCache(prev => ({
            ...prev,
            [well]: { ...(prev[well] || {}), citations: data }
          }));
        }
      }
    } catch (e) {
      console.error(`Failed to load ${tab} for ${well}:`, e);
    } finally {
      setLoadingTab(false);
    }
  };

  const loadDirect3D = async (
    mode: 'trajectory' | 'cube',
    targetWell?: string,
    customTop?: number,
    customBot?: number,
    customColorBy?: 'sweetspots' | 'rdeep' | 'gr' | 'tvdss',
    highlightTop?: number,
    highlightBase?: number,
    highlightLabelStr?: string,
    horizonOverride?: boolean
  ) => {
    setLoading3d(true);
    setCurrent3dMode(mode);
    const well = targetWell || selectedWell;
    const top = customTop !== undefined ? customTop : depthMin;
    const bot = customBot !== undefined ? customBot : depthMax;
    const activeColor = customColorBy !== undefined ? customColorBy : trajColorBy;
    const useHorizon = horizonOverride !== undefined ? horizonOverride : showHorizon;
    if (customColorBy !== undefined) {
      setTrajColorBy(customColorBy);
    }
    try {
      const endpoint = mode === 'trajectory' ? '/api/tools/plot_3d_trajectory' : '/api/tools/plot_3d_cube';
      const body: any = { well_id: well, top_depth: top, bottom_depth: bot };
      if (mode === 'trajectory') {
        body.color_by = activeColor;
        body.show_horizon = useHorizon;
        if (highlightTop !== undefined) body.highlight_top = highlightTop;
        if (highlightBase !== undefined) body.highlight_base = highlightBase;
        if (highlightLabelStr !== undefined) body.highlight_label = highlightLabelStr;
      }
      const data = await postTool<{ figure_json?: string }>(endpoint, body);
      if (data?.figure_json) {
        setWellDataCache(prev => ({
          ...prev,
          [well]: { ...(prev[well] || {}), '3d': data.figure_json as string }
        }));
        setActiveTab('3d');
      }
    } catch (e) {
      console.error('Failed to load 3D plot:', e);
    } finally {
      setLoading3d(false);
    }
  };

  const ensureTabData = async (tab: typeof activeTab, well: string) => {
    const cached = wellDataCache[well];
    if (tab === '3d') {
      if (!cached?.['3d']) {
        loadDirect3D(current3dMode, well);
      }
      return;
    }
    if (tab === '1d' && cached?.['1d']) return;
    if (tab === '2d' && cached?.['2d']) return;
    if (tab === 'sweetspots' && cached?.sweetspots) return;
    if (tab === 'kpis' && cached?.kpis) return;
    if (tab === 'citations' && cached?.citations) return;

    await loadTabForWell(tab, well);
  };

  const handleInspectWell = (well: string) => {
    if (onSelectWell) onSelectWell(well);
    ensureTabData(activeTab, well);
  };

  const handleTabClick = (tab: typeof activeTab) => {
    setActiveTab(tab);
    ensureTabData(tab, selectedWell);
  };

  const handleSwitchDepthInterval = async (top: number, bot: number) => {
    setDepthMin(top);
    setDepthMax(bot);
    setLoadingTab(true);
    try {
      const data = await postTool<{ figure_json?: string }>('/api/tools/plot_1d', { well_id: selectedWell, top_depth: top, bottom_depth: bot });
      if (data?.figure_json) {
        setWellDataCache(prev => ({
          ...prev,
          [selectedWell]: { ...(prev[selectedWell] || {}), '1d': data.figure_json as string }
        }));
      }
    } catch (e) {
      console.error('Failed to switch depth interval:', e);
    } finally {
      setLoadingTab(false);
    }
  };

  const currentWellCache = wellDataCache[selectedWell] || {};
  const current1dJson = currentWellCache['1d'];
  const current2dJson = currentWellCache['2d'];
  const current3dJson = currentWellCache['3d'];
  const currentSweetspots = currentWellCache.sweetspots || (sweetspotsData?.well_id === selectedWell ? sweetspotsData : null);
  const currentNetPay = currentWellCache.kpis || (netPayData?.well_id === selectedWell ? netPayData : null);
  const currentCitations = currentWellCache.citations || (citations?.length && citations[0]?.well_name?.includes(selectedWell) ? citations : []);

  // Net Pay dashboard helpers: sensitivity heat range + cumulative-chart facies shading.
  // Coherent color key across the dashboard: pay = green, wet reservoir = blue, non-pay = gray.
  const sensCells = currentNetPay?.cutoff_sensitivity ?? [];
  const sensMin = sensCells.length ? Math.min(...sensCells.map((c) => c.net_pay_m)) : 0;
  const sensMax = sensCells.length ? Math.max(...sensCells.map((c) => c.net_pay_m)) : 0;
  const sensHeatBg = (v: number) => {
    const t = sensMax > sensMin ? (v - sensMin) / (sensMax - sensMin) : 0.5;
    return `rgba(16, 185, 129, ${(0.06 + 0.55 * t).toFixed(3)})`;
  };
  const cumCurve = currentNetPay?.cum_pay_curve ?? [];
  const cumXMax = cumCurve.length ? Math.max(...cumCurve.map((p) => p.cum_pay_m)) : 0;
  const cumXEnd = cumXMax > 0 ? cumXMax * 1.07 : 1;
  // Flat runs (constant cum pay across consecutive samples) = non-pay intervals.
  const nonPayRuns: Array<{ y0: number; y1: number; cum: number }> = [];
  for (let i = 0; i + 1 < cumCurve.length; i++) {
    if (Math.abs(cumCurve[i + 1].cum_pay_m - cumCurve[i].cum_pay_m) < 1e-9) {
      const last = nonPayRuns[nonPayRuns.length - 1];
      if (last && Math.abs(last.y1 - cumCurve[i].depth) < 1e-9) {
        last.y1 = cumCurve[i + 1].depth;
      } else {
        nonPayRuns.push({ y0: cumCurve[i].depth, y1: cumCurve[i + 1].depth, cum: cumCurve[i].cum_pay_m });
      }
    }
  }

  // Parsed 1D Curves with Interactive 1 Continuous Horizontal Line Across 3 Graphs
  const parsed1DPlot = useMemo(() => {
    const rawJson = current1dJson;
    if (!rawJson) return null;
    try {
      const spec = JSON.parse(rawJson);
      if (plotKindOf(spec) !== '1d') return null;

      // Extract depth limits
      let dMin = selectedWell === 'Well1' ? 1850 : 3650;
      let dMax = selectedWell === 'Well1' ? 1950 : 3750;
      if (spec.data && Array.isArray(spec.data)) {
        spec.data.forEach((trace: any) => {
          if (Array.isArray(trace.y) && trace.y.length > 0) {
            trace.y.forEach((val: any) => {
              if (typeof val === 'number' && !isNaN(val)) {
                if (val < dMin) dMin = val;
                if (val > dMax) dMax = val;
              }
            });
          }
        });
      }

      const activeDepth = depthMarker !== null 
        ? depthMarker 
        : (spec.layout?.shapes?.[0]?.y0 ?? (selectedWell === 'Well1' ? 1908.0 : 3685.0));

      const lightLayout = toLightLayout(
        spec,
        `${selectedWell}-${Math.round(dMin)}-${Math.round(dMax)}`,
        {
          margin: spec.layout?.margin ? { ...spec.layout.margin, r: 120 } : { l: 60, r: 120, t: 112, b: 70 },
          plotBg: true,
          hovermode: 'y'
        }
      );

      Object.keys(lightLayout).forEach(key => {
        if (key.startsWith('xaxis') || key.startsWith('yaxis')) {
          lightLayout[key] = {
            ...lightLayout[key],
            gridcolor: '#F1F5F9',
            zerolinecolor: '#E2E8F0',
            tickcolor: '#94A3B8',
            linecolor: '#CBD5E1',
            tickfont: { color: lightLayout[key]?.tickfont?.color || '#64748B', size: 10 }
          };
          if (key.startsWith('yaxis')) {
            lightLayout[key].showspikes = true;
            lightLayout[key].spikemode = 'across';
            lightLayout[key].spikesnap = 'cursor';
            lightLayout[key].spikethickness = 1.5;
            lightLayout[key].spikedash = 'solid';
            lightLayout[key].spikecolor = '#64748b';
          }
        }
      });

      // Filter out previous depth marker shape/annotation if any
      const existingShapes = (lightLayout.shapes || []).filter((s: any) => s.name !== 'Depth Correlation Line');
      const existingAnnotations = (lightLayout.annotations || []).filter((a: any) => !a.text?.includes('Depth:'));

      if (showCorrelationLine) {
        lightLayout.shapes = [
          ...existingShapes,
          {
            type: 'line',
            xref: 'paper',
            yref: 'y',
            x0: 0,
            x1: 1,
            y0: activeDepth,
            y1: activeDepth,
            line: { color: '#ef4444', width: 2, dash: 'dash' },
            name: 'Depth Correlation Line'
          }
        ];
        lightLayout.annotations = [
          ...existingAnnotations,
          {
            xref: 'paper',
            yref: 'y',
            x: 1.008,
            y: activeDepth,
            text: `<b>Depth: ${activeDepth.toFixed(1)}m</b>`,
            showarrow: false,
            xanchor: 'left',
            yanchor: 'middle',
            font: { color: '#b91c1c', size: 10, family: 'Inter, monospace' },
            bgcolor: 'rgba(254, 242, 242, 0.95)',
            bordercolor: '#f87171',
            borderwidth: 1,
            borderpad: 3
          }
        ];
      } else {
        lightLayout.shapes = existingShapes;
        lightLayout.annotations = existingAnnotations;
      }

      return { data: spec.data, layout: lightLayout, dMin, dMax, activeDepth };
    } catch (e) {
      console.error('Failed to parse 1D Plotly JSON:', e);
      return null;
    }
  }, [current1dJson, depthMarker, showCorrelationLine, selectedWell]);

  // Parsed 2D Crossplot
  const parsed2DPlot = useMemo(() => {
    const rawJson = current2dJson;
    if (!rawJson) return null;
    try {
      const spec = JSON.parse(rawJson);
      if (plotKindOf(spec) === '3d') return null;

      const lightLayout = toLightLayout(spec, `${selectedWell}-2d`, {
        margin: spec.layout?.margin || { l: 65, r: 110, t: 80, b: 55 },
        legend: spec.layout?.legend || {
          orientation: 'h',
          yanchor: 'bottom',
          y: 1.02,
          xanchor: 'right',
          x: 1.0,
          bgcolor: 'rgba(255,255,255,0.85)',
          bordercolor: '#E2E8F0',
          borderwidth: 1,
          font: { size: 10 }
        },
        plotBg: true
      });

      return { data: spec.data, layout: lightLayout };
    } catch (e) {
      console.error('Failed to parse 2D Plotly JSON:', e);
      return null;
    }
  }, [current2dJson, selectedWell]);

  // Parsed 3D WebGL
  const parsed3DPlot = useMemo(() => {
    const rawJson = current3dJson;
    if (!rawJson) return null;
    try {
      const spec = JSON.parse(rawJson);
      // Include the active target in uirevision so "Locate on 3D Wellbore"
      // forces Plotly to apply the backend's zoomed camera/axis ranges.
      // (With a static uirevision, Plotly keeps the old camera and the new
      // target stays a tiny dot in a wide view, looking detached.)
      const targetKey = highlightedSweetspot
        ? `-${highlightedSweetspot.top}-${highlightedSweetspot.base}`
        : '-full';
      const lightLayout = toLightLayout(spec, `${selectedWell}-3d${targetKey}`, {
        margin: spec.layout?.margin || { l: 20, r: 160, t: 40, b: 20 },
        legend: spec.layout?.legend || {
          orientation: 'v',
          yanchor: 'top',
          y: 0.98,
          xanchor: 'left',
          x: 1.02,
          bgcolor: 'rgba(255, 255, 255, 0.92)',
          bordercolor: '#E2E8F0',
          borderwidth: 1,
          font: { size: 10, color: '#334155' }
        }
      });
      const axisStyle = (ax: any) => ({
        ...(ax || {}),
        gridcolor: '#E2E8F0',
        zerolinecolor: '#CBD5E1',
        linecolor: '#94A3B8'
      });
      lightLayout.scene = {
        ...(spec.layout?.scene || {}),
        bgcolor: '#FAFAFA',
        xaxis: axisStyle(spec.layout?.scene?.xaxis),
        yaxis: axisStyle(spec.layout?.scene?.yaxis),
        zaxis: axisStyle(spec.layout?.scene?.zaxis),
      };
      return { data: spec.data, layout: lightLayout };
    } catch (e) {
      console.error('Failed to parse 3D Plotly JSON:', e);
      return null;
    }
  }, [current3dJson, selectedWell, highlightedSweetspot]);

  const handleDownloadJson = () => {
    const jsonToDownload = activeTab === '3d' 
      ? current3dJson 
      : activeTab === '2d' 
      ? current2dJson 
      : current1dJson;

    if (!jsonToDownload) return;
    const blob = new Blob([jsonToDownload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${activeTab}_${selectedWell}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex-1 flex flex-col bg-[#F8FAFC] h-full overflow-hidden">
      {/* Top Navigation Tabs & Well Switcher */}
      <div className="h-12 px-4 border-b border-zinc-200 flex items-center justify-between bg-white shrink-0">
        <div className="flex items-center gap-6 overflow-x-auto">
          <button
            onClick={() => handleTabClick('1d')}
            className={`text-xs font-medium pb-3 pt-3 relative transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === '1d' ? 'text-zinc-900 font-semibold' : 'text-zinc-400 hover:text-zinc-700'
            }`}
          >
            1D Log Curves
            {activeTab === '1d' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-zinc-900 rounded-full" />}
          </button>

          <button
            onClick={() => handleTabClick('3d')}
            className={`text-xs font-medium pb-3 pt-3 relative transition-colors whitespace-nowrap flex items-center gap-1.5 cursor-pointer ${
              activeTab === '3d' ? 'text-zinc-900 font-semibold' : 'text-zinc-400 hover:text-zinc-700'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>3D Subsurface</span>
            {activeTab === '3d' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-zinc-900 rounded-full" />}
          </button>

          <button
            onClick={() => handleTabClick('sweetspots')}
            className={`text-xs font-medium pb-3 pt-3 relative transition-colors whitespace-nowrap flex items-center gap-1.5 cursor-pointer ${
              activeTab === 'sweetspots' ? 'text-zinc-900 font-semibold' : 'text-zinc-400 hover:text-zinc-700'
            }`}
          >
            <span>Reservoir Sweet Spots</span>
            {currentSweetspots && currentSweetspots.sweetspots && (
              <span className="px-1.5 py-0.2 rounded bg-emerald-100 text-emerald-800 text-[10px] font-mono">
                {currentSweetspots.total_sweetspots_found}
              </span>
            )}
            {activeTab === 'sweetspots' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-zinc-900 rounded-full" />}
          </button>

          <button
            onClick={() => handleTabClick('2d')}
            className={`text-xs font-medium pb-3 pt-3 relative transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === '2d' ? 'text-zinc-900 font-semibold' : 'text-zinc-400 hover:text-zinc-700'
            }`}
          >
            2D Crossplot
            {activeTab === '2d' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-zinc-900 rounded-full" />}
          </button>

          <button
            onClick={() => handleTabClick('kpis')}
            className={`text-xs font-medium pb-3 pt-3 relative transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'kpis' ? 'text-zinc-900 font-semibold' : 'text-zinc-400 hover:text-zinc-700'
            }`}
          >
            Net Pay Metrics
            {activeTab === 'kpis' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-zinc-900 rounded-full" />}
          </button>

          <button
            onClick={() => handleTabClick('citations')}
            className={`text-xs font-medium pb-3 pt-3 relative transition-colors whitespace-nowrap cursor-pointer ${
              activeTab === 'citations' ? 'text-zinc-900 font-semibold' : 'text-zinc-400 hover:text-zinc-700'
            }`}
          >
            Stratigraphy Catalog
            {activeTab === 'citations' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-zinc-900 rounded-full" />}
          </button>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          {/* Universal Inspect Well Switcher */}
          <div className="flex items-center gap-1 bg-zinc-100 p-0.5 rounded border border-zinc-200 text-xs font-mono shadow-2xs">
            <span className="px-1.5 text-[10px] text-zinc-400 uppercase font-sans font-semibold">Inspect:</span>
            <button
              onClick={() => handleInspectWell('Well1')}
              className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                selectedWell === 'Well1' 
                  ? 'bg-white text-zinc-900 font-semibold shadow-xs' 
                  : 'text-zinc-500 hover:text-zinc-800'
              }`}
              title="Inspect Well 1 (Target Alpha)"
            >
              Well 1
            </button>
            <button
              onClick={() => handleInspectWell('Well2')}
              className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                selectedWell === 'Well2' 
                  ? 'bg-white text-zinc-900 font-semibold shadow-xs' 
                  : 'text-zinc-500 hover:text-zinc-800'
              }`}
              title="Inspect Well 2 (Exploration Beta)"
            >
              Well 2
            </button>
          </div>

          <button
            onClick={handleDownloadJson}
            className="flex items-center gap-1 px-2 py-1 rounded border border-zinc-200 bg-white hover:bg-zinc-50 text-zinc-600 text-xs font-mono transition-colors shrink-0 cursor-pointer"
            title="Download active plot JSON spec"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Spec</span>
          </button>
        </div>
      </div>

      {/* Main Visual Canvas Area */}
      <div className="flex-1 overflow-y-auto p-4 flex flex-col">
        {/* Tab 1: 1D Log Curves */}
        {activeTab === '1d' && (
          <div className="flex-1 flex flex-col bg-white rounded-lg border border-zinc-200 p-2 min-h-[600px] shadow-xs">
            {/* Horizontal Depth Correlation Line Control Banner */}
            {parsed1DPlot && (
              <div className="flex flex-wrap items-center justify-between px-3 py-2 bg-slate-50 border border-slate-200 rounded-md mb-2 text-xs gap-2 shrink-0">
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-1.5 font-semibold text-slate-800">
                    <span className={`w-2.5 h-2.5 rounded-full ${showCorrelationLine ? 'bg-red-500 animate-pulse' : 'bg-slate-300'}`} />
                    <span>Horizontal Correlation Line</span>
                  </div>
                  <div className="flex items-center gap-1 bg-white px-2 py-0.5 rounded border border-slate-200 shadow-2xs">
                    <span className="text-slate-500 font-mono">Depth:</span>
                    <input
                      type="number"
                      step="0.5"
                      value={parsed1DPlot.activeDepth.toFixed(1)}
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) setDepthMarker(val);
                      }}
                      className="w-20 px-1 py-0.5 font-mono text-red-600 font-semibold border-b border-red-300 focus:outline-none focus:border-red-600 bg-transparent text-center"
                    />
                    <span className="text-slate-400 font-mono">m</span>
                  </div>
                  <input
                    type="range"
                    min={parsed1DPlot.dMin}
                    max={parsed1DPlot.dMax}
                    step="0.2"
                    value={parsed1DPlot.activeDepth}
                    onChange={(e) => setDepthMarker(parseFloat(e.target.value))}
                    className="w-32 sm:w-44 accent-red-600 cursor-pointer"
                    title="Slide horizontal correlation line across tracks"
                  />
                  {/* Depth Window Presets (100m Pay vs 200m Overview) */}
                  <div className="flex items-center gap-1 bg-white p-0.5 rounded border border-slate-200 text-xs font-mono shadow-2xs">
                    <button
                      onClick={() => handleSwitchDepthInterval(selectedWell === 'Well1' ? 1850 : 3650, selectedWell === 'Well1' ? 1950 : 3750)}
                      className={`px-2 py-0.5 rounded transition-colors cursor-pointer text-[10px] ${
                        Math.abs((parsed1DPlot.dMax - parsed1DPlot.dMin) - 100) < 15
                          ? 'bg-slate-900 text-white font-medium'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                      title="100m Pay Window (1:1 aspect ratio matching Well 1)"
                    >
                      100m Pay
                    </button>
                    <button
                      onClick={() => handleSwitchDepthInterval(selectedWell === 'Well1' ? 1800 : 3600, selectedWell === 'Well1' ? 2000 : 3800)}
                      className={`px-2 py-0.5 rounded transition-colors cursor-pointer text-[10px] ${
                        Math.abs((parsed1DPlot.dMax - parsed1DPlot.dMin) - 200) < 15
                          ? 'bg-slate-900 text-white font-medium'
                          : 'text-slate-600 hover:text-slate-900'
                      }`}
                      title="200m Regional Overview"
                    >
                      200m Overview
                    </button>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-[11px] text-slate-400 italic hidden xl:inline">
                    Click track to place line
                  </span>
                  <button
                    onClick={() => setDepthMarker(selectedWell === 'Well1' ? 1908.0 : 3685.0)}
                    className="px-2 py-1 bg-white hover:bg-slate-100 text-slate-700 rounded border border-slate-200 font-mono text-[11px] transition-colors cursor-pointer"
                    title="Snap horizontal line to primary sweet spot"
                  >
                    Pay Peak
                  </button>
                  <button
                    onClick={() => setShowCorrelationLine(prev => !prev)}
                    className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors cursor-pointer ${
                      showCorrelationLine 
                        ? 'bg-red-50 text-red-700 border border-red-200 font-semibold' 
                        : 'bg-white text-slate-500 border border-slate-200'
                    }`}
                  >
                    {showCorrelationLine ? '1 Line: ON' : '1 Line: OFF'}
                  </button>
                </div>
              </div>
            )}

            {loadingTab ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-2 py-20">
                <div className="w-6 h-6 border-2 border-zinc-300 border-t-zinc-700 rounded-full animate-spin" />
                <div className="text-xs font-medium text-zinc-600 font-mono">Loading {selectedWell} 1D log curves...</div>
              </div>
            ) : parsed1DPlot ? (
              <div className="w-full flex-1 min-h-[680px] relative">
                <Plot
                  data={parsed1DPlot.data}
                  layout={parsed1DPlot.layout}
                  useResizeHandler={true}
                  className="w-full h-full"
                  style={{ width: '100%', height: '100%' }}
                  config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                  onClick={(e: any) => {
                    if (e && e.points && e.points.length > 0) {
                      const clickedY = e.points[0].y;
                      if (typeof clickedY === 'number' && !isNaN(clickedY)) {
                        setDepthMarker(Math.round(clickedY * 10) / 10);
                      }
                    }
                  }}
                />
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-3 py-20">
                <LineChart className="w-10 h-10 text-zinc-300 stroke-1" />
                <div className="text-xs font-medium text-zinc-600">No log curves rendered for {selectedWell}</div>
                <div className="text-[11px] text-zinc-400 max-w-xs text-center">
                  Click below to render the 3-track composite log (GR, Resistivity, Density-Neutron crossover) for {selectedWell}.
                </div>
                <button
                  onClick={() => loadTabForWell('1d', selectedWell)}
                  className="px-3 py-1.5 text-xs font-mono bg-zinc-900 text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                >
                  Load {selectedWell} 1D Curves
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 2: 3D Subsurface */}
        {activeTab === '3d' && (
          <div className="flex-1 flex flex-col bg-white rounded-lg border border-zinc-200 p-2 min-h-[600px] shadow-xs">
            <div className="flex flex-col border-b border-zinc-100 bg-zinc-50/90 rounded-t-md mb-2">
              <div className="flex items-center justify-between px-3 py-2">
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => loadDirect3D('cube', selectedWell, depthMin, depthMax)}
                    disabled={loading3d}
                    className={`px-3 py-1 text-xs font-mono rounded border transition-colors cursor-pointer flex items-center gap-1.5 ${
                      current3dMode === 'cube' 
                        ? 'bg-white text-zinc-900 border-zinc-300 font-semibold shadow-xs' 
                        : 'bg-transparent text-zinc-500 border-transparent hover:text-zinc-800'
                    }`}
                  >
                    <Box className="w-3.5 h-3.5 text-zinc-600" />
                    3D Petrophysical Cluster Space (NPHI · RHOB · DT)
                  </button>
                  <button
                    onClick={() => loadDirect3D('trajectory', selectedWell, depthMin, depthMax)}
                    disabled={loading3d}
                    className={`px-3 py-1 text-xs font-mono rounded border transition-colors cursor-pointer flex items-center gap-1.5 ${
                      current3dMode === 'trajectory' 
                        ? 'bg-white text-zinc-900 border-zinc-300 font-semibold shadow-xs' 
                        : 'bg-transparent text-zinc-500 border-transparent hover:text-zinc-800'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5 text-zinc-600" />
                    3D Wellbore & Horizon
                  </button>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-zinc-400 font-mono hidden lg:flex">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
                  <span>Interactive WebGL · Left-Click Rotate · Scroll Zoom · Right-Click Pan</span>
                </div>
              </div>

              {/* Depth Slicing / Interactive Filtering Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 bg-white/70 border-t border-zinc-200/70 text-xs font-mono">
                <div className="flex items-center gap-2">
                  <span className="text-zinc-500 font-semibold uppercase text-[10px] flex items-center gap-1">
                    <Sliders className="w-3 h-3 text-zinc-500" />
                    Depth Slice (MD):
                  </span>
                  <div className="flex items-center gap-1">
                    <input
                      type="number"
                      value={depthMin}
                      onChange={(e) => setDepthMin(Number(e.target.value))}
                      className="w-16 px-1.5 py-0.5 bg-white border border-zinc-300 rounded text-center text-xs font-mono text-zinc-800 focus:outline-none focus:border-zinc-500"
                    />
                    <span className="text-zinc-400">m –</span>
                    <input
                      type="number"
                      value={depthMax}
                      onChange={(e) => setDepthMax(Number(e.target.value))}
                      className="w-16 px-1.5 py-0.5 bg-white border border-zinc-300 rounded text-center text-xs font-mono text-zinc-800 focus:outline-none focus:border-zinc-500"
                    />
                    <span className="text-zinc-400">m</span>
                  </div>
                  <button
                    onClick={() => loadDirect3D(current3dMode, selectedWell, depthMin, depthMax)}
                    disabled={loading3d}
                    className="px-2.5 py-0.5 bg-zinc-800 text-white rounded hover:bg-zinc-700 transition-colors text-[11px] cursor-pointer"
                  >
                    Apply Slice
                  </button>
                </div>

                <div className="flex items-center gap-1.5">
                  <span className="text-[10px] text-zinc-400 uppercase">Step Chrono:</span>
                  <button
                    onClick={() => {
                      const step = 25;
                      const newTop = Math.max(0, depthMin - step);
                      const newBot = Math.max(step, depthMax - step);
                      setDepthMin(newTop);
                      setDepthMax(newBot);
                      loadDirect3D(current3dMode, selectedWell, newTop, newBot);
                    }}
                    disabled={loading3d}
                    className="px-2 py-0.5 bg-white border border-zinc-200 rounded hover:bg-zinc-100 text-zinc-700 transition-colors cursor-pointer text-[11px] flex items-center gap-0.5 shadow-2xs"
                    title="Step Up 25m"
                  >
                    <ChevronLeft className="w-3 h-3" /> -25m
                  </button>
                  <button
                    onClick={() => {
                      const step = 25;
                      const newTop = depthMin + step;
                      const newBot = depthMax + step;
                      setDepthMin(newTop);
                      setDepthMax(newBot);
                      loadDirect3D(current3dMode, selectedWell, newTop, newBot);
                    }}
                    disabled={loading3d}
                    className="px-2 py-0.5 bg-white border border-zinc-200 rounded hover:bg-zinc-100 text-zinc-700 transition-colors cursor-pointer text-[11px] flex items-center gap-0.5 shadow-2xs"
                    title="Step Down 25m"
                  >
                    +25m <ChevronRight className="w-3 h-3" />
                  </button>
                  <button
                    onClick={() => {
                      const top = selectedWell === 'Well1' ? 1800 : 3550;
                      const bot = selectedWell === 'Well1' ? 2000 : 3850;
                      setDepthMin(top);
                      setDepthMax(bot);
                      loadDirect3D(current3dMode, selectedWell, top, bot);
                    }}
                    disabled={loading3d}
                    className="px-2 py-0.5 bg-white border border-zinc-200 rounded hover:bg-zinc-100 text-zinc-600 transition-colors cursor-pointer text-[11px] flex items-center gap-0.5 shadow-2xs"
                    title="Reset to Full Reservoir Zone"
                  >
                    <RotateCcw className="w-3 h-3" /> Reset Zone
                  </button>
                </div>
              </div>

              {/* Trajectory Color By Switcher */}
              {current3dMode === 'trajectory' && (
                <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-1.5 bg-zinc-100/70 border-t border-zinc-200 text-xs font-mono">
                  <span className="text-zinc-600 font-semibold uppercase text-[10px] flex items-center gap-1">
                    Wellbore Color Attribute:
                  </span>
                  <div className="flex items-center gap-1">
                    {[
                      { key: 'sweetspots', label: 'Sweet Spots (Pay)' },
                      { key: 'rdeep', label: 'Resistivity (Rt)' },
                      { key: 'gr', label: 'Gamma Ray (GR)' },
                      { key: 'tvdss', label: 'Depth (TVDSS)' }
                    ].map((attr) => (
                      <button
                        key={attr.key}
                        onClick={() => loadDirect3D('trajectory', selectedWell, depthMin, depthMax, attr.key as any)}
                        disabled={loading3d}
                        className={`px-2 py-0.5 rounded text-[11px] transition-colors cursor-pointer border ${
                          trajColorBy === attr.key
                            ? 'bg-zinc-900 text-white border-zinc-900 font-medium shadow-2xs'
                            : 'bg-white text-zinc-600 border-zinc-200 hover:bg-zinc-100'
                        }`}
                      >
                        {attr.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Show Geological Horizon Toggle */}
              {current3dMode === 'trajectory' && (
                <div className="flex items-center gap-2 px-3 py-1 bg-amber-50/70 border-t border-amber-100 text-xs font-mono">
                  <label className="flex items-center gap-1.5 cursor-pointer select-none text-amber-800">
                    <input
                      type="checkbox"
                      checked={showHorizon}
                      onChange={(e) => {
                        setShowHorizon(e.target.checked);
                        loadDirect3D('trajectory', selectedWell, depthMin, depthMax, undefined, undefined, undefined, undefined, e.target.checked);
                      }}
                      className="accent-amber-600 w-3.5 h-3.5"
                    />
                    <span className="font-semibold">Show Geological Horizon</span>
                    <span className="text-amber-600/70 text-[10px]">(Top Reservoir Surface · Earth colormap · Dip ~3.1° SE)</span>
                  </label>
                </div>
              )}
            </div>

            {/* Active Target Locator Banner */}
            {highlightedSweetspot && activeTab === '3d' && (
              <div className="flex items-center justify-between gap-2 px-3 py-1.5 bg-amber-50 border-b border-amber-200 text-xs font-mono">
                <div className="flex items-center gap-1.5 text-amber-800">
                  <span className="font-semibold">Target Located on 3D Wellbore:</span>
                  <span className="text-amber-700">{highlightedSweetspot.label}</span>
                </div>
                <button
                  onClick={() => {
                    setHighlightedSweetspot(null);
                    const fullTop = selectedWell === 'Well1' ? 1800 : 3550;
                    const fullBot = selectedWell === 'Well1' ? 2000 : 3850;
                    setDepthMin(fullTop);
                    setDepthMax(fullBot);
                    loadDirect3D('trajectory', selectedWell, fullTop, fullBot, 'sweetspots');
                  }}
                  className="px-2 py-0.5 bg-amber-100 hover:bg-amber-200 text-amber-900 border border-amber-300 rounded text-[10px] cursor-pointer transition-colors"
                >
                  ✕ Reset View
                </button>
              </div>
            )}

            {loading3d ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-2 py-20 min-h-[500px]">
                <div className="w-6 h-6 border-2 border-zinc-300 border-t-zinc-700 rounded-full animate-spin" />
                <div className="text-xs font-medium text-zinc-600 font-mono">Generating 3D spatial representation for {selectedWell}...</div>
              </div>
            ) : parsed3DPlot ? (
              <div className="w-full flex-1 min-h-[620px] relative">
                <Plot
                  data={parsed3DPlot.data}
                  layout={parsed3DPlot.layout}
                  useResizeHandler={true}
                  className="w-full h-full"
                  style={{ width: '100%', height: '100%' }}
                  config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                />
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-3 py-20 min-h-[500px]">
                <Layers className="w-10 h-10 text-zinc-300 stroke-1" />
                <div className="text-xs font-medium text-zinc-600">No 3D Model Loaded for {selectedWell}</div>
                <div className="text-[11px] text-zinc-400 max-w-xs text-center">
                  Click an option above to generate the interactive 3D wellbore trajectory or 3D petrophysical cube.
                </div>
                <button
                  onClick={() => loadDirect3D('trajectory', selectedWell)}
                  className="px-3 py-1.5 text-xs font-mono bg-zinc-900 text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                >
                  Launch 3D Subsurface View
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 3: Reservoir Sweet Spots */}
        {activeTab === 'sweetspots' && (
          <div className="flex-1 flex flex-col space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider font-mono">
                Automated Reservoir Sweet Spot Delineation ({selectedWell})
              </h3>
              {currentSweetspots && (
                <span className="text-xs text-zinc-500 font-mono">
                  Total Cumulative Pay: <strong className="text-emerald-600">{currentSweetspots.total_pay_thickness_m} m</strong>
                </span>
              )}
            </div>

            {loadingTab ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-2 py-20 bg-white rounded-lg border border-zinc-200">
                <div className="w-6 h-6 border-2 border-zinc-300 border-t-zinc-700 rounded-full animate-spin" />
                <div className="text-xs font-medium text-zinc-600 font-mono">Scanning {selectedWell} for sweet spots...</div>
              </div>
            ) : currentSweetspots && currentSweetspots.sweetspots && currentSweetspots.sweetspots.length > 0 ? (
              <div className="space-y-3">
                {currentSweetspots.sweetspots.map((z, idx) => (
                  <div key={idx} className="bg-white border border-zinc-200 rounded-lg p-4 shadow-xs hover:border-zinc-300 transition-colors">
                    <div className="flex flex-wrap items-center justify-between border-b border-zinc-100 pb-2 mb-3 gap-2">
                      <div className="flex items-center gap-2">
                        <span className="w-6 h-6 rounded bg-emerald-50 text-emerald-700 font-mono text-xs flex items-center justify-center font-bold">
                          #{idx + 1}
                        </span>
                        <span className="text-xs font-bold text-zinc-900">
                          {z.zone_name.startsWith('Rank') ? z.zone_name : `Rank #${idx + 1}: Zone ${z.top_depth}m – ${z.base_depth}m (${z.thickness_m}m Pay)`}
                        </span>
                        <span className="text-[11px] font-mono text-zinc-500">
                          {z.top_depth}m – {z.base_depth}m
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold font-mono text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                          Net Pay: {z.thickness_m} m
                        </span>
                        <button
                          onClick={() => {
                            // Tight context window so a thin target fills the view.
                            // (Was ±200m: a 3.8m target was <1% of the track and the
                            // 20m-offset beacon looked detached from the wellbore.)
                            const thickness = Math.max(1, z.base_depth - z.top_depth);
                            const ctxMargin = Math.min(80, Math.max(40, thickness * 4 + 20));
                            const ctxTop = Math.max(0, Math.floor(z.top_depth - ctxMargin));
                            const ctxBot = Math.ceil(z.base_depth + ctxMargin);
                            setDepthMin(ctxTop);
                            setDepthMax(ctxBot);
                            setCurrent3dMode('trajectory');
                            const label = `Zone ${z.top_depth}–${z.base_depth}m | Net Pay ${z.thickness_m}m`;
                            setHighlightedSweetspot({ label, top: z.top_depth, base: z.base_depth });
                            loadDirect3D(
                              'trajectory', selectedWell, ctxTop, ctxBot, 'sweetspots',
                              z.top_depth, z.base_depth, label
                            );
                          }}
                          className="px-2.5 py-1 text-xs font-mono bg-zinc-900 hover:bg-zinc-800 text-white rounded flex items-center gap-1.5 transition-colors cursor-pointer shadow-2xs"
                          title="View this sweet spot on the 3D wellbore trajectory"
                        >
                          <Compass className="w-3.5 h-3.5 text-emerald-400" />
                          Locate on 3D Wellbore
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-3 text-center">
                      <div className="p-2 bg-zinc-50 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 font-mono uppercase">Avg Porosity</div>
                        <div className="text-sm font-bold font-mono text-zinc-900 mt-0.5">
                          {(z.avg_porosity * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="p-2 bg-zinc-50 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 font-mono uppercase">Water Saturation</div>
                        <div className="text-sm font-bold font-mono text-zinc-900 mt-0.5">
                          {(z.avg_sw * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="p-2 bg-zinc-50 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 font-mono uppercase">Shale Volume</div>
                        <div className="text-sm font-bold font-mono text-zinc-900 mt-0.5">
                          {(z.avg_vsh * 100).toFixed(1)}%
                        </div>
                      </div>

                      <div className="p-2 bg-zinc-50 rounded border border-zinc-100">
                        <div className="text-[10px] text-zinc-400 font-mono uppercase">HCPV Index</div>
                        <div className="text-sm font-bold font-mono text-blue-600 mt-0.5">
                          {z.hcpv_index}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-3 py-20 bg-white rounded-lg border border-zinc-200">
                <Flame className="w-10 h-10 text-zinc-300 stroke-1" />
                <div className="text-xs font-medium text-zinc-600">No sweet spots scanned yet for {selectedWell}</div>
                <div className="text-[11px] text-zinc-400 max-w-xs text-center">
                  Click below to delineate high-grade pay intervals meeting volumetric cutoffs.
                </div>
                <button
                  onClick={() => loadTabForWell('sweetspots', selectedWell)}
                  className="px-3 py-1.5 text-xs font-mono bg-zinc-900 text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                >
                  Scan {selectedWell} Sweet Spots
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 4: 2D Crossplot */}
        {activeTab === '2d' && (
          <div className="flex-1 flex flex-col bg-white rounded-lg border border-zinc-200 p-2 min-h-[600px] shadow-xs">
            {loadingTab ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-2 py-20">
                <div className="w-6 h-6 border-2 border-zinc-300 border-t-zinc-700 rounded-full animate-spin" />
                <div className="text-xs font-medium text-zinc-600 font-mono">Generating {selectedWell} 2D crossplot...</div>
              </div>
            ) : parsed2DPlot ? (
              <div className="w-full flex-1 min-h-[620px] relative">
                <Plot
                  data={parsed2DPlot.data}
                  layout={parsed2DPlot.layout}
                  useResizeHandler={true}
                  className="w-full h-full"
                  style={{ width: '100%', height: '100%' }}
                  config={{ responsive: true, displayModeBar: true, displaylogo: false }}
                />
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-3 py-20">
                <ScatterChart className="w-10 h-10 text-zinc-300 stroke-1" />
                <div className="text-xs font-medium text-zinc-600">No crossplot loaded for {selectedWell}</div>
                <div className="text-[11px] text-zinc-400 max-w-xs text-center">
                  Click below to render the RHOB vs NPHI crossplot colored by Gamma Ray with mineral trendlines.
                </div>
                <button
                  onClick={() => loadTabForWell('2d', selectedWell)}
                  className="px-3 py-1.5 text-xs font-mono bg-zinc-900 text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                >
                  Generate {selectedWell} Crossplot
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 5: Net Pay Metrics */}
        {activeTab === 'kpis' && (
          <div className="flex-1 flex flex-col space-y-4">
            {loadingTab ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-2 py-20 bg-white rounded-lg border border-zinc-200">
                <div className="w-6 h-6 border-2 border-zinc-300 border-t-zinc-700 rounded-full animate-spin" />
                <div className="text-xs font-medium text-zinc-600 font-mono">Computing {selectedWell} Net Pay...</div>
              </div>
            ) : currentNetPay ? (
              <>
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider font-mono">
                    Net Pay Volumetric Evaluation ({currentNetPay.well_id})
                  </h3>
                  <span className="font-mono text-xs text-zinc-500">
                    Depth: {currentNetPay.top_depth}m – {currentNetPay.bottom_depth}m
                  </span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-white border border-zinc-200 rounded-lg p-3.5 shadow-xs">
                    <div className="text-[10px] uppercase font-mono tracking-wider text-zinc-400">Gross Interval</div>
                    <div className="text-2xl font-bold font-mono text-zinc-900 mt-1">
                      {currentNetPay.gross_interval_m} <span className="text-xs text-zinc-400 font-sans">m</span>
                    </div>
                  </div>

                  <div className="bg-white border border-zinc-200 rounded-lg p-3.5 shadow-xs">
                    <div className="text-[10px] uppercase font-mono tracking-wider text-zinc-400">Net Reservoir</div>
                    <div className="text-2xl font-bold font-mono text-blue-600 mt-1">
                      {currentNetPay.net_reservoir_m} <span className="text-xs text-zinc-400 font-sans">m</span>
                    </div>
                    <div className="text-[10px] text-zinc-400 mt-1 font-mono">
                      {(currentNetPay.reservoir_to_gross * 100).toFixed(1)}% of gross
                    </div>
                  </div>

                  <div className="bg-white border border-zinc-200 rounded-lg p-3.5 shadow-xs">
                    <div className="text-[10px] uppercase font-mono tracking-wider text-zinc-400">Net Pay</div>
                    <div className="text-2xl font-bold font-mono text-emerald-600 mt-1">
                      {currentNetPay.net_pay_m} <span className="text-xs text-zinc-400 font-sans">m</span>
                    </div>
                    <div className="text-[10px] text-zinc-400 mt-1 font-mono">
                      Hydrocarbon productive
                    </div>
                    {currentNetPay.net_pay_uncertainty && (
                      <div className="text-[10px] text-emerald-700 mt-1 font-mono" title={currentNetPay.net_pay_uncertainty.basis}>
                        P90–P10: {currentNetPay.net_pay_uncertainty.p90_m}–{currentNetPay.net_pay_uncertainty.p10_m} m (±{currentNetPay.net_pay_uncertainty.plus_minus_m})
                      </div>
                    )}
                  </div>

                  <div className="bg-white border border-zinc-200 rounded-lg p-3.5 shadow-xs">
                    <div className="text-[10px] uppercase font-mono tracking-wider text-zinc-400">Net-to-Gross (NTG)</div>
                    <div className="text-2xl font-bold font-mono text-emerald-600 mt-1">
                      {(currentNetPay.net_to_gross * 100).toFixed(1)}%
                    </div>
                    <div className="w-full bg-zinc-100 h-1 rounded-full mt-2 overflow-hidden border border-zinc-200">
                      <div
                        className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, Math.max(3, currentNetPay.net_to_gross * 100))}%` }}
                      />
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-4 text-[10px] font-mono text-zinc-400 px-1">
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> Pay (green)</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-blue-500 inline-block" /> Wet reservoir (blue)</span>
                  <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-slate-400 inline-block" /> Non-pay / gross (gray)</span>
                </div>

                <div className="bg-white border border-zinc-200 rounded-lg p-4 shadow-xs">
                  <h4 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider font-mono mb-3">
                    Pay Zone Average Petrophysical Properties
                  </h4>
                  <div className="grid grid-cols-3 gap-3">
                    <div className="p-3 bg-zinc-50 rounded border border-zinc-200">
                      <div className="text-[10px] text-zinc-400 uppercase font-mono">Avg Porosity (Pay)</div>
                      <div className="text-lg font-bold font-mono text-zinc-900 mt-1">
                        {((currentNetPay.pay_zone_averages?.average_porosity ?? 0) * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="p-3 bg-zinc-50 rounded border border-zinc-200">
                      <div className="text-[10px] text-zinc-400 uppercase font-mono">Avg Sw (Pay)</div>
                      <div className="text-lg font-bold font-mono text-blue-600 mt-1">
                        {((currentNetPay.pay_zone_averages?.average_water_saturation ?? 0) * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="p-3 bg-zinc-50 rounded border border-zinc-200">
                      <div className="text-[10px] text-zinc-400 uppercase font-mono">Avg Vshale (Pay)</div>
                      <div className="text-lg font-bold font-mono text-amber-600 mt-1">
                        {((currentNetPay.pay_zone_averages?.average_shale_volume ?? 0) * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </div>

                {/* Facies Volumetric Breakdown */}
                {currentNetPay.facies_breakdown && (
                  <div className="bg-white border border-zinc-200 rounded-lg p-4 shadow-xs">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider font-mono">
                        Interval Facies & Fluid Distribution
                      </h4>
                      <span className="text-[11px] font-mono text-zinc-500">
                        Total Evaluated: {currentNetPay.gross_interval_m} m
                      </span>
                    </div>

                    {/* Visual Segmented Stacked Bar */}
                    <div className="w-full flex h-3 rounded-full overflow-hidden border border-zinc-200 mb-3">
                      <div 
                        className="bg-emerald-500 h-full transition-all duration-500" 
                        style={{ width: `${currentNetPay.facies_breakdown.pay_pct}%` }} 
                        title={`Hydrocarbon Pay: ${currentNetPay.facies_breakdown.pay_m}m (${currentNetPay.facies_breakdown.pay_pct}%)`} 
                      />
                      <div 
                        className="bg-blue-500 h-full transition-all duration-500" 
                        style={{ width: `${currentNetPay.facies_breakdown.wet_reservoir_pct}%` }} 
                        title={`Wet Reservoir: ${currentNetPay.facies_breakdown.wet_reservoir_m}m (${currentNetPay.facies_breakdown.wet_reservoir_pct}%)`} 
                      />
                      <div 
                        className="bg-slate-400 h-full transition-all duration-500" 
                        style={{ width: `${currentNetPay.facies_breakdown.non_reservoir_pct}%` }} 
                        title={`Non-Reservoir / Shale: ${currentNetPay.facies_breakdown.non_reservoir_m}m (${currentNetPay.facies_breakdown.non_reservoir_pct}%)`} 
                      />
                    </div>

                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div className="p-2.5 bg-emerald-50/70 border border-emerald-200/80 rounded-lg">
                        <div className="text-[10px] uppercase font-mono text-emerald-800 font-bold flex items-center justify-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> Hydrocarbon Pay
                        </div>
                        <div className="text-base font-bold font-mono text-emerald-700 mt-1">
                          {currentNetPay.facies_breakdown.pay_m} m
                        </div>
                        <div className="text-[10px] text-emerald-600 font-mono mt-0.5">
                          {currentNetPay.facies_breakdown.pay_pct}% of gross
                        </div>
                      </div>

                      <div className="p-2.5 bg-blue-50/70 border border-blue-200/80 rounded-lg">
                        <div className="text-[10px] uppercase font-mono text-blue-800 font-bold flex items-center justify-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-blue-500 inline-block" /> Wet Reservoir Sand
                        </div>
                        <div className="text-base font-bold font-mono text-blue-700 mt-1">
                          {currentNetPay.facies_breakdown.wet_reservoir_m} m
                        </div>
                        <div className="text-[10px] text-blue-600 font-mono mt-0.5">
                          {currentNetPay.facies_breakdown.wet_reservoir_pct}% of gross
                        </div>
                      </div>

                      <div className="p-2.5 bg-zinc-50 border border-zinc-200 rounded-lg">
                        <div className="text-[10px] uppercase font-mono text-zinc-600 font-bold flex items-center justify-center gap-1">
                          <span className="w-2 h-2 rounded-full bg-slate-400 inline-block" /> Tight / Shale Non-Pay
                        </div>
                        <div className="text-base font-bold font-mono text-zinc-800 mt-1">
                          {currentNetPay.facies_breakdown.non_reservoir_m} m
                        </div>
                        <div className="text-[10px] text-zinc-500 font-mono mt-0.5">
                          {currentNetPay.facies_breakdown.non_reservoir_pct}% of gross
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Cumulative Pay Step Chart */}
                {currentNetPay.cum_pay_curve && currentNetPay.cum_pay_curve.length > 0 && (
                  <div className="bg-white border border-zinc-200 rounded-lg p-4 shadow-xs">
                    <h4 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider font-mono mb-2">
                      Cumulative Pay Buildup Profile vs. Depth (MD)
                    </h4>
                    <div className="w-full h-[260px]">
                      <Plot
                        data={[
                          {
                            x: currentNetPay.cum_pay_curve.map((p) => p.cum_pay_m),
                            y: currentNetPay.cum_pay_curve.map((p) => p.depth),
                            type: 'scatter',
                            mode: 'lines',
                            line: { shape: 'hv', color: '#059669', width: 2.2 },
                            fill: 'tozerox',
                            fillcolor: 'rgba(16, 185, 129, 0.15)',
                            hovertemplate: 'Cum pay: %{x:.2f} m<br>Depth: %{y:.1f} m MD<extra></extra>',
                            name: 'Cum Pay (m)'
                          },
                          ...(nonPayRuns.length > 0 ? [{
                            x: nonPayRuns.map(() => cumXEnd * 0.5),
                            y: nonPayRuns.map((r) => (r.y0 + r.y1) / 2),
                            type: 'scatter' as const,
                            mode: 'markers' as const,
                            marker: { size: 14, opacity: 0, color: 'rgba(0,0,0,0)' },
                            hovertemplate: nonPayRuns.map((r) =>
                              `Non-pay interval (wet or tight/shale)<br>${r.y0.toFixed(1)}–${r.y1.toFixed(1)} m MD<br>Cum pay flat at ${r.cum.toFixed(2)} m<extra></extra>`
                            ),
                            hoverlabel: { bgcolor: '#27272a', font: { color: '#fafafa', size: 10 } },
                            showlegend: false,
                            name: 'Non-pay'
                          }] : [])
                        ]}
                        layout={{
                          autosize: true,
                          margin: { l: 60, r: 25, t: 20, b: 40 },
                          paper_bgcolor: '#FFFFFF',
                          plot_bgcolor: '#FFFFFF',
                          shapes: nonPayRuns.map((r) => ({
                            type: 'rect' as const,
                            layer: 'below' as const,
                            xref: 'x' as const,
                            yref: 'y' as const,
                            x0: 0,
                            x1: cumXEnd,
                            y0: r.y0,
                            y1: r.y1,
                            fillcolor: 'rgba(148, 163, 184, 0.13)',
                            line: { width: 0 }
                          })),
                          yaxis: {
                            autorange: 'reversed',
                            title: { text: 'Depth MD (m)', font: { size: 10, color: '#64748b' } },
                            gridcolor: '#F1F5F9'
                          },
                          xaxis: {
                            title: { text: 'Cumulative Net Pay (m)', font: { size: 10, color: '#64748b' } },
                            gridcolor: '#F1F5F9',
                            range: [0, cumXEnd]
                          }
                        }}
                        useResizeHandler={true}
                        className="w-full h-full"
                        config={{ responsive: true, displayModeBar: false }}
                      />
                    </div>
                  </div>
                )}

                {/* Cutoff Sensitivity Matrix */}
                {currentNetPay.cutoff_sensitivity && currentNetPay.cutoff_sensitivity.length > 0 && (
                  <div className="bg-white border border-zinc-200 rounded-lg p-4 shadow-xs">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="text-xs font-semibold text-zinc-800 uppercase tracking-wider font-mono">
                        Volumetric Cutoff Sensitivity Matrix (Sw ≤ {((currentNetPay.cutoff_sensitivity?.[0]?.sw_cutoff ?? 0.5) * 100).toFixed(0)}%)
                      </h4>
                      <span className="text-[10px] text-zinc-500 font-mono">
                        Net Pay (m) · Net-to-Gross (%) · darker green = higher pay
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs font-mono border-collapse border border-zinc-200 text-center">
                        <thead>
                          <tr className="bg-zinc-100 text-zinc-700">
                            <th className="border border-zinc-200 p-2 text-left">Vshale Cutoff</th>
                            <th className="border border-zinc-200 p-2">Φ ≥ 8%</th>
                            <th className="border border-zinc-200 p-2 bg-emerald-50 text-emerald-900 border-emerald-300">
                              Φ ≥ 10% (Base Case)
                            </th>
                            <th className="border border-zinc-200 p-2">Φ ≥ 12%</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[0.20, 0.30, 0.40].map((vCut) => {
                            const isBaseVsh = vCut === 0.30;
                            return (
                              <tr key={vCut} className={isBaseVsh ? 'bg-zinc-50/60 font-semibold' : ''}>
                                <td className={`border border-zinc-200 p-2 text-left ${isBaseVsh ? 'text-emerald-900 font-bold' : 'text-zinc-600'}`}>
                                  Vsh ≤ {(vCut * 100).toFixed(0)}% {isBaseVsh && '(Base)'}
                                </td>
                                {[0.08, 0.10, 0.12].map((pCut) => {
                                  const cell = currentNetPay.cutoff_sensitivity?.find(
                                    (s) => Math.abs(s.vsh_cutoff - vCut) < 0.01 && Math.abs(s.phi_cutoff - pCut) < 0.01
                                  );
                                  const isBaseCell = vCut === 0.30 && pCut === 0.10;
                                  const heat = cell ? sensHeatBg(cell.net_pay_m) : undefined;
                                  return (
                                    <td
                                      key={pCut}
                                      style={heat ? { backgroundColor: heat } : undefined}
                                      title={cell ? `Vsh ≤ ${vCut}, Φ ≥ ${pCut}: ${cell.net_pay_m} m pay` : undefined}
                                      className={`border border-zinc-200 p-2 ${
                                        isBaseCell
                                          ? 'text-emerald-900 font-bold border-emerald-500 ring-1 ring-inset ring-emerald-500'
                                          : 'text-zinc-800'
                                      }`}
                                    >
                                      {cell ? (
                                        <div>
                                          <span className="text-sm">{cell.net_pay_m} m</span>
                                          <div className="text-[10px] text-zinc-500">
                                            NTG: {(cell.net_to_gross * 100).toFixed(1)}%
                                          </div>
                                        </div>
                                      ) : '—'}
                                    </td>
                                  );
                                })}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {/* Methodology audit footnote */}
                {currentNetPay.methodology && (
                  <div className="text-[10px] font-mono text-zinc-400 px-1 leading-relaxed">
                    Method: {currentNetPay.methodology}
                    {currentNetPay.net_pay_uncertainty && (
                      <span> Uncertainty: {currentNetPay.net_pay_uncertainty.basis}.</span>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-3 py-20 bg-white rounded-lg border border-zinc-200">
                <BarChart2 className="w-10 h-10 text-zinc-300 stroke-1" />
                <div className="text-xs font-medium text-zinc-600">No Net Pay computed yet for {selectedWell}</div>
                <div className="text-[11px] text-zinc-400 max-w-xs text-center">
                  Click below to calculate volumetric net pay with industry standard cutoffs (Vsh ≤ 0.3, Φ ≥ 0.1, Sw ≤ 0.5).
                </div>
                <button
                  onClick={() => loadTabForWell('kpis', selectedWell)}
                  className="px-3 py-1.5 text-xs font-mono bg-zinc-900 text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                >
                  Calculate {selectedWell} Net Pay
                </button>
              </div>
            )}
          </div>
        )}

        {/* Tab 6: Stratigraphy Catalog */}
        {activeTab === 'citations' && (
          <div className="flex-1 flex flex-col space-y-4">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-800 font-mono">
              Local Stratigraphy & Mudlog Catalog ({selectedWell})
            </h3>

            {!currentCitations || currentCitations.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center text-zinc-400 gap-3 py-20 bg-white rounded-lg border border-zinc-200">
                <BookOpen className="w-10 h-10 text-zinc-300 stroke-1" />
                <div className="text-xs font-medium text-zinc-600">No catalog records loaded for {selectedWell}</div>
                <div className="text-[11px] text-zinc-400 max-w-xs text-center">
                  Click below to inspect formation tops and hydrocarbon show notes for {selectedWell}.
                </div>
                <button
                  onClick={() => loadTabForWell('citations', selectedWell)}
                  className="px-3 py-1.5 text-xs font-mono bg-zinc-900 text-white rounded hover:bg-zinc-800 transition-colors cursor-pointer"
                >
                  Load {selectedWell} Catalog
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                {currentCitations.map((c, idx) => (
                  <div key={idx} className="bg-white border border-zinc-200 rounded-lg p-4 space-y-3 shadow-xs">
                    <div className="flex items-center justify-between border-b border-zinc-200 pb-2">
                      <span className="text-xs font-semibold text-zinc-900 font-mono flex items-center gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        {c.well_name} Geological Record
                      </span>
                      <span className="text-[10px] px-2 py-0.5 rounded bg-zinc-100 text-zinc-600 border border-zinc-200 font-mono">
                        Local Catalog
                      </span>
                    </div>

                    <div>
                      <h4 className="text-[11px] font-semibold text-zinc-700 uppercase tracking-wider font-mono mb-1.5">
                        Formation Tops & Intervals:
                      </h4>
                      <div className="text-xs text-zinc-800 bg-zinc-50 p-3 rounded border border-zinc-200 overflow-x-auto leading-relaxed [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-zinc-200 [&_th]:bg-zinc-100 [&_th]:px-2.5 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-mono [&_td]:border [&_td]:border-zinc-200 [&_td]:px-2.5 [&_td]:py-1.5 [&_td]:font-mono">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {c.formation_tops}
                        </ReactMarkdown>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-[11px] font-semibold text-zinc-700 uppercase tracking-wider font-mono mb-1.5">
                        Mudlog Hydrocarbon Notes:
                      </h4>
                      <div className="text-xs text-zinc-800 bg-zinc-50 p-3 rounded border border-zinc-200 leading-relaxed [&_table]:w-full [&_table]:border-collapse [&_th]:border [&_th]:border-zinc-200 [&_th]:bg-zinc-100 [&_th]:px-2.5 [&_th]:py-1.5 [&_th]:text-left [&_td]:border [&_td]:border-zinc-200 [&_td]:px-2.5 [&_td]:py-1.5">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {c.lithology_notes}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
