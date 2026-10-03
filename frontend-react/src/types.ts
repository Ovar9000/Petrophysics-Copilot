export interface CurveInfo {
  mnemonic: string;
  unit: string;
  description: string;
  valid_samples: number;
  min: number | null;
  max: number | null;
}

export interface DepthWindow {
  top: number;
  bottom: number;
  marker: number;
  basis: string;
}

export interface WellData {
  well_id: string;
  well_name: string;
  uwi: string;
  start_depth: number;
  stop_depth: number;
  step: number;
  total_curves: number;
  curves: CurveInfo[];
  available_mnemonics: string[];
  capabilities?: Record<string, boolean>;
  default_window?: DepthWindow;
}

// A geology-report statement checked against the well's logs (backend/report_check.py).
export interface ReportCheck {
  statement: string;
  status: 'consistent' | 'partly_consistent' | 'contradicted' | 'not_checkable';
  claimed?: string;
  actual?: Record<string, any>;
  rule?: string;
  reason?: string;
  note?: string;
  interval_m?: [number, number];
}

export interface CatalogRecord {
  well_id?: string;
  well_name: string;
  formation_tops: string;
  lithology_notes: string;
  verification?: { summary?: Record<string, number>; checks?: ReportCheck[]; status?: string };
}

// "What to look at / what to ask next", built by the backend from tool results.
export interface Guide {
  observe: string[];
  next: Array<{ prompt: string; why?: string }>;
}

// Default 100 m viewing window for a well (backend-derived), with a safe fallback.
export const windowFor = (wells: WellData[], wellId: string): DepthWindow => {
  const w = wells.find((x) => x.well_id === wellId);
  if (w?.default_window) return w.default_window;
  const top = w ? Math.floor(w.start_depth) : 0;
  return { top, bottom: top + 100, marker: top + 50, basis: 'start of log' };
};

export interface ToolCallItem {
  name: string;
  args: Record<string, any>;
  result?: any;
  via?: string; // e.g. "MCP (stdio)": how the backend executed the tool
}

export interface MessageItem {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  toolCalls?: ToolCallItem[];
  figures?: string[];
  guide?: Guide;
  timestamp: string;
}

export interface NetPayKPIs {
  well_id: string;
  top_depth: number;
  bottom_depth: number;
  gross_interval_m: number;
  net_reservoir_m: number;
  net_pay_m: number;
  net_to_gross: number;
  reservoir_to_gross: number;
  pay_zone_averages: {
    average_porosity: number;
    average_water_saturation: number;
    average_shale_volume: number;
  };
  cum_pay_curve?: Array<{ depth: number; cum_pay_m: number }>;
  facies_breakdown?: {
    pay_m: number;
    pay_pct: number;
    wet_reservoir_m: number;
    wet_reservoir_pct: number;
    non_reservoir_m: number;
    non_reservoir_pct: number;
  };
  cutoff_sensitivity?: Array<{
    vsh_cutoff: number;
    phi_cutoff: number;
    sw_cutoff: number;
    net_pay_m: number;
    net_to_gross: number;
  }>;
  net_pay_uncertainty?: {
    p90_m: number;
    p50_m: number;
    p10_m: number;
    plus_minus_m: number;
    basis: string;
  };
  methodology?: string;
}

export interface SweetspotItem {
  zone_name: string;
  rank?: number;
  top_depth: number;
  base_depth: number;
  thickness_m: number;
  avg_porosity: number;
  avg_sw: number;
  avg_vsh: number;
  hcpv_index: number;
}

export interface SweetspotScanResult {
  well_id: string;
  total_sweetspots_found: number;
  total_pay_thickness_m: number;
  sweetspots: SweetspotItem[];
}
