import React, { useState } from 'react';
import { 
  Layers, 
  Database, 
  Activity, 
  ChevronDown, 
  ChevronRight, 
  LineChart, 
  ScatterChart, 
  Flame, 
  BookOpen, 
  Server, 
  Radio, 
  CheckCircle2, 
  PanelLeftClose, 
  PanelLeftOpen 
} from 'lucide-react';
import { WellData } from '../types';

interface SidebarProps {
  wells: WellData[];
  selectedWell: string;
  onSelectWell: (wellId: string) => void;
  onSelectPrompt: (promptText: string) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  backendOnline: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  wells,
  selectedWell,
  onSelectWell,
  onSelectPrompt,
  collapsed,
  onToggleCollapse,
  backendOnline,
}) => {
  const [expandedWells, setExpandedWells] = useState<Record<string, boolean>>({
    'Well1': true,
    'Well2': false
  });

  const toggleWellExpand = (wellId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedWells(prev => ({ ...prev, [wellId]: !prev[wellId] }));
  };

  const presetPrompts = [
    {
      icon: <LineChart className="w-4 h-4 text-petro-cyan" />,
      label: '1D Triple Combo Plot',
      prompt: `Plot the 1D well log curves for ${selectedWell} from ${selectedWell === 'Well1' ? '1850m to 1950m' : '3600m to 3800m'}.`
    },
    {
      icon: <ScatterChart className="w-4 h-4 text-petro-amber" />,
      label: 'RHOB vs NPHI Crossplot',
      prompt: `Generate a Density vs Neutron crossplot colored by GR for ${selectedWell}.`
    },
    {
      icon: <Flame className="w-4 h-4 text-petro-emerald" />,
      label: 'Calculate Net Pay',
      prompt: `Calculate net pay for ${selectedWell} between ${selectedWell === 'Well1' ? '1850m and 1950m' : '3600m and 3800m'} with Vsh cutoff 0.3, Porosity cutoff 0.1, and Sw cutoff 0.5.`
    },
    {
      icon: <BookOpen className="w-4 h-4 text-petro-purple" />,
      label: 'Weaviate Stratigraphy RAG',
      prompt: `What geological formation and hydrocarbon shows occur around ${selectedWell === 'Well1' ? '1900m in Well 1' : '3700m in Well 2'}?`
    }
  ];

  if (collapsed) {
    return (
      <div className="w-14 bg-dark-850 border-r border-dark-800 flex flex-col items-center py-4 justify-between transition-all duration-300">
        <div className="flex flex-col items-center gap-4">
          <button 
            onClick={onToggleCollapse} 
            className="p-2 rounded hover:bg-dark-800 text-slate-400 hover:text-slate-200"
            title="Expand Sidebar"
          >
            <PanelLeftOpen className="w-5 h-5" />
          </button>
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-petro-cyan to-blue-600 flex items-center justify-center font-bold text-white text-sm shadow-lg shadow-cyan-900/30">
            GA
          </div>
        </div>
        <div className="flex flex-col items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" title="System Online" />
        </div>
      </div>
    );
  }

  return (
    <aside className="w-[270px] bg-dark-850 border-r border-dark-800 flex flex-col justify-between shrink-0 select-none transition-all duration-300">
      {/* Header */}
      <div>
        <div className="h-14 px-4 border-b border-dark-800 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-md bg-gradient-to-br from-petro-cyan to-blue-600 flex items-center justify-center font-black text-white text-xs shadow-md shadow-cyan-950">
              GA
            </div>
            <div>
              <div className="text-xs font-bold tracking-wider text-slate-200 uppercase flex items-center gap-1.5">
                GeoAgent <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-petro-cyan font-mono border border-cyan-500/20">STUDIO</span>
              </div>
              <div className="text-[10px] text-slate-400 font-mono">v2.4 Petrophysics AI</div>
            </div>
          </div>
          <button 
            onClick={onToggleCollapse}
            className="p-1.5 rounded hover:bg-dark-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        </div>

        {/* Well Explorer Tree */}
        <div className="p-3">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 px-2 py-1.5 flex items-center justify-between">
            <span>Well Explorer</span>
            <span className="text-[10px] bg-dark-800 px-1.5 py-0.5 rounded text-slate-300 font-mono">{wells.length || 2}</span>
          </div>

          <div className="space-y-1.5 mt-1">
            {wells.length === 0 ? (
              <div className="text-xs text-slate-400 p-2 italic">Loading wells...</div>
            ) : (
              wells.map(w => {
                const isSelected = selectedWell.toLowerCase() === w.well_name.toLowerCase();
                const isExpanded = !!expandedWells[w.well_name];

                return (
                  <div 
                    key={w.well_name}
                    className={`rounded-lg border transition-all ${
                      isSelected 
                        ? 'bg-dark-800/80 border-cyan-500/40 shadow-sm shadow-cyan-950/20' 
                        : 'bg-dark-900/40 border-dark-800 hover:border-dark-700'
                    }`}
                  >
                    <div 
                      onClick={() => onSelectWell(w.well_name)}
                      className="p-2.5 flex items-center justify-between cursor-pointer"
                    >
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={(e) => toggleWellExpand(w.well_name, e)}
                          className="text-slate-400 hover:text-slate-200 p-0.5"
                        >
                          {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        </button>
                        <div>
                          <div className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                            {w.well_name === 'Well1' ? 'Well 01 - Alpha' : 'Well 02 - Beta'}
                            {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                          </div>
                          <div className="text-[10px] text-slate-400 font-mono">
                            {w.start_depth}m - {w.stop_depth}m
                          </div>
                        </div>
                      </div>
                      <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-dark-950 text-slate-300 border border-dark-800">
                        {w.total_curves} crv
                      </span>
                    </div>

                    {isExpanded && (
                      <div className="px-3 pb-2.5 pt-1 border-t border-dark-800/60 bg-dark-950/40 rounded-b-lg">
                        <div className="text-[10px] text-slate-400 uppercase font-mono mb-1">Key Mnemonics:</div>
                        <div className="flex flex-wrap gap-1">
                          {w.available_mnemonics.slice(0, 10).map(m => (
                            <span key={m} className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-dark-900 text-slate-300 border border-dark-800">
                              {m}
                            </span>
                          ))}
                          {w.available_mnemonics.length > 10 && (
                            <span className="text-[9px] font-mono px-1 text-slate-400">
                              +{w.available_mnemonics.length - 10}
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Preset Quick Prompts */}
        <div className="p-3 border-t border-dark-800">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 px-2 py-1.5">
            Analytical Presets
          </div>
          <div className="space-y-1.5 mt-1">
            {presetPrompts.map((p, idx) => (
              <button
                key={idx}
                onClick={() => onSelectPrompt(p.prompt)}
                className="w-full text-left px-2.5 py-2 rounded-lg bg-dark-900/50 hover:bg-dark-800 border border-dark-800 hover:border-slate-700 flex items-center gap-2.5 transition-all group"
              >
                <div className="p-1 rounded bg-dark-850 group-hover:scale-105 transition-transform shrink-0">
                  {p.icon}
                </div>
                <div className="truncate">
                  <div className="text-xs font-medium text-slate-200 group-hover:text-cyan-300 transition-colors">
                    {p.label}
                  </div>
                  <div className="text-[10px] text-slate-400 truncate">{p.prompt}</div>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* System Status Footer */}
      <div className="p-3 border-t border-dark-800 bg-dark-950/60">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-petro-cyan" /> Weaviate RAG
            </span>
            <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> Active
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px]">
            <span className="text-slate-400 flex items-center gap-1.5">
              <Server className="w-3.5 h-3.5 text-petro-amber" /> FastAPI / MCP
            </span>
            <span className={`inline-flex items-center gap-1 text-[10px] font-mono ${backendOnline ? 'text-emerald-400' : 'text-amber-400'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? 'bg-emerald-400' : 'bg-amber-400'}`} /> 
              {backendOnline ? 'Online' : 'Connecting'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
