import React, { useState, useEffect, useRef } from 'react';
import { ChatStream } from './components/ChatStream';
import { PlotlyDeck } from './components/PlotlyDeck';
import { WellData, MessageItem, NetPayKPIs, SweetspotScanResult, Guide, CatalogRecord } from './types';

const API_BASE = typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://127.0.0.1:8000';

// Tools whose results populate a deck tab (so an answer using them opens the plot window).
const DECK_TOOLS = new Set(['compute_net_pay', 'scan_reservoir_sweetspots', 'query_geology_metadata']);

const nowStamp = () => new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

const figureWell = (figJson: string): string | null => {
  try {
    return JSON.parse(figJson)?.layout?.meta?.well_id ?? null;
  } catch {
    return null;
  }
};

export type UploadStatus = { name: string; state: 'uploading' | 'done' | 'error'; detail?: string };

export const App: React.FC = () => {
  const [wells, setWells] = useState<WellData[]>([]);
  const [selectedWell, setSelectedWell] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [uploads, setUploads] = useState<UploadStatus[]>([]);

  // The plot window stays closed until a question produces something to show.
  const [deckOpen, setDeckOpen] = useState<boolean>(false);
  const [hasDeckContent, setHasDeckContent] = useState<boolean>(false);

  // Visualization Deck State
  const [activeFigureJson, setActiveFigureJson] = useState<string | null>(null);
  const [netPayData, setNetPayData] = useState<NetPayKPIs | null>(null);
  const [sweetspotsData, setSweetspotsData] = useState<SweetspotScanResult | null>(null);
  const [citations, setCitations] = useState<CatalogRecord[]>([]);

  const [messages, setMessages] = useState<MessageItem[]>([
    {
      id: 'init-1',
      role: 'assistant',
      content: `### Drop a well log to begin
Drag a **.las** file onto the box below (or click it to browse). The file is checked and loaded, but nothing is plotted until you ask.

You can also ask about a well that's already loaded. Plots open on the right when a question produces one, and every answer ends with **what to look at** and **what to ask next**.`,
      timestamp: nowStamp()
    }
  ]);
  const welcomeGuideFor = useRef<string | null>(null);

  const fetchWells = async (): Promise<WellData[]> => {
    try {
      const res = await fetch(`${API_BASE}/api/wells`);
      if (res.ok) {
        const data = await res.json();
        const list: WellData[] = (data.wells || []).filter((w: WellData) => w.well_id);
        setWells(list);
        return list;
      }
    } catch (err) {
      console.warn('Backend connecting...', err);
    }
    return [];
  };

  useEffect(() => {
    fetchWells();
    const interval = setInterval(fetchWells, 10000);
    return () => clearInterval(interval);
  }, []);

  // Keep the selection valid as wells load or change.
  useEffect(() => {
    if (wells.length && !wells.some((w) => w.well_id === selectedWell)) {
      setSelectedWell(wells[0].well_id);
    }
  }, [wells, selectedWell]);

  // Starter suggestions on the welcome message, for the selected well, until the
  // conversation moves on.
  useEffect(() => {
    if (!selectedWell || messages.length !== 1 || welcomeGuideFor.current === selectedWell) return;
    welcomeGuideFor.current = selectedWell;
    fetch(`${API_BASE}/api/guide?well_id=${encodeURIComponent(selectedWell)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((guide: Guide | null) => {
        if (!guide) return;
        setMessages((prev) => (prev.length === 1 ? [{ ...prev[0], guide }] : prev));
      })
      .catch(() => undefined);
  }, [selectedWell, messages.length]);

  const pushAssistant = (content: string, extra: Partial<MessageItem> = {}) => {
    setMessages((prev) => [
      ...prev,
      { id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, role: 'assistant', content, timestamp: nowStamp(), ...extra }
    ]);
  };

  const handleUploadFiles = async (files: File[]) => {
    for (const file of files) {
      setUploads((prev) => [...prev.filter((u) => u.name !== file.name), { name: file.name, state: 'uploading' }]);
      try {
        const res = await fetch(`${API_BASE}/api/wells/upload?filename=${encodeURIComponent(file.name)}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/octet-stream' },
          body: file
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || res.statusText);

        const well: WellData = data.well;
        setUploads((prev) => prev.map((u) => (u.name === file.name ? { ...u, state: 'done', detail: well.well_id } : u)));
        await fetchWells();
        setSelectedWell(well.well_id);
        const depthIndex = new Set(['DEPTH', 'DEPT', 'MD']);
        const curves = well.available_mnemonics.filter((m) => !depthIndex.has(m.toUpperCase()));
        pushAssistant(
          `**Loaded ${well.well_id}** from \`${file.name}\`: ${well.start_depth}–${well.stop_depth} m MD, ` +
            `${curves.length} curves (${curves.join(', ')}).\n\nNothing is plotted yet. Ask for a view when you're ready.`,
          { guide: data.guide }
        );
      } catch (err: any) {
        setUploads((prev) => prev.map((u) => (u.name === file.name ? { ...u, state: 'error', detail: err.message } : u)));
        pushAssistant(`Could not load \`${file.name}\`: ${err.message}`);
      }
    }
  };

  const handleSendMessage = async (text: string) => {
    const userMsg: MessageItem = { id: `msg-${Date.now()}`, role: 'user', content: text, timestamp: nowStamp() };

    setMessages(prev => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const historyPayload = messages.map(m => ({
        role: m.role,
        content: m.content
      }));

      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          well_id: selectedWell,
          session_id: 'session-minimal',
          chat_history: historyPayload
        })
      });

      if (!res.ok) {
        throw new Error(`API Error: ${res.statusText}`);
      }

      const data = await res.json();
      let producedDeckContent = false;

      // Parse tool calls for Net Pay, Sweetspots, or Citations
      if (data.tool_calls && data.tool_calls.length > 0) {
        data.tool_calls.forEach((t: any) => {
          if (t.name === 'compute_net_pay' && t.result && t.result.gross_interval_m) {
            setNetPayData(t.result);
          }
          if (t.name === 'scan_reservoir_sweetspots' && t.result && t.result.sweetspots) {
            setSweetspotsData(t.result);
          }
          if (t.name === 'query_geology_metadata' && t.result && t.result.results) {
            setCitations(t.result.results);
          }
          if (DECK_TOOLS.has(t.name) && t.result && !t.result.error) producedDeckContent = true;
        });
      }

      // Update active figure if returned (PlotlyDeck derives the kind from the figure spec)
      if (data.figures && data.figures.length > 0) {
        const lastFigure = data.figures[data.figures.length - 1];
        setActiveFigureJson(lastFigure);
        // Show the well the plot belongs to, not whichever was selected before.
        const owner = figureWell(lastFigure);
        if (owner) setSelectedWell(owner);
        producedDeckContent = true;
      }

      if (producedDeckContent) {
        setHasDeckContent(true);
        setDeckOpen(true);
      }

      const botMsg: MessageItem = {
        id: `bot-${Date.now()}`,
        role: 'assistant',
        content: data.text || 'Calculation completed successfully.',
        toolCalls: data.tool_calls || [],
        figures: data.figures || [],
        guide: data.guide,
        timestamp: nowStamp()
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err: any) {
      const errorMsg: MessageItem = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `Error: Unable to reach backend engine (${err.message}). Ensure server is running on http://127.0.0.1:8000.`,
        timestamp: nowStamp()
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen w-screen flex bg-[#F8FAFC] text-slate-900 overflow-hidden font-sans">
      {/* Panel 1: Chat (full width until the first plot, then left half) */}
      <div className={`h-full flex flex-col transition-all duration-300 ${deckOpen ? 'w-1/2' : 'w-full'}`}>
        <ChatStream
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          selectedWell={selectedWell}
          onSelectWell={setSelectedWell}
          wells={wells}
          onUploadFiles={handleUploadFiles}
          uploads={uploads}
          deckOpen={deckOpen}
          canToggleDeck={hasDeckContent}
          onToggleDeck={() => setDeckOpen((v) => !v)}
        />
      </div>

      {/* Panel 2: Plot window, mounted only once opened */}
      {deckOpen && selectedWell && (
        <div className="w-1/2 h-full flex flex-col">
          <PlotlyDeck
            activeFigureJson={activeFigureJson}
            netPayData={netPayData}
            sweetspotsData={sweetspotsData}
            citations={citations}
            selectedWell={selectedWell}
            onSelectWell={setSelectedWell}
            wells={wells}
          />
        </div>
      )}
    </div>
  );
};

export default App;
