import React, { useState, useEffect } from 'react';
import { ChatStream } from './components/ChatStream';
import { PlotlyDeck } from './components/PlotlyDeck';
import { WellData, MessageItem, NetPayKPIs, SweetspotScanResult } from './types';

const API_BASE = typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://127.0.0.1:8000';

export const App: React.FC = () => {
  const [wells, setWells] = useState<WellData[]>([]);
  const [selectedWell, setSelectedWell] = useState<string>('Well1');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Visualization Deck State
  const [activeFigureJson, setActiveFigureJson] = useState<string | null>(null);
  const [netPayData, setNetPayData] = useState<NetPayKPIs | null>(null);
  const [sweetspotsData, setSweetspotsData] = useState<SweetspotScanResult | null>(null);
  const [citations, setCitations] = useState<Array<{ well_name: string; formation_tops: string; lithology_notes: string }>>([]);

  const [messages, setMessages] = useState<MessageItem[]>([
    {
      id: 'init-1',
      role: 'assistant',
      content: `### Multi-Well Asset Workspace Ready
Both field wells are loaded in memory for integrated subsurface evaluation:

* **Well 1 (Target Alpha · 0–2500m MD)**: Prolific shoreface gas sandstone interval between **1850m and 1950m** (primary pay zone: 1906.1m – 1914.6m).
* **Well 2 (Exploration Beta · 1176–3960m MD)**: Deep exploration section featuring **28 stacked hydrocarbon sweet spots** between **3590m and 3850m**.

Ask any question about either well individually, request a cross-well comparison (e.g. *"Compare the reservoir sweet spots in both Well 1 and Well 2"*), or explore interactive 1D log curves and 3D subsurface models on the right.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
  ]);

  useEffect(() => {
    const fetchWells = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/wells`);
        if (res.ok) {
          const data = await res.json();
          setWells(data.wells || []);
        }
      } catch (err) {
        console.warn('Backend connecting...', err);
      }
    };

    fetchWells();
    const interval = setInterval(fetchWells, 10000);
    return () => clearInterval(interval);
  }, []);

  const handleSendMessage = async (text: string) => {
    const userMsg: MessageItem = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

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
        });
      }

      // Update active figure if returned (PlotlyDeck derives the kind from the figure spec)
      if (data.figures && data.figures.length > 0) {
        setActiveFigureJson(data.figures[data.figures.length - 1]);
      }

      const botMsg: MessageItem = {
        id: `bot-${Date.now()}`,
        role: 'assistant',
        content: data.text || 'Calculation completed successfully.',
        toolCalls: data.tool_calls || [],
        figures: data.figures || [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setMessages(prev => [...prev, botMsg]);
    } catch (err: any) {
      const errorMsg: MessageItem = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `Error: Unable to reach backend engine (${err.message}). Ensure server is running on http://127.0.0.1:8000.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="h-screen w-screen flex bg-[#F8FAFC] text-slate-900 overflow-hidden font-sans">
      {/* Panel 1: Left 50% - Conversational Petrophysicist */}
      <div className="w-1/2 h-full flex flex-col">
        <ChatStream
          messages={messages}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
          selectedWell={selectedWell}
          onSelectWell={setSelectedWell}
          wells={wells}
        />
      </div>

      {/* Panel 2: Right 50% - Dynamic Inspection & Plotly Deck */}
      <div className="w-1/2 h-full flex flex-col">
        <PlotlyDeck
          activeFigureJson={activeFigureJson}
          netPayData={netPayData}
          sweetspotsData={sweetspotsData}
          citations={citations}
          selectedWell={selectedWell}
          onSelectWell={setSelectedWell}
        />
      </div>
    </div>
  );
};

export default App;
