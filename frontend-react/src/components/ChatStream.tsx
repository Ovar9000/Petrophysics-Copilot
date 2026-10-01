import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowUp,
  Terminal,
  ChevronDown,
  ChevronRight,
  Loader2,
  Database,
  Upload,
  CheckCircle2,
  AlertCircle,
  Eye,
  Lightbulb,
  PanelRightOpen,
  PanelRightClose
} from 'lucide-react';
import { Guide, MessageItem, ToolCallItem, WellData } from '../types';
import type { UploadStatus } from '../App';

interface ChatStreamProps {
  messages: MessageItem[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  selectedWell: string;
  onSelectWell: (well: string) => void;
  wells: WellData[];
  onUploadFiles: (files: File[]) => void;
  uploads: UploadStatus[];
  deckOpen: boolean;
  canToggleDeck: boolean;
  onToggleDeck: () => void;
}

// Drop box for .las files: drag-and-drop or click to browse.
const WellDropZone: React.FC<{ onFiles: (files: File[]) => void; uploads: UploadStatus[]; disabled: boolean }> = ({
  onFiles,
  uploads,
  disabled
}) => {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = (list: FileList | null) => {
    const files = Array.from(list || []);
    if (files.length) onFiles(files);
  };

  return (
    <div className="px-3 pt-3">
      <div
        role="button"
        tabIndex={0}
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && !disabled && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) accept(e.dataTransfer.files);
        }}
        className={`flex items-center justify-center gap-2 rounded-lg border-2 border-dashed px-3 py-2.5 text-xs cursor-pointer transition-colors ${
          dragging ? 'border-emerald-500 bg-emerald-50 text-emerald-800' : 'border-zinc-300 bg-zinc-50/60 text-zinc-500 hover:border-zinc-400 hover:bg-zinc-50'
        }`}
        title="Drop .las well log files here, or click to browse"
      >
        <Upload className="w-4 h-4 shrink-0" />
        <span>
          <span className="font-semibold text-zinc-700">Drop .las well logs here</span> or click to browse
        </span>
        <input
          ref={inputRef}
          type="file"
          accept=".las,.LAS"
          multiple
          className="hidden"
          onChange={(e) => {
            accept(e.target.files);
            e.target.value = '';
          }}
        />
      </div>
      {uploads.length > 0 && (
        <ul className="mt-1.5 space-y-0.5 text-[11px] font-mono">
          {uploads.slice(-4).map((u) => (
            <li key={u.name} className="flex items-center gap-1.5 text-zinc-600">
              {u.state === 'uploading' && <Loader2 className="w-3 h-3 animate-spin text-zinc-500" />}
              {u.state === 'done' && <CheckCircle2 className="w-3 h-3 text-emerald-600" />}
              {u.state === 'error' && <AlertCircle className="w-3 h-3 text-red-600" />}
              <span className="truncate">{u.name}</span>
              {u.state === 'done' && <span className="text-emerald-700">loaded as {u.detail}</span>}
              {u.state === 'error' && <span className="text-red-700 truncate">{u.detail}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

// "Look for" notes + clickable next questions under an answer.
const GuidePanel: React.FC<{ guide: Guide; onAsk: (prompt: string) => void; disabled: boolean }> = ({ guide, onAsk, disabled }) => {
  if (!guide.observe?.length && !guide.next?.length) return null;
  return (
    <div className="mt-2.5 pt-2.5 border-t border-zinc-100 space-y-2">
      {guide.observe?.length > 0 && (
        <div>
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider font-semibold text-zinc-500 mb-1">
            <Eye className="w-3 h-3" /> What to look at
          </div>
          <ul className="list-disc pl-4 space-y-0.5 text-zinc-700">
            {guide.observe.map((o) => (
              <li key={o}>{o}</li>
            ))}
          </ul>
        </div>
      )}
      {guide.next?.length > 0 && (
        <div>
          <div className="flex items-center gap-1 text-[10px] uppercase tracking-wider font-semibold text-zinc-500 mb-1">
            <Lightbulb className="w-3 h-3" /> Ask next
          </div>
          <div className="flex flex-col gap-1">
            {guide.next.map((n) => (
              <button
                key={n.prompt}
                onClick={() => onAsk(n.prompt)}
                disabled={disabled}
                title={n.why}
                className="text-left px-2.5 py-1.5 rounded-md border border-zinc-200 bg-zinc-50 hover:bg-emerald-50 hover:border-emerald-300 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <span className="text-zinc-900 font-medium">{n.prompt}</span>
                {n.why && <span className="block text-[11px] text-zinc-500">{n.why}</span>}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export const ChatStream: React.FC<ChatStreamProps> = ({
  messages,
  onSendMessage,
  isLoading,
  selectedWell,
  onSelectWell,
  wells,
  onUploadFiles,
  uploads,
  deckOpen,
  canToggleDeck,
  onToggleDeck,
}) => {
  const [inputText, setInputText] = useState('');
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  // Suggestions are only offered on the latest assistant answer.
  const lastAssistantIdx = messages.map((m) => m.role).lastIndexOf('assistant');

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const toggleToolExpand = (key: string) => {
    setExpandedTools(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSubmit = () => {
    if (!inputText.trim() || isLoading) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const renderToolBadge = (tool: ToolCallItem, msgIndex: number, toolIndex: number) => {
    const key = `${msgIndex}-${toolIndex}`;
    const isExpanded = !!expandedTools[key];
    const isRAG = tool.name === 'query_geology_metadata';

    return (
      <div key={key} className="my-1.5 border border-zinc-200 rounded-md bg-zinc-50 overflow-hidden text-xs">
        <div 
          onClick={() => toggleToolExpand(key)}
          className="px-2.5 py-1.5 flex items-center justify-between cursor-pointer hover:bg-zinc-100 transition-colors"
        >
          <div className="flex items-center gap-2 truncate">
            {isRAG ? (
              <Database className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
            ) : (
              <Terminal className="w-3.5 h-3.5 text-zinc-500 shrink-0" />
            )}
            <span className="font-mono text-zinc-700 font-medium">
              {tool.name}
            </span>
            {tool.via && (
              <span className="px-1.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-100 text-[10px] font-mono shrink-0">
                via {tool.via}
              </span>
            )}
            <span className="text-[11px] text-zinc-400 font-mono truncate max-w-[280px]">
              {JSON.stringify(tool.args)}
            </span>
          </div>
          <button className="text-zinc-400 hover:text-zinc-600 p-0.5">
            {isExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
          </button>
        </div>

        {isExpanded && (
          <div className="p-2.5 border-t border-zinc-200 bg-white font-mono text-[11px] space-y-1.5">
            <div>
              <span className="text-zinc-400 uppercase text-[10px]">Parameters:</span>
              <pre className="mt-0.5 text-zinc-800 bg-zinc-50 p-2 rounded border border-zinc-200 overflow-x-auto">
                {JSON.stringify(tool.args, null, 2)}
              </pre>
            </div>
            {tool.result && (
              <div>
                <span className="text-zinc-400 uppercase text-[10px]">Result Payload:</span>
                <pre className="mt-0.5 text-zinc-800 bg-zinc-50 p-2 rounded border border-zinc-200 max-h-40 overflow-y-auto">
                  {JSON.stringify(tool.result, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex-1 flex flex-col bg-white border-r border-zinc-200 h-full overflow-hidden">
      {/* Minimalist Top Bar */}
      <div className="h-12 px-4 border-b border-zinc-200 flex items-center justify-between gap-3 shrink-0 bg-white">
        <div className="flex items-center gap-3 min-w-0">
          <div className="text-xs font-semibold text-zinc-900 tracking-tight flex items-center gap-1.5 whitespace-nowrap">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Petrophysical Copilot
          </div>
          {!deckOpen && (
            <span className="text-[11px] text-zinc-500 font-mono whitespace-nowrap truncate hidden lg:inline">
              | Gemini · answers from tool results
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 shrink-0">
          {/* Active well: any loaded well, including dropped-in files */}
          <label className="flex items-center gap-1.5 font-mono text-[11px] bg-zinc-50 border border-zinc-200 rounded-md px-2 py-1 text-zinc-600">
            <span className="text-zinc-400 uppercase text-[10px] tracking-wider font-semibold hidden xl:inline">Active:</span>
            <span className={`w-1.5 h-1.5 rounded-full ${selectedWell ? 'bg-emerald-500' : 'bg-zinc-300'}`} />
            <select
              value={selectedWell}
              onChange={(e) => onSelectWell(e.target.value)}
              className="bg-transparent text-zinc-900 font-semibold focus:outline-none cursor-pointer max-w-[220px]"
              disabled={wells.length === 0}
            >
              {wells.length === 0 && <option value="">no wells loaded</option>}
              {wells.map((w) => (
                <option key={w.well_id} value={w.well_id}>
                  {w.well_id} ({Math.round(w.start_depth)}–{Math.round(w.stop_depth)} m)
                </option>
              ))}
            </select>
          </label>
          {canToggleDeck && (
            <button
              onClick={onToggleDeck}
              className="flex items-center gap-1 px-2 py-1 rounded-md border border-zinc-200 bg-white hover:bg-zinc-50 text-zinc-600 text-[11px] font-mono cursor-pointer"
              title={deckOpen ? 'Hide the plot window' : 'Show the plot window'}
            >
              {deckOpen ? <PanelRightClose className="w-3.5 h-3.5" /> : <PanelRightOpen className="w-3.5 h-3.5" />}
              <span className="hidden xl:inline">{deckOpen ? 'Hide plots' : 'Show plots'}</span>
            </button>
          )}
        </div>
      </div>

      <WellDropZone onFiles={onUploadFiles} uploads={uploads} disabled={isLoading} />

      {/* Messages Stream */}
      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
        {messages.map((msg, mIdx) => {
          const isUser = msg.role === 'user';
          return (
            <div 
              key={msg.id || mIdx}
              className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[88%] rounded-lg p-3.5 text-xs transition-all ${
                isUser 
                  ? 'bg-zinc-100 text-zinc-900 border border-zinc-200/80 font-medium' 
                  : 'bg-white text-zinc-800 border border-zinc-200 shadow-xs'
              }`}>
                {/* Collapsible Tool Badges */}
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="mb-2">
                    {msg.toolCalls.map((t, tIdx) => renderToolBadge(t, mIdx, tIdx))}
                  </div>
                )}

                {/* Markdown text response */}
                <div className="prose prose-zinc prose-xs max-w-none text-zinc-800 leading-relaxed font-sans">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    components={{
                      table: ({ node, ...props }) => (
                        <div className="overflow-x-auto my-2 border border-zinc-200 rounded">
                          <table className="min-w-full divide-y divide-zinc-200 text-[11px] font-mono" {...props} />
                        </div>
                      ),
                      th: ({ node, ...props }) => (
                        <th className="bg-zinc-50 px-2.5 py-1.5 text-left text-zinc-700 font-semibold" {...props} />
                      ),
                      td: ({ node, ...props }) => (
                        <td className="px-2.5 py-1.5 border-t border-zinc-200 text-zinc-800" {...props} />
                      ),
                      p: ({ node, ...props }) => (
                        <p className="mb-1.5 last:mb-0" {...props} />
                      ),
                      ul: ({ node, ...props }) => (
                        <ul className="list-disc pl-4 my-1.5 space-y-0.5" {...props} />
                      ),
                      ol: ({ node, ...props }) => (
                        <ol className="list-decimal pl-4 my-1.5 space-y-0.5" {...props} />
                      ),
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>
                </div>

                {!isUser && msg.guide && mIdx === lastAssistantIdx && (
                  <GuidePanel guide={msg.guide} onAsk={onSendMessage} disabled={isLoading} />
                )}

                <div className="mt-1 text-[10px] text-zinc-400 font-mono text-right">
                  {msg.timestamp}
                </div>
              </div>
            </div>
          );
        })}

        {isLoading && (
          <div className="flex items-center gap-2 text-zinc-500 text-xs font-mono py-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-zinc-600" />
            <span>Computing petrophysical response...</span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Bar */}
      <div className="p-3 border-t border-zinc-200 bg-white shrink-0">
        <div className="flex items-center gap-2 rounded-lg border border-zinc-200 bg-zinc-50/50 px-3 py-2 focus-within:border-zinc-400 focus-within:bg-white transition-all shadow-xs">
          <textarea
            ref={textareaRef}
            rows={1}
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={selectedWell ? `Ask about ${selectedWell}, e.g. "Show the 1D log", "Scan for sweet spots"...` : 'Drop a .las file above to get started...'}
            className="flex-1 bg-transparent text-xs text-zinc-900 placeholder-zinc-400 focus:outline-none resize-none font-sans py-1 max-h-28"
            disabled={isLoading}
          />

          <button
            onClick={handleSubmit}
            disabled={!inputText.trim() || isLoading}
            className="w-7 h-7 rounded-md bg-zinc-900 hover:bg-zinc-800 text-white flex items-center justify-center disabled:opacity-30 disabled:cursor-not-allowed transition-colors shrink-0"
            title="Transmit (Enter ↵)"
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
