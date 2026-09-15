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
  Cpu
} from 'lucide-react';
import { MessageItem, ToolCallItem, WellData } from '../types';

interface ChatStreamProps {
  messages: MessageItem[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  selectedWell: string;
  onSelectWell: (well: string) => void;
  wells: WellData[];
}

export const ChatStream: React.FC<ChatStreamProps> = ({
  messages,
  onSendMessage,
  isLoading,
  selectedWell,
  onSelectWell,
  wells,
}) => {
  const [inputText, setInputText] = useState('');
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
      <div className="h-12 px-4 border-b border-zinc-200 flex items-center justify-between shrink-0 bg-white">
        <div className="flex items-center gap-3">
          <div className="text-xs font-semibold text-zinc-900 tracking-tight flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            Petrophysical Copilot
          </div>
          <span className="text-zinc-300 text-xs">|</span>
          <span className="text-[11px] text-zinc-500 font-mono">Gemini 3.5 Flash · Unified Asset Copilot</span>
        </div>

        {/* Unified Multi-Well Asset Selector */}
        <div className="flex items-center gap-1.5 font-mono text-[11px] bg-zinc-50 border border-zinc-200 rounded-md p-1 text-zinc-600">
          <span className="text-zinc-400 uppercase text-[10px] tracking-wider font-semibold px-1">Active:</span>
          <button
            onClick={() => onSelectWell('Well1')}
            className={`flex items-center gap-1.5 px-2 py-0.5 rounded transition-colors cursor-pointer ${
              selectedWell === 'Well1'
                ? 'bg-white text-zinc-900 font-semibold shadow-xs border border-zinc-200'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${selectedWell === 'Well1' ? 'bg-emerald-500' : 'bg-zinc-300'}`} />
            <span>Well 1 (0–2500m)</span>
          </button>
          <span className="text-zinc-300">·</span>
          <button
            onClick={() => onSelectWell('Well2')}
            className={`flex items-center gap-1.5 px-2 py-0.5 rounded transition-colors cursor-pointer ${
              selectedWell === 'Well2'
                ? 'bg-white text-zinc-900 font-semibold shadow-xs border border-zinc-200'
                : 'text-zinc-600 hover:text-zinc-900'
            }`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${selectedWell === 'Well2' ? 'bg-emerald-500' : 'bg-zinc-300'}`} />
            <span>Well 2 (1176–3960m)</span>
          </button>
        </div>
      </div>

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
            placeholder="Ask about Well 1, Well 2, compare sweet spots across the asset, or request 1D/3D plots..."
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
