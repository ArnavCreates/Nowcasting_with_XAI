import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, 
  User, 
  Send, 
  Sparkles, 
  Terminal, 
  ArrowRight, 
  RefreshCw 
} from 'lucide-react';

export const SimulatorChat = ({ riskData }) => {
  const [messages, setMessages] = useState([
    {
      id: '1',
      sender: 'assistant',
      text: 'Welcome to the Nowcast Scenario Copilot. Adjust telemetry sliders on the left or select a prompt below to evaluate model sensitivity.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputQuery, setInputQuery] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking]);

  useEffect(() => {
    if (!riskData) return;
    const { riskScore, hazardType, confidence } = riskData;

    let insight = `Model recalculated frame: ${hazardType} hazard detected at ${riskScore}/100 risk score (Confidence: ${
      confidence ? (confidence * 100).toFixed(0) : '--'
    }%).`;
    
    if (riskScore > 70) {
      insight += ' Heavy moisture flux convergence detected. Operational threshold exceeded.';
    } else if (riskScore > 40) {
      insight += ' Elevated convective potential. Monitor upper-level moisture trends.';
    }

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now().toString(),
        sender: 'assistant',
        text: insight,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  }, [riskData?.riskScore, riskData?.hazardType]);

  const handleSend = (textToSend) => {
    const query = textToSend || inputQuery;
    if (!query.trim()) return;

    const userMsg = {
      id: Date.now().toString(),
      sender: 'user',
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    if (!textToSend) setInputQuery('');
    setIsThinking(true);

    setTimeout(() => {
      setIsThinking(false);
      const aiResponse = {
        id: (Date.now() + 1).toString(),
        sender: 'assistant',
        text: `Inference Query Result for "${query}": The model indicates precipitation rate and integrated water vapor are the dominant drivers in this scenario array.`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages((prev) => [...prev, aiResponse]);
    }, 500);
  };

  const promptSuggestions = [
    'Explain primary risk driver',
    'What if wind increases?',
    'Recommended response plan',
  ];

  return (
    <div className="p-5 bg-white text-slate-900 rounded-2xl border border-slate-200 shadow-sm flex flex-col h-105">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 pb-3 mb-3">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-blue-50 border border-blue-200 text-blue-600">
            <Sparkles className="w-4 h-4 animate-pulse" />
          </div>
          <div>
            <h3 className="font-bold text-xs text-slate-800 uppercase tracking-wider">
              Scenario Copilot
            </h3>
            <p className="text-[10px] text-slate-500">
              Neural Network Sensitivity Assistant
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] font-semibold text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-0.5 rounded-full">
          <Terminal className="w-3 h-3 text-emerald-600" />
          <span>Engine Active</span>
        </div>
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin scrollbar-thumb-slate-200">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex items-start gap-2 ${
              msg.sender === 'user' ? 'flex-row-reverse' : 'flex-row'
            }`}
          >
            <div
              className={`p-1.5 rounded-lg border text-xs shrink-0 ${
                msg.sender === 'user'
                  ? 'bg-blue-600 border-blue-600 text-white'
                  : 'bg-slate-100 border-slate-200 text-blue-600'
              }`}
            >
              {msg.sender === 'user' ? (
                <User className="w-3.5 h-3.5" />
              ) : (
                <Bot className="w-3.5 h-3.5" />
              )}
            </div>

            <div
              className={`max-w-[80%] p-3 rounded-xl text-xs leading-relaxed border ${
                msg.sender === 'user'
                  ? 'bg-blue-600 border-blue-600 text-white rounded-tr-none'
                  : 'bg-slate-50 border-slate-200 text-slate-800 rounded-tl-none'
              }`}
            >
              <p>{msg.text}</p>
              <span className={`block text-[9px] mt-1.5 text-right opacity-80 ${
                msg.sender === 'user' ? 'text-blue-100' : 'text-slate-400'
              }`}>
                {msg.timestamp}
              </span>
            </div>
          </div>
        ))}

        {isThinking && (
          <div className="flex items-center gap-2 text-xs text-slate-500 bg-slate-50 p-2.5 rounded-lg border border-slate-200 w-fit">
            <RefreshCw className="w-3.5 h-3.5 text-blue-600 animate-spin" />
            <span>Analyzing feature attribution...</span>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      {/* Shortcut Prompt Pills */}
      <div className="flex gap-1.5 my-2 overflow-x-auto pb-1 scrollbar-none">
        {promptSuggestions.map((prompt, idx) => (
          <button
            key={idx}
            type="button"
            onClick={() => handleSend(prompt)}
            className="text-[10px] whitespace-nowrap bg-slate-100 hover:bg-slate-200 border border-slate-200 text-slate-700 px-2.5 py-1 rounded-full transition-colors flex items-center gap-1 shrink-0 font-medium"
          >
            <span>{prompt}</span>
            <ArrowRight className="w-2.5 h-2.5 text-blue-600" />
          </button>
        ))}
      </div>

      {/* Input Bar */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          handleSend();
        }}
        className="flex gap-2"
      >
        <input
          type="text"
          placeholder="Ask AI about parameter sensitivity..."
          value={inputQuery}
          onChange={(e) => setInputQuery(e.target.value)}
          className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 text-xs text-slate-900 placeholder-slate-400 focus:outline-none focus:border-blue-500 focus:bg-white transition-colors"
        />
        <button
          type="submit"
          disabled={!inputQuery.trim()}
          className="px-3.5 py-2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 shadow-sm"
        >
          <Send className="w-3.5 h-3.5" />
        </button>
      </form>
    </div>
  );
};

export default SimulatorChat;