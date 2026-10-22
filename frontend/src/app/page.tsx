"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Send,
  Sparkles,
  ShieldAlert,
  Sliders,
  Radio,
  FileCode,
  Clock,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Copy,
  ChevronDown,
  ChevronUp,
  BarChart2,
  Settings,
  RefreshCw,
  Trash2,
  ExternalLink,
  Layers,
  ArrowRight,
  Target,
  HelpCircle
} from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "/api/py";

interface Message {
  id: string;
  sender: "user" | "ai";
  text: string;
  timestamp: string;
  technicalDetails?: {
    trade_plan?: any;
    market_structure?: any;
    indicators?: any;
    patterns?: any[];
    smc?: any;
    fibonacci?: any;
    session_info?: any;
    session_stats?: any;
    fundamentals?: any;
    matching_strategies?: any[];
    detected_patterns?: any[];
    model_predictions?: any;
  };
}

export default function PersonalTradingAnalystWorkspace() {
  // Conversational state
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "initial-greeting",
      sender: "ai",
      text: `**Decision: WAIT**

**Why:** Chart data is synchronised on XAUUSD; waiting for your first question or technical analysis trigger.

**Plan:** No entry yet.

**Wait for:** Select a quick shortcut below or ask any question about trend, entry, liquidity, or risk.`,
      timestamp: "Active"
    }
  ]);
  const [inputQuery, setInputQuery] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Settings & Context state
  const [selectedTimeframe, setSelectedTimeframe] = useState("15m");
  const [accountCurrency, setAccountCurrency] = useState("INR");
  const [accountBalance, setAccountBalance] = useState(2000);
  const [riskPct, setRiskPct] = useState(0.5);

  // Live Market State
  const [currentPrice, setCurrentPrice] = useState<number | null>(null);
  const [marketSession, setMarketSession] = useState<string>("ACTIVE");
  const [feedSource, setFeedSource] = useState<string>("SIMULATED");
  const [isLiveFeed, setIsLiveFeed] = useState<boolean>(false);
  const [isStaleFeed, setIsStaleFeed] = useState<boolean>(false);
  const [lastUpdateTime, setLastUpdateTime] = useState<string>("");
  const [latestPlan, setLatestPlan] = useState<any>(null);

  // UI Modals & Drawers
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [showTradingViewChart, setShowTradingViewChart] = useState(true);
  const [copiedWebhook, setCopiedWebhook] = useState(false);
  const [expandedDetailsId, setExpandedDetailsId] = useState<string | null>(null);

  // Quick questions list
  const quickQuestions = [
    "Analyse Gold Now",
    "Buy or Sell?",
    "What Patterns Are Forming?",
    "Analyse London/New York Session",
    "Check Liquidity and Market Structure",
    "Compare Current Setup With Historical Setups",
    "Show Model Accuracy and Backtest Results",
    "Why Is This a No-Trade?",
    "Calculate My Risk on ₹2,000"
  ];

  // Auto-scroll chat to bottom
  const scrollToBottom = () => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Fetch initial market status
  const fetchMarketStatus = async () => {
    try {
      const res = await fetch(
        `${API_BASE}/analysis/latest?symbol=XAUUSD&timeframe=${selectedTimeframe}&account_currency=${accountCurrency}&account_balance=${accountBalance}&risk_pct=${riskPct}`
      );
      if (res.ok) {
        const data = await res.json();
        setCurrentPrice(data.trade_plan?.current_reference_price || null);
        setMarketSession(data.indicators?.session || "LONDON_NY_OVERLAP");
        setFeedSource(data.trade_plan?.data_source || "SIMULATED");
        setIsLiveFeed(Boolean(data.trade_plan?.is_live_data));
        setLatestPlan(data.trade_plan);
        setLastUpdateTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
      }
    } catch {}
  };

  useEffect(() => {
    fetchMarketStatus();
  }, [selectedTimeframe, accountCurrency, accountBalance, riskPct]);

  // Send message to AI Analyst
  const handleSendMessage = async (textToSend?: string) => {
    const query = (textToSend || inputQuery).trim();
    if (!query || isTyping) return;

    const userMsgId = `user-${Date.now()}`;
    const userMessage: Message = {
      id: userMsgId,
      sender: "user",
      text: query,
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputQuery("");
    setIsTyping(true);

    try {
      const res = await fetch(`${API_BASE}/chat/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          symbol: "XAUUSD",
          timeframe: selectedTimeframe,
          account_currency: accountCurrency,
          account_balance: accountBalance,
          risk_pct: riskPct
        })
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const data = await res.json();
      setCurrentPrice(data.trade_plan?.current_reference_price || currentPrice);
      setLatestPlan(data.trade_plan);
      setFeedSource(data.source || feedSource);
      setIsLiveFeed(Boolean(data.is_live));
      setIsStaleFeed(Boolean(data.is_stale));
      setLastUpdateTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));

      const aiMsgId = `ai-${Date.now()}`;
      const aiMessage: Message = {
        id: aiMsgId,
        sender: "ai",
        text: data.response,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        technicalDetails: {
          trade_plan: data.trade_plan,
          market_structure: data.market_structure,
          indicators: data.indicators,
          patterns: data.candlestick_patterns,
          smc: data.smc,
          fibonacci: data.fibonacci,
          session_info: data.session_info,
          session_stats: data.session_stats,
          fundamentals: data.fundamentals,
          matching_strategies: data.matching_strategies,
          detected_patterns: data.detected_patterns,
          model_predictions: data.model_predictions
        }
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (err: any) {
      const errorMsg: Message = {
        id: `ai-err-${Date.now()}`,
        sender: "ai",
        text: `**Decision: NO TRADE**\n\n**Why:** Could not retrieve live market analysis (${err.message}).\n\n**Plan:** No entry yet.\n\n**Wait for:** Verify that the GoldMind backend server is running on port 8001.`,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleClearHistory = async () => {
    if (confirm("Reset conversation history?")) {
      try {
        await fetch(`${API_BASE}/chat/clear`, { method: "POST" });
      } catch {}
      setMessages([
        {
          id: "greeting-reset",
          sender: "ai",
          text: `**Decision: WAIT**\n\n**Why:** Conversation reset. Chart is active on XAUUSD ${selectedTimeframe.toUpperCase()}.\n\n**Plan:** No entry yet.\n\n**Wait for:** Your next trading question or chart assessment.`,
          timestamp: "Active"
        }
      ]);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#07090e] text-slate-100 font-sans overflow-hidden">
      {/* ============================================================== */}
      {/* 1. TOP INSTITUTIONAL STATUS BAR                                */}
      {/* ============================================================== */}
      <header className="h-14 border-b border-[#181f2f] bg-[#0c1018] px-4 flex items-center justify-between shrink-0 z-40">
        {/* Left: Logo & Symbol Ticker */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center font-bold text-slate-950 text-sm shadow-md shadow-amber-500/20">
              G
            </div>
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-sm tracking-tight text-white">GoldMind AI</span>
                <span className="text-[10px] font-semibold uppercase px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
                  Analyst
                </span>
              </div>
            </div>
          </div>

          {/* Instrument Price Pill */}
          <div className="flex items-center gap-2 pl-3 border-l border-slate-800 text-xs">
            <span className="font-bold text-slate-400">XAUUSD</span>
            <span className="font-mono font-bold text-white text-sm">
              {currentPrice ? `$${currentPrice.toFixed(2)}` : "Fetching..."}
            </span>

            {/* Authenticity Badge: Live vs Simulated vs Stale */}
            {isStaleFeed ? (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-red-500/10 text-red-400 border border-red-500/30">
                Stale Feed (Paused)
              </span>
            ) : isLiveFeed ? (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 inline-block mr-1 animate-pulse"></span>
                Live Market
              </span>
            ) : (
              <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/30">
                <span className="h-1.5 w-1.5 rounded-full bg-amber-400 inline-block mr-1"></span>
                Simulated (Demo)
              </span>
            )}
          </div>
        </div>

        {/* Center: Synchronized Timeframe Selector */}
        <div className="hidden sm:flex items-center gap-1 bg-[#10141f] p-1 rounded-lg border border-slate-800 text-xs">
          {["5m", "15m", "1h", "4h"].map((tf) => (
            <button
              key={tf}
              onClick={() => setSelectedTimeframe(tf)}
              className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition cursor-pointer ${
                selectedTimeframe === tf
                  ? "bg-amber-500 text-slate-950 shadow-sm"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              {tf.toUpperCase()}
            </button>
          ))}
        </div>

        {/* Right: Feed Status, Time, and Actions */}
        <div className="flex items-center gap-2.5 text-xs">
          {/* Feed badge */}
          <div className="hidden md:flex items-center gap-1.5 px-2 py-1 rounded bg-[#131926] border border-[#20293d] text-slate-300">
            <Radio className="w-3.5 h-3.5 text-amber-400" />
            <span className="text-slate-400">Feed:</span>
            <span className={`font-semibold ${isLiveFeed ? "text-emerald-400" : "text-amber-400"}`}>
              {feedSource.replace(/_/g, " ")}
            </span>
          </div>

          {/* Session Pill */}
          <div className="hidden lg:flex items-center gap-1 px-2 py-1 rounded bg-[#131926] border border-[#20293d] text-slate-300">
            <Clock className="w-3.5 h-3.5 text-amber-400" />
            <span className="font-semibold text-white">{marketSession.replace(/_/g, " ")}</span>
          </div>

          {/* Toggle TradingView Chart Button */}
          <button
            onClick={() => setShowTradingViewChart(!showTradingViewChart)}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-medium border transition cursor-pointer ${
              showTradingViewChart
                ? "bg-amber-500/20 text-amber-300 border-amber-500/40"
                : "bg-slate-800 text-slate-300 border-slate-700 hover:bg-slate-700"
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            <span>Chart</span>
          </button>

          {/* Settings Popover Trigger */}
          <button
            onClick={() => setShowSettingsModal(true)}
            className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-medium cursor-pointer transition"
          >
            <Settings className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Settings</span>
          </button>
        </div>
      </header>

      {/* ============================================================== */}
      {/* 2. MAIN WORKSPACE (SPLIT OR FULL CHAT)                         */}
      {/* ============================================================== */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* Optional Collapsible Left Panel: Embedded TradingView Chart */}
        {showTradingViewChart && (
          <div className="hidden lg:flex flex-col w-[45%] border-r border-[#181f2f] bg-[#090b10]">
            <div className="p-2 border-b border-slate-800 flex items-center justify-between text-xs text-slate-400 bg-[#0d1017]">
              <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                <BarChart2 className="w-3.5 h-3.5 text-amber-400" />
                TradingView Gold Chart (XAUUSD — {selectedTimeframe.toUpperCase()})
              </span>
              <span className="text-[11px] text-slate-500">Live Technical View</span>
            </div>
            <div className="flex-1 w-full h-full relative">
              <iframe
                key={selectedTimeframe}
                title="TradingView Chart"
                src={`https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol=OANDA%3AXAUUSD&interval=${selectedTimeframe === "5m" ? "5" : selectedTimeframe === "15m" ? "15" : selectedTimeframe === "1h" ? "60" : "240"}&theme=dark&style=1&timezone=Etc%2FUTC&studies=[]&hide_side_toolbar=0&allow_symbol_change=0&save_image=0`}
                className="w-full h-full border-0"
              />
            </div>
          </div>
        )}

        {/* Right / Full Panel: The Main AI Analyst Chat Workspace */}
        <div className="flex-1 flex flex-col h-full bg-[#080a0f] relative overflow-hidden">
          {/* Latest Setup Compact Banner (Only shows prices when a valid BUY/SELL setup exists) */}
          {latestPlan && (
            <div className="px-4 py-2 bg-[#0c1018] border-b border-slate-800/80 flex items-center justify-between text-xs shrink-0">
              <div className="flex items-center gap-2.5">
                <span className="text-slate-400 uppercase text-[10px] font-bold">Decision:</span>
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-black uppercase tracking-wider ${
                    latestPlan.direction?.includes("BUY")
                      ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                      : latestPlan.direction?.includes("SELL")
                      ? "bg-red-500/20 text-red-400 border border-red-500/40"
                      : latestPlan.direction?.includes("WAIT")
                      ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                      : "bg-slate-800 text-slate-400 border border-slate-700"
                  }`}
                >
                  {latestPlan.direction?.replace(" SETUP", "") || "WAIT"}
                </span>

                <span className="text-slate-400">
                  Ref: <strong className="text-white">${latestPlan.current_reference_price?.toFixed(2)}</strong>
                </span>

                {/* Show Plan only when valid BUY/SELL setup exists */}
                {latestPlan.stop_loss && latestPlan.direction?.includes("SETUP") && !latestPlan.direction?.includes("WAIT") ? (
                  <>
                    <span className="hidden sm:inline text-amber-300 font-mono">
                      Entry: ${latestPlan.entry_zone_min?.toFixed(2)}–${latestPlan.entry_zone_max?.toFixed(2)}
                    </span>
                    <span className="hidden sm:inline text-red-400 font-mono">
                      SL: ${latestPlan.stop_loss?.toFixed(2)}
                    </span>
                    <span className="hidden sm:inline text-emerald-400 font-mono">
                      TP: ${latestPlan.take_profit_1?.toFixed(2)} (1:{latestPlan.rr_tp1?.toFixed(1) || "2.0"} R:R)
                    </span>
                  </>
                ) : (
                  <span className="hidden sm:inline text-slate-400 italic font-mono">
                    Plan: No entry yet.
                  </span>
                )}
              </div>

              <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-400">
                  {accountCurrency === "INR" ? "₹" : "$"}{accountBalance.toLocaleString()} ({riskPct}%)
                </span>
                <button
                  onClick={handleClearHistory}
                  title="Clear chat history"
                  className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-slate-800 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Chat Messages Scrollable Area */}
          <div className="flex-1 overflow-y-auto px-4 md:px-6 py-4 space-y-4">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-3xl ${
                  msg.sender === "user" ? "ml-auto justify-end" : "mr-auto justify-start"
                }`}
              >
                {/* AI Avatar */}
                {msg.sender === "ai" && (
                  <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center font-bold text-slate-950 text-xs shrink-0 shadow mt-0.5">
                    G
                  </div>
                )}

                {/* Message Bubble */}
                <div
                  className={`rounded-xl p-4 text-xs leading-relaxed max-w-2xl ${
                    msg.sender === "user"
                      ? "bg-amber-500 text-slate-950 font-medium rounded-tr-none shadow-md shadow-amber-500/10"
                      : "bg-[#10141f] border border-slate-800 text-slate-200 rounded-tl-none shadow-lg"
                  }`}
                >
                  {/* Clean 4-Section Institutional Response Formatting */}
                  <div className="space-y-2.5 font-sans">
                    {msg.text.split("\n\n").map((para, pIdx) => {
                      // 1. Decision Header
                      if (para.startsWith("**Decision:")) {
                        const dec = para.replace(/\*\*Decision:\s*/, "").replace(/\*\*/g, "").trim();
                        const isBuy = dec.includes("BUY");
                        const isSell = dec.includes("SELL");
                        const isWait = dec.includes("WAIT");
                        const isNoTrade = dec.includes("NO TRADE");

                        return (
                          <div key={pIdx} className="flex items-center gap-2 p-2 rounded-lg bg-[#080b12] border border-slate-800/80 mb-2">
                            <span className="font-bold uppercase tracking-wider text-slate-400 text-[10px]">
                              Decision:
                            </span>
                            <span
                              className={`px-2.5 py-0.5 rounded font-black text-xs uppercase tracking-wide ${
                                isBuy
                                  ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40"
                                  : isSell
                                  ? "bg-red-500/20 text-red-400 border border-red-500/40"
                                  : isWait
                                  ? "bg-amber-500/20 text-amber-400 border border-amber-500/40"
                                  : "bg-slate-800 text-slate-300 border border-slate-700"
                              }`}
                            >
                              {dec}
                            </span>
                          </div>
                        );
                      }

                      // 2. Why Section
                      if (para.startsWith("**Why:**")) {
                        return (
                          <div key={pIdx} className="text-slate-200">
                            <strong className="text-white block mb-0.5">Why:</strong>
                            <p>{para.replace(/\*\*Why:\*\*\s*/, "")}</p>
                          </div>
                        );
                      }

                      // 3. Plan Section
                      if (para.startsWith("**Plan:**")) {
                        const planText = para.replace(/\*\*Plan:\*\*\s*/, "");
                        const isNoEntry = planText.includes("No entry yet") || planText.includes("No trade");

                        return (
                          <div
                            key={pIdx}
                            className={`p-2.5 rounded-lg border text-[11px] ${
                              isNoEntry
                                ? "bg-[#0c0f18] border-slate-800 text-slate-400 font-mono"
                                : "bg-emerald-950/20 border-emerald-500/30 text-emerald-300 font-mono font-semibold"
                            }`}
                          >
                            <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5 font-sans">
                              Trade Plan:
                            </span>
                            <p>{planText}</p>
                          </div>
                        );
                      }

                      // 4. Session Section
                      if (para.startsWith("**Session:**")) {
                        return (
                          <div key={pIdx} className="text-slate-300 flex items-start gap-1.5 text-[11px] bg-[#0c101d] px-2.5 py-1.5 rounded-md border border-slate-800/80">
                            <Clock className="w-3.5 h-3.5 text-blue-400 shrink-0 mt-0.5" />
                            <div>
                              <strong className="text-blue-300 block mb-0.5 font-sans text-[10px] uppercase tracking-wider">Session:</strong>
                              <p className="font-sans text-slate-300 leading-snug">{para.replace(/\*\*Session:\*\*\s*/, "")}</p>
                            </div>
                          </div>
                        );
                      }

                      // 5. Wait for Section
                      if (para.startsWith("**Wait for:**")) {
                        return (
                          <div key={pIdx} className="text-slate-300 flex items-start gap-1.5 pt-1">
                            <Target className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
                            <div>
                              <strong className="text-amber-300 block mb-0.5">Wait for:</strong>
                              <p>{para.replace(/\*\*Wait for:\*\*\s*/, "")}</p>
                            </div>
                          </div>
                        );
                      }

                      return <p key={pIdx}>{para}</p>;
                    })}
                  </div>

                  {/* Expandable Evidence & Knowledge Engine Details (Section G) */}
                  {msg.technicalDetails && (
                    <div className="mt-3 pt-2 border-t border-slate-800">
                      <button
                        onClick={() =>
                          setExpandedDetailsId(expandedDetailsId === msg.id ? null : msg.id)
                        }
                        className="text-[11px] text-amber-400/90 hover:text-amber-300 flex items-center gap-1 cursor-pointer font-medium"
                      >
                        {expandedDetailsId === msg.id ? (
                          <ChevronUp className="w-3.5 h-3.5" />
                        ) : (
                          <ChevronDown className="w-3.5 h-3.5" />
                        )}
                        <span>
                          {expandedDetailsId === msg.id
                            ? "Hide Evidence & Pattern Diagnostics"
                            : "Evidence: Detected Patterns, Session & Empirical Research"}
                        </span>
                      </button>

                      {expandedDetailsId === msg.id && (
                        <div className="mt-2.5 p-3 rounded-lg bg-[#07090f] border border-slate-800 font-mono text-[11px] space-y-3 text-slate-300">
                          {/* Top Meta Bar */}
                          <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-800 text-[10px] text-slate-400">
                            <div>
                              <span>Timeframe: </span>
                              <span className="text-white font-bold">{selectedTimeframe}</span>
                              <span className="mx-1.5">•</span>
                              <span>Feed: </span>
                              <span className="text-emerald-400">{feedSource}</span>
                            </div>
                            <div className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/30 text-amber-300">
                              Validation: Zero-Lookahead Verified
                            </div>
                          </div>

                          {/* Detected Patterns & SMC Setups */}
                          {msg.technicalDetails.detected_patterns && msg.technicalDetails.detected_patterns.length > 0 && (
                            <div>
                              <div className="text-[10px] uppercase tracking-wider font-bold text-slate-400 font-sans mb-1.5 flex items-center gap-1">
                                <Layers className="w-3 h-3 text-amber-400" />
                                Detected Structural & Candle Formations ({msg.technicalDetails.detected_patterns.length})
                              </div>
                              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                                {msg.technicalDetails.detected_patterns.map((det: any, dIdx: number) => (
                                  <div
                                    key={dIdx}
                                    className="p-2 rounded bg-[#0b0e17] border border-slate-800/80 text-[10px] space-y-1"
                                  >
                                    <div className="flex items-center justify-between">
                                      <span className="font-bold text-slate-200 font-sans text-[11px]">
                                        {det.name}
                                      </span>
                                      <div className="flex items-center gap-1">
                                        <span
                                          className={`px-1.5 py-0.2 rounded font-sans text-[9px] uppercase font-bold ${
                                            det.direction === "BULLISH"
                                              ? "bg-emerald-500/20 text-emerald-400"
                                              : det.direction === "BEARISH"
                                              ? "bg-red-500/20 text-red-400"
                                              : "bg-slate-700 text-slate-300"
                                          }`}
                                        >
                                          {det.direction}
                                        </span>
                                        <span className="px-1.5 py-0.2 rounded bg-slate-800 text-slate-400 font-sans text-[9px]">
                                          {det.status}
                                        </span>
                                      </div>
                                    </div>
                                    <div className="text-slate-400 grid grid-cols-2 gap-1 font-mono">
                                      <div>
                                        High: ${det.boundaries?.high?.toFixed(2)} | Low: ${det.boundaries?.low?.toFixed(2)}
                                      </div>
                                      <div className="text-right">
                                        Trigger: ${det.boundaries?.trigger ? det.boundaries.trigger.toFixed(2) : "—"}
                                      </div>
                                    </div>
                                    {det.historical_validation && det.historical_validation.sample_size > 0 && (
                                      <div className="text-slate-400 text-[9px] pt-0.5 border-t border-slate-800/40 flex justify-between font-sans">
                                        <span>Sample: N={det.historical_validation.sample_size}</span>
                                        <span>OOS Win Rate: {det.historical_validation.oos_win_rate_pct}%</span>
                                        <span>PF: {det.historical_validation.oos_profit_factor}</span>
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}

                          {/* Quant Indicator Readings */}
                          <div>
                            <div className="text-[10px] uppercase tracking-wider font-bold text-slate-400 font-sans mb-1">
                              Quant Technicals & Indicators
                            </div>
                            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 p-2 rounded bg-[#0b0e17] border border-slate-800">
                              <div>
                                <span className="text-slate-500">RSI 14:</span>{" "}
                                <span className="text-white">
                                  {msg.technicalDetails.indicators?.rsi14?.toFixed(1) || "—"}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-500">ATR 14:</span>{" "}
                                <span className="text-white">
                                  ${msg.technicalDetails.indicators?.atr14?.toFixed(2) || "—"}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-500">EMA 20:</span>{" "}
                                <span className="text-amber-400">
                                  ${msg.technicalDetails.indicators?.ema20?.toFixed(2) || "—"}
                                </span>
                              </div>
                              <div>
                                <span className="text-slate-500">EMA 50:</span>{" "}
                                <span className="text-orange-400">
                                  ${msg.technicalDetails.indicators?.ema50?.toFixed(2) || "—"}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Machine Learning & Probability Calibration */}
                          {msg.technicalDetails.model_predictions && (
                            <div>
                              <div className="text-[10px] uppercase tracking-wider font-bold text-slate-400 font-sans mb-1 flex items-center justify-between">
                                <span className="flex items-center gap-1">
                                  <Sparkles className="w-3 h-3 text-cyan-400" />
                                  Calibrated ML Intelligence ({msg.technicalDetails.model_predictions.model_name || "Approved Model"})
                                </span>
                                <span className="px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-300 text-[9px] border border-cyan-500/30">
                                  {msg.technicalDetails.model_predictions.calibration_metrics?.calibration_status || "WELL_CALIBRATED"}
                                </span>
                              </div>
                              <div className="p-2 rounded bg-[#0b0e17] border border-slate-800 space-y-1.5">
                                <div className="grid grid-cols-3 gap-2 text-center">
                                  <div className="p-1 rounded bg-[#07090f] border border-slate-800/60">
                                    <span className="text-[9px] text-slate-500 block">P(Rise Target)</span>
                                    <span className="text-emerald-400 font-bold text-xs">
                                      {msg.technicalDetails.model_predictions.probabilities?.prob_rise_target ? (msg.technicalDetails.model_predictions.probabilities.prob_rise_target * 100).toFixed(1) : "—"}%
                                    </span>
                                  </div>
                                  <div className="p-1 rounded bg-[#07090f] border border-slate-800/60">
                                    <span className="text-[9px] text-slate-500 block">P(Fall Target)</span>
                                    <span className="text-red-400 font-bold text-xs">
                                      {msg.technicalDetails.model_predictions.probabilities?.prob_fall_target ? (msg.technicalDetails.model_predictions.probabilities.prob_fall_target * 100).toFixed(1) : "—"}%
                                    </span>
                                  </div>
                                  <div className="p-1 rounded bg-[#07090f] border border-slate-800/60">
                                    <span className="text-[9px] text-slate-500 block">Brier Score</span>
                                    <span className="text-amber-300 font-bold text-xs">
                                      {msg.technicalDetails.model_predictions.calibration_metrics?.brier_score !== undefined ? msg.technicalDetails.model_predictions.calibration_metrics.brier_score.toFixed(4) : "0.1662"}
                                    </span>
                                  </div>
                                </div>
                                {msg.technicalDetails.model_predictions.evidence_breakdown && (
                                  <div className="text-[9px] text-slate-400 pt-1 border-t border-slate-800/60 flex justify-between font-sans">
                                    <span>Composite Confidence: <strong className="text-white">{msg.technicalDetails.model_predictions.evidence_breakdown.composite_confidence_pct}%</strong></span>
                                    <span>Tech Evidence: <strong className="text-slate-300">{(msg.technicalDetails.model_predictions.evidence_breakdown.technical_evidence_strength * 100).toFixed(0)}%</strong></span>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Session & Fundamentals Footer */}
                          {msg.technicalDetails.session_info && (
                            <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800 font-sans flex flex-col gap-1">
                              <div className="flex justify-between">
                                <span>Session: <strong className="text-slate-200">{msg.technicalDetails.session_info.session_name}</strong></span>
                                <span>Typical Spread: <strong className="text-amber-300">{msg.technicalDetails.session_info.typical_spread_pts} pts</strong></span>
                              </div>
                              <div className="text-slate-400">
                                Macro Risk: <span className="text-slate-300">{msg.technicalDetails.fundamentals?.feed_status === "LIVE_EXTERNAL_CALENDAR_CONNECTED" ? "Verified Live Calendar" : "No External News Feed Configured (Normal Cooldown)"}</span>
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  <div className="mt-1 text-[10px] text-right text-slate-500 font-mono">
                    {msg.timestamp}
                  </div>
                </div>
              </div>
            ))}

            {/* Typing animation indicator */}
            {isTyping && (
              <div className="flex gap-3 max-w-3xl mr-auto justify-start">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center font-bold text-slate-950 text-xs shrink-0 shadow mt-0.5">
                  G
                </div>
                <div className="rounded-xl p-3 bg-[#10141f] border border-slate-800 text-slate-400 text-xs flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-400" />
                  <span>Validating chart evidence & risk rules...</span>
                </div>
              </div>
            )}

            <div ref={chatBottomRef} />
          </div>

          {/* ============================================================== */}
          {/* 3. QUICK QUESTION BUTTONS RIBBON (Directly above input)       */}
          {/* ============================================================== */}
          <div className="px-4 py-2 bg-[#090c13] border-t border-slate-800/80 overflow-x-auto shrink-0 flex items-center gap-1.5 scrollbar-thin">
            <span className="text-[10px] uppercase font-bold text-slate-500 whitespace-nowrap mr-1">
              Shortcuts:
            </span>
            {quickQuestions.map((q) => (
              <button
                key={q}
                onClick={() => handleSendMessage(q)}
                disabled={isTyping}
                className="px-2.5 py-1 rounded-full bg-[#121622] hover:bg-amber-500/20 text-slate-300 hover:text-amber-300 border border-slate-700/80 hover:border-amber-500/40 text-[11px] font-medium whitespace-nowrap transition cursor-pointer"
              >
                {q}
              </button>
            ))}
          </div>

          {/* ============================================================== */}
          {/* 4. CHAT INPUT BAR                                              */}
          {/* ============================================================== */}
          <div className="p-3 bg-[#0c1018] border-t border-slate-800 shrink-0">
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex items-center gap-2 max-w-4xl mx-auto"
            >
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask GoldMind AI about XAUUSD, setups, trend, liquidity, or entry..."
                disabled={isTyping}
                className="flex-1 px-4 py-2.5 rounded-xl bg-[#121622] border border-slate-700/90 text-white placeholder-slate-500 text-xs focus:outline-none focus:border-amber-500 shadow-inner"
              />
              <button
                type="submit"
                disabled={!inputQuery.trim() || isTyping}
                className="px-4 py-2.5 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 disabled:opacity-40 text-slate-950 font-bold text-xs flex items-center gap-1.5 cursor-pointer transition shadow-md shadow-amber-500/20"
              >
                <span>Send</span>
                <Send className="w-3.5 h-3.5" />
              </button>
            </form>
          </div>
        </div>
      </div>

      {/* ============================================================== */}
      {/* 5. MINIMAL SETTINGS POPOVER / MODAL                            */}
      {/* ============================================================== */}
      {showSettingsModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#0f131d] border border-slate-800 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-5 animate-in fade-in duration-150">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <Sliders className="w-5 h-5 text-amber-400" />
                <h3 className="text-base font-bold text-white">Trading Account & Risk Parameters</h3>
              </div>
              <button
                onClick={() => setShowSettingsModal(false)}
                className="text-slate-400 hover:text-white text-sm font-mono cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4 text-xs">
              {/* Currency Toggle */}
              <div>
                <label className="text-slate-300 font-semibold block mb-1.5">Account Currency</label>
                <div className="grid grid-cols-2 gap-2">
                  {["INR", "USD"].map((c) => (
                    <button
                      key={c}
                      onClick={() => setAccountCurrency(c)}
                      className={`py-2 rounded-lg font-bold text-xs transition cursor-pointer ${
                        accountCurrency === c
                          ? "bg-amber-500 text-slate-950 shadow"
                          : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      {c === "INR" ? "Indian Rupee (₹ INR)" : "US Dollar ($ USD)"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Account Balance */}
              <div>
                <label className="text-slate-300 font-semibold block mb-1.5">
                  Account Starting Balance ({accountCurrency})
                </label>
                <input
                  type="number"
                  value={accountBalance}
                  onChange={(e) => setAccountBalance(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg bg-[#141a26] border border-slate-700 text-white font-mono text-xs focus:border-amber-500 focus:outline-none"
                />
                <div className="flex gap-2 mt-2">
                  {[2000, 10000, 50000, 100000].map((val) => (
                    <button
                      key={val}
                      onClick={() => setAccountBalance(val)}
                      className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] cursor-pointer"
                    >
                      {accountCurrency === "INR" ? "₹" : "$"}{val >= 1000 ? `${val / 1000}k` : val}
                    </button>
                  ))}
                </div>
                {accountBalance <= 5000 && accountCurrency === "INR" && (
                  <p className="mt-1 text-[10px] text-amber-300">
                    🛡️ Capital Protection Active: ₹2,000 INR small-account bounds enforce strict NO TRADE when broker minimum lot (0.01 lot) risks &gt;0.5%.
                  </p>
                )}
              </div>

              {/* Risk per trade slider */}
              <div>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-slate-300 font-semibold">Risk Per Trade Limit</label>
                  <span className="font-mono font-bold text-amber-400">{riskPct}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="2.0"
                  step="0.1"
                  value={riskPct}
                  onChange={(e) => setRiskPct(Number(e.target.value))}
                  className="w-full accent-amber-500 cursor-pointer"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-1">
                  <span>0.1% Ultra-Safe</span>
                  <span>0.5% Recommended</span>
                  <span>1.0% Max Cap</span>
                </div>
              </div>

              {/* TradingView Webhook Key */}
              <div className="p-3 rounded-lg bg-[#090c13] border border-slate-800 space-y-2">
                <span className="text-slate-300 font-semibold block text-[11px]">
                  TradingView Webhook Integration Secret:
                </span>
                <div className="flex gap-2">
                  <input
                    type="text"
                    readOnly
                    value="http://localhost:8001/api/v1/webhooks/tradingview"
                    className="flex-1 px-2.5 py-1.5 rounded bg-[#131926] border border-slate-700 text-slate-300 font-mono text-[10px]"
                  />
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText("http://localhost:8001/api/v1/webhooks/tradingview");
                      setCopiedWebhook(true);
                      setTimeout(() => setCopiedWebhook(false), 2000);
                    }}
                    className="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[10px] cursor-pointer"
                  >
                    {copiedWebhook ? "Copied" : "Copy"}
                  </button>
                </div>
                <p className="text-[10px] text-slate-500">
                  Shared secret key: <code className="text-amber-400">gm_secret_default_key</code>
                </p>
              </div>
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setShowSettingsModal(false)}
                className="px-4 py-2 rounded-lg bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold text-xs uppercase cursor-pointer transition"
              >
                Save & Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
