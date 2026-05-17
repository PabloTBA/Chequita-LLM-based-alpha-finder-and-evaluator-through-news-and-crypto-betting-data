import { useState, useEffect, useRef, useMemo } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { createChart, ColorType, AreaSeries, HistogramSeries } from 'lightweight-charts';
import traderData from "../../../reports/TraderSummary.md?raw";

interface SummaryProps {
  onBack: () => void;
}

export const TraderSummary = ({ onBack }: SummaryProps) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const [renderError, setRenderError] = useState<string | null>(null);
  const [activeChart, setActiveChart] = useState<'EQUITY' | 'DRAWDOWN' | 'TRADES'>('EQUITY');

  // --- 1. DYNAMIC ISOLATED TABLE PARSING ---
  const { 
    equityData, 
    drawdownData, 
    tradeData,
    tickersQualified, 
    marketBias, 
    actionState, 
    actionColor, 
    biasColor,
    rawContent 
  } = useMemo(() => {
    let tq = "0", mb = "NEUTRAL", as = "STANDBY";
    let ac = "text-red-900/50 italic", bc = "text-[#e3bf17] animate-pulse";
    
    let content = typeof traderData === 'string' ? traderData 
      : (traderData && typeof traderData === 'object' && 'default' in traderData) ? (traderData as any).default 
      : String(traderData || "");

    try {
      if (content.length > 0) {
        const tickerMatch = content.match(/\*\*(\d+)\s+ticker\(s\)/i);
        if (tickerMatch && tickerMatch[1]) tq = tickerMatch[1];
        const biasMatch = content.match(/\*\*Market bias:\*\*\s*([A-Z]+)/i);
        if (biasMatch && biasMatch[1]) mb = biasMatch[1];
        if (parseInt(tq) > 0) {
          as = "ACTIVE";
          ac = "text-green-500 animate-pulse font-bold";
        }
      }
    } catch (err) { console.error(err); }

    const extractTableData = (type: 'EQUITY' | 'DRAWDOWN' | 'TRADES') => {
      const tables: string[][] = [];
      let currentTable: string[] = [];
      
      // Isolate all tables into separate arrays
      for (let line of content.split('\n')) {
        if (line.includes('|')) currentTable.push(line);
        else if (currentTable.length > 0) {
          tables.push(currentTable);
          currentTable = [];
        }
      }
      if (currentTable.length > 0) tables.push(currentTable);

      // Analyze each table locally
      for (let tableLines of tables) {
        if (tableLines.length < 3) continue;

        const headers = tableLines[0].split('|').map(h => h.trim().toLowerCase());
        
        let dateIdx = headers.findIndex(h => h.includes('date') || h.includes('time'));
        if (dateIdx === -1) dateIdx = 1; // Default markdown table 1st col

        let valueIdx = -1;
        if (type === 'EQUITY') {
          valueIdx = headers.findIndex(h => h.includes('equity') || h.includes('value') || h.includes('portfolio') || h.includes('balance') || h.includes('cumulative') || h.includes('curve'));
        } else if (type === 'DRAWDOWN') {
          valueIdx = headers.findIndex(h => h.includes('drawdown') || h.includes('dd') || h.includes('underwater'));
        } else if (type === 'TRADES') {
          valueIdx = headers.findIndex(h => h.includes('p&l') || h.includes('profit') || h.includes('net') || h.includes('return') || h.includes('trade'));
        }

        // If target header found, parse ONLY this table
        if (valueIdx !== -1) {
          const data = new Map();
          for (let i = 2; i < tableLines.length; i++) {
             const row = tableLines[i];
             if (row.includes('---')) continue;
             
             const cells = row.split('|').map(c => c.trim());
             if (cells.length > valueIdx && cells.length > dateIdx) {
                const dateMatch = cells[dateIdx].match(/\d{4}-\d{2}-\d{2}/);
                const time = dateMatch ? dateMatch[0] : null;

                const cleanVal = cells[valueIdx].replace(/[$%,]/g, '');
                const numMatch = cleanVal.match(/[+-]?\d*\.?\d+/);
                const value = numMatch ? parseFloat(numMatch[0]) : NaN;

                if (time && !isNaN(value)) data.set(time, { time, value });
             }
          }
          if (data.size > 0) {
            return Array.from(data.values()).sort((a: any, b: any) => new Date(a.time).getTime() - new Date(b.time).getTime());
          }
        }
      }

      // FALLBACK: If explicit headers are missing entirely
      let keyword = type === 'EQUITY' ? 'Equity' : type === 'DRAWDOWN' ? 'Drawdown' : 'Trade';
      const regex = new RegExp(`(?:### |#### |\\*\\*)*${keyword}[\\s\\S]*?(?=\\n(?:####|###|##|#)\\s|$)`, 'i');
      let match = content.match(regex);
      if (match) {
         const fallbackData = new Map();
         for (let r of match[0].split('\n')) {
           if (r.includes('|') && !r.includes('---') && !/[a-zA-Z]/.test(r)) { // Skips header strings
              const cells = r.split('|').map(c => c.trim()).filter(c => c.length > 0);
              if (cells.length < 2) continue;

              const dateMatch = cells[0].match(/\d{4}-\d{2}-\d{2}/);
              const time = dateMatch ? dateMatch[0] : null;

              const cleanVal = cells[cells.length - 1].replace(/[$%,]/g, '');
              const numMatch = cleanVal.match(/[+-]?\d*\.?\d+/);
              const value = numMatch ? parseFloat(numMatch[0]) : NaN;

              if (time && !isNaN(value)) fallbackData.set(time, { time, value });
           }
         }
         if (fallbackData.size > 0) {
           return Array.from(fallbackData.values()).sort((a: any, b: any) => new Date(a.time).getTime() - new Date(b.time).getTime());
         }
      }
      return [];
    };

    return { 
      equityData: extractTableData('EQUITY'),
      drawdownData: extractTableData('DRAWDOWN'),
      tradeData: extractTableData('TRADES'),
      tickersQualified: tq, marketBias: mb, actionState: as, actionColor: ac, biasColor: bc, rawContent: content
    };
  }, []);
  // -------------------------------

  // --- 2. DYNAMIC CHART SWAPPER LOGIC ---
  useEffect(() => {
    if (!chartContainerRef.current) return;

    let activeData: { time: string, value: number, color?: string }[] = [];
    if (activeChart === 'EQUITY') activeData = equityData;
    if (activeChart === 'DRAWDOWN') activeData = drawdownData;
    if (activeChart === 'TRADES') {
      activeData = tradeData.map(d => ({
        time: d.time,
        value: d.value,
        color: d.value >= 0 ? '#22c55e' : '#ef4444' 
      }));
    }

    // Prevents ghost charts by wiping the container if no data is found
    chartContainerRef.current.innerHTML = '';
    if (activeData.length === 0) return;

    try {
      const chart = createChart(chartContainerRef.current, {
        autoSize: true, 
        height: 350,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: 'rgba(227, 191, 23, 0.6)',
          fontFamily: 'monospace',
        },
        grid: {
          vertLines: { color: 'rgba(227, 191, 23, 0.05)' },
          horzLines: { color: 'rgba(227, 191, 23, 0.05)' },
        },
      });

      if (activeChart === 'EQUITY') {
        const series = chart.addSeries(AreaSeries, {
          lineColor: '#e3bf17',
          topColor: 'rgba(227, 191, 23, 0.4)',
          bottomColor: 'rgba(227, 191, 23, 0.0)',
          lineWidth: 2,
        });
        series.setData(activeData);
      } 
      else if (activeChart === 'DRAWDOWN') {
        const series = chart.addSeries(AreaSeries, {
          lineColor: '#ef4444',
          topColor: 'rgba(239, 68, 68, 0.0)',
          bottomColor: 'rgba(239, 68, 68, 0.4)',
          lineWidth: 2,
        });
        series.setData(activeData);
      } 
      else if (activeChart === 'TRADES') {
        const series = chart.addSeries(HistogramSeries, {});
        series.setData(activeData);
      }

      chart.timeScale().fitContent();

      return () => { chart.remove(); };
    } catch (error) { setRenderError(String(error)); }
  }, [activeChart, equityData, drawdownData, tradeData]);
  // -------------------------------

  if (renderError) {
    return (
      <div className="absolute inset-0 bg-red-950 flex flex-col items-center justify-center z-[200]">
        <h1 className="text-white text-4xl mb-4 font-jersey">CHART RENDER FAILED</h1>
        <p className="text-red-300 font-mono bg-black p-4 rounded">{renderError}</p>
        <button onClick={onBack} className="mt-8 text-white border px-6 py-2">RETURN</button>
      </div>
    );
  }

  return (
    <motion.div 
      onClick={(e) => e.stopPropagation()} 
      initial={{ x: '100vw' }}
      animate={{ x: 0 }}
      exit={{ x: '100vw' }}
      transition={{ type: "spring", damping: 25, stiffness: 120 }}
      className="absolute inset-0 flex flex-col items-center bg-black/95 backdrop-blur-3xl z-[100] overflow-y-auto p-6 md:p-20"
    >
      <div className="fixed top-0 left-0 right-0 p-8 flex justify-between items-center z-[110] pointer-events-none">
        <button 
          onClick={onBack}
          className="pointer-events-auto font-jersey text-3xl text-[#e3bf17] hover:tracking-widest transition-all uppercase flex items-center gap-4"
        >
          {'<<<'} RETURN_TO_HUB
        </button>
      </div>

      <div className="max-w-5xl w-full mt-24 mb-20 space-y-8">
        <div className="border-l-4 border-[#e3bf17] bg-[#e3bf17]/5 p-8 backdrop-blur-md">
          <h1 className="font-jersey text-8xl text-[#e3bf17] uppercase m-0 leading-none">
            TRADER_DOSSIER
          </h1>
        </div>

        {/* METRICS GRID */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="border border-[#e3bf17]/20 p-6 bg-black/40">
            <p className="font-jersey text-sm text-[#e3bf17]/40 uppercase mb-2">Market_Bias</p>
            <p className={`font-jersey text-5xl ${biasColor}`}>{marketBias}</p>
          </div>
          <div className="border border-[#e3bf17]/20 p-6 bg-black/40">
            <p className="font-jersey text-sm text-[#e3bf17]/40 uppercase mb-2">Tickers_Qualified</p>
            <p className={`font-jersey text-5xl ${parseInt(tickersQualified) > 0 ? 'text-[#e3bf17]' : 'text-zinc-600'}`}>
              {tickersQualified}
            </p>
          </div>
          <div className="border border-[#e3bf17]/20 p-6 bg-black/40">
            <p className="font-jersey text-sm text-[#e3bf17]/40 uppercase mb-2">Action_State</p>
            <p className={`font-jersey text-5xl ${actionColor}`}>
              {actionState}
            </p>
          </div>
        </div>

        {/* --- THE TACTICAL MULTI-CHART --- */}
        <div className="border border-[#e3bf17]/20 p-6 bg-black/40">
          
          <div className="flex gap-8 border-b border-[#e3bf17]/10 pb-4 mb-6">
            <button 
              onClick={() => setActiveChart('EQUITY')} 
              className={`font-jersey text-3xl uppercase transition-all tracking-wide ${activeChart === 'EQUITY' ? 'text-[#e3bf17] drop-shadow-[0_0_10px_rgba(227,191,23,0.8)]' : 'text-[#e3bf17]/30 hover:text-[#e3bf17]/60'}`}
            >
              EQUITY_CURVE
            </button>
            <button 
              onClick={() => setActiveChart('DRAWDOWN')} 
              className={`font-jersey text-3xl uppercase transition-all tracking-wide ${activeChart === 'DRAWDOWN' ? 'text-red-500 drop-shadow-[0_0_10px_rgba(239,68,68,0.8)]' : 'text-[#e3bf17]/30 hover:text-[#e3bf17]/60'}`}
            >
              DRAWDOWN
            </button>
            <button 
              onClick={() => setActiveChart('TRADES')} 
              className={`font-jersey text-3xl uppercase transition-all tracking-wide ${activeChart === 'TRADES' ? 'text-green-500 drop-shadow-[0_0_10px_rgba(34,197,94,0.8)]' : 'text-[#e3bf17]/30 hover:text-[#e3bf17]/60'}`}
            >
              TRADE_P&L
            </button>
          </div>
          
          <div className="relative w-full h-[350px]">
            {((activeChart === 'EQUITY' && equityData.length > 0) || 
              (activeChart === 'DRAWDOWN' && drawdownData.length > 0) || 
              (activeChart === 'TRADES' && tradeData.length > 0)) ? (
              <div ref={chartContainerRef} className="absolute inset-0" />
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-[#e3bf17]/40 font-mono text-sm border border-dashed border-[#e3bf17]/20">
                AWAITING_DATA_FOR_{activeChart}
              </div>
            )}
          </div>
        </div>

        {/* MARKDOWN TEXT */}
        <div className="border border-[#e3bf17]/20 p-12 bg-black/40 rounded-br-3xl">
          <div className="prose prose-invert prose-yellow max-w-none">
            <ReactMarkdown
              components={{
                h1: () => null,
                h2: ({node, ...props}) => <h2 className="font-jersey text-5xl text-[#e3bf17] mt-12 mb-6 uppercase border-b border-[#e3bf17]/10 pb-2" {...props} />,
                h3: ({node, ...props}) => <h3 className="font-jersey text-4xl text-[#e3bf17]/80 mt-10 mb-4 uppercase" {...props} />,
                h4: ({node, ...props}) => <h4 className="font-jersey text-3xl text-[#e3bf17]/60 mt-8 mb-4 uppercase" {...props} />,
                p: ({node, ...props}) => <p className="font-jersey text-2xl text-[#e3bf17]/80 leading-relaxed mb-6 tracking-wide" {...props} />,
                strong: ({node, ...props}) => <strong className="font-jersey text-[#e3bf17] font-normal border-b border-[#e3bf17]/30" {...props} />,
                li: ({node, ...props}) => <li className="font-jersey text-xl text-[#e3bf17]/70 mb-4 list-none border-l-2 border-[#e3bf17]/20 pl-4 hover:border-[#e3bf17] transition-colors" {...props} />,
                table: ({node, ...props}) => (
                  <div className="overflow-x-auto overflow-y-auto max-h-[400px] border border-[#e3bf17]/20 rounded-lg mb-8 custom-scrollbar bg-black/50">
                    <table className="w-full text-left border-collapse" {...props} />
                  </div>
                ),
                th: ({node, ...props}) => <th className="sticky top-0 bg-black border-b-2 border-[#e3bf17]/30 p-4 font-jersey text-2xl text-[#e3bf17]" {...props} />,
                td: ({node, ...props}) => <td className="border-b border-[#e3bf17]/10 p-4 font-mono text-[#e3bf17]/80" {...props} />,
              }}
            >
              {rawContent}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </motion.div>
  );
};