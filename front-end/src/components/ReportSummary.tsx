import { useMemo } from 'react';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import reportData from "../../../reports/ReportSummary.md?raw";

interface SummaryProps {
  onBack: () => void;
}

export const ReportSummary = ({ onBack }: SummaryProps) => {
  const categories = [
    "Executive Summary",
    "Macro Environment",
    "Shortlisted Tickers",
    "Regime Classification",
    "Strategy Parameters",
    "Diagnostic Results"
  ];

  const scrollToSection = (idText: string) => {
    const targetId = idText.replace(/\s+/g, '-').toLowerCase();
    const element = document.getElementById(targetId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const scrollToTicker = (symbol: string) => {
    const element = document.getElementById(`ticker-${symbol}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  // --- 1. DYNAMIC MULTI-TICKER EXTRACTION ---
  const { tickerDashboards, rawContent } = useMemo(() => {
    let content = typeof reportData === 'string' ? reportData 
      : (reportData && typeof reportData === 'object' && 'default' in reportData) ? (reportData as any).default 
      : String(reportData || "");

    const tickerMetrics: any[] = [];
    const chunks = content.split(/(?=\n### )/g);

    chunks.forEach(chunk => {
      if (chunk.includes('Walk-Forward Returns') || chunk.includes('Best 3 trades') || chunk.includes('SPY Buy-and-Hold')) {
        
        const lines = chunk.trim().split('\n');
        const rawTitle = lines[0].replace(/#/g, '').trim();

        if (rawTitle.toUpperCase().includes('PORTFOLIO') || rawTitle.toUpperCase().includes('SUMMARY')) {
          return; 
        }

        let spyLine = '';
        let tickerLine = '';
        let descLines: string[] = [];
        
        const topSectionMatch = chunk.match(/[\s\S]*?(?=\n(?:###|####|\*\*Best|\*\*Worst|Walk-Forward|Return Dist))/i);
        const topSectionRaw = topSectionMatch ? topSectionMatch[0] : chunk;
        
        const topLines = topSectionRaw.split('\n').slice(1);
        topLines.forEach(line => {
          const lower = line.toLowerCase();
          if (lower.includes('spy buy-and-hold')) {
            spyLine = line.replace(/\*\*/g, '').trim();
          } else if (lower.includes('alphacombined') && line.includes(':')) {
            tickerLine = line.replace(/\*\*/g, '').trim();
          } else if (line.trim().length > 0 && !line.startsWith('#')) {
            descLines.push(line.replace(/\*\*/g, '').trim());
          }
        });

        const extractBlock = (text: string, keyword: string) => {
          const regex = new RegExp(`(?:### |#### |\\*\\*|)*${keyword}[\\s\\S]*?(?=\\n(?:####|###|##|#)\\s|$)`, 'i');
          const match = text.match(regex);
          return match ? match[0].trim() : '';
        };

        const extractAndFormatCodeBlock = (text: string, keyword: string) => {
          const rawText = extractBlock(text, keyword);
          if (!rawText) return '';
          const blockLines = rawText.split('\n');
          if (blockLines.length <= 1) return rawText;
          const title = blockLines[0];
          const body = blockLines.slice(1).join('\n');
          return `${title}\n\n\`\`\`text\n${body}\n\`\`\``;
        };

        tickerMetrics.push({
          title: rawTitle,
          spyLine,
          tickerLine,
          description: descLines.join(' '), 
          bestTrades: extractBlock(chunk, 'Best 3 trades'),
          worstTrades: extractBlock(chunk, 'Worst 3 trades'),
          walkForward: extractAndFormatCodeBlock(chunk, 'Walk-Forward Returns'),
          returnDist: extractAndFormatCodeBlock(chunk, 'Return Distribution'),
        });
      }
    });

    return { 
      tickerDashboards: tickerMetrics,
      rawContent: content
    };
  }, []);
  // -----------------------------------

  // --- 2. PREMIUM TERMINAL STYLING ---
  const sharedMarkdownComponents: any = {
    h1: () => null,
    h2: ({node, children, ...props}: any) => {
      const sectionId = String(children).replace(/\s+/g, '-').toLowerCase();
      return (
        <h2 id={sectionId} className="font-jersey text-5xl text-[#e3bf17] mt-16 mb-6 uppercase border-b border-[#e3bf17]/10 pb-2 scroll-mt-32" {...props}>
          {children}
        </h2>
      );
    },
    h3: ({node, ...props}: any) => <h3 className="font-jersey text-4xl text-[#e3bf17]/80 mt-10 mb-4 uppercase" {...props} />,
    h4: ({node, ...props}: any) => <h4 className="font-jersey text-3xl text-[#e3bf17]/60 mt-8 mb-4 uppercase" {...props} />,
    p: ({node, ...props}: any) => <p className="font-jersey text-2xl text-[#e3bf17]/80 leading-relaxed mb-6 tracking-wide" {...props} />,
    strong: ({node, ...props}: any) => <strong className="font-jersey text-[#e3bf17] font-normal border-b border-[#e3bf17]/30" {...props} />,
    li: ({node, ...props}: any) => <li className="font-jersey text-xl text-[#e3bf17]/70 mb-4 list-none border-l-2 border-[#e3bf17]/20 pl-4 hover:border-[#e3bf17] transition-colors" {...props} />,
    
    pre: ({node, ...props}: any) => (
      <div className="relative mt-8 mb-12 group">
        <div className="absolute top-0 left-0 right-0 h-7 bg-[#e3bf17]/20 rounded-t-xl border-b border-[#e3bf17]/30 flex items-center px-4 gap-2 backdrop-blur-md z-10">
          <div className="w-2.5 h-2.5 rounded-full bg-red-500/80"></div>
          <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/80"></div>
          <div className="w-2.5 h-2.5 rounded-full bg-green-500/80"></div>
          <span className="ml-3 font-mono text-[9px] text-[#e3bf17]/50 tracking-widest uppercase">SYS_DATA_GRID</span>
        </div>
        <pre className="pt-10 pb-6 px-5 font-mono text-[11.5px] leading-relaxed text-[#e3bf17]/90 overflow-x-auto custom-scrollbar bg-gradient-to-b from-black/90 to-[#e3bf17]/5 border border-[#e3bf17]/30 rounded-xl shadow-[0_10px_40px_rgba(227,191,23,0.08)] group-hover:border-[#e3bf17]/60 transition-colors" {...props} />
      </div>
    ),

    code: ({node, inline, children, ...props}: any) => {
      if (inline) return <code className="font-mono text-[#e3bf17] bg-[#e3bf17]/10 px-1 py-0.5 rounded" {...props}>{children}</code>;
      
      const content = String(children || '').replace(/\n$/, '');
      const lines = content.split('\n');
      let headerFound = false;

      return (
        <code className="font-mono text-[11.5px] leading-[1.6]" {...props}>
          {lines.map((line: string, i: number) => {
            if (line.trim() === '') return <div key={i} className="h-4"></div>;
            
            const isHeader = !headerFound;
            if (isHeader) headerFound = true;

            return (
              <div key={i} className={isHeader 
                ? "font-bold text-[#e3bf17] border-b border-[#e3bf17]/40 pb-2 mb-3 tracking-wide uppercase whitespace-pre" 
                : "text-[#e3bf17]/80 hover:text-white transition-colors whitespace-pre"
              }>
                {line}
              </div>
            );
          })}
        </code>
      );
    },

    table: ({node, ...props}: any) => (
      <div className="overflow-x-auto overflow-y-auto max-h-[400px] border border-[#e3bf17]/20 rounded-lg mb-8 custom-scrollbar bg-black/50">
        <table className="w-full text-left border-collapse" {...props} />
      </div>
    ),
    th: ({node, ...props}: any) => <th className="sticky top-0 bg-black border-b-2 border-[#e3bf17]/30 p-4 font-jersey text-2xl text-[#e3bf17]" {...props} />,
    td: ({node, ...props}: any) => <td className="border-b border-[#e3bf17]/10 p-4 font-mono text-[#e3bf17]/80" {...props} />,
    blockquote: ({node, ...props}: any) => <blockquote className="border-l-4 border-[#e3bf17] bg-[#e3bf17]/5 p-8 my-10 font-jersey text-2xl text-[#e3bf17]/90 rounded-r-xl tracking-tight" {...props} />,
  };

  return (
    <motion.div 
      onClick={(e) => e.stopPropagation()} 
      initial={{ x: '100vw' }}
      animate={{ x: 0 }}
      exit={{ x: '100vw' }}
      transition={{ type: "spring", damping: 25, stiffness: 120 }}
      className="absolute inset-0 bg-black/95 backdrop-blur-3xl z-[100] overflow-y-auto"
    >
      <div className="fixed top-0 left-0 right-0 p-6 md:p-8 flex justify-between items-center z-[110] pointer-events-none bg-gradient-to-b from-black/95 via-black/80 to-transparent">
        <button 
          onClick={onBack}
          className="pointer-events-auto font-jersey text-3xl text-[#e3bf17] hover:tracking-widest transition-all uppercase flex items-center gap-4"
        >
          {'<<<'} RETURN_TO_HUB
        </button>
        <div className="font-jersey text-sm text-[#e3bf17]/40 tracking-[0.4em] uppercase text-right">
          QUANT_ENGINE: ACTIVE // REPORT_ID: RS_V1.0 // STATUS: VERIFIED
        </div>
      </div>

      <div className="flex w-full max-w-[1600px] mx-auto px-4 md:px-8 pt-24 pb-32 gap-8 relative items-start">
        
        {/* LEFT COLUMN */}
        <div className="hidden xl:flex w-44 flex-col gap-1 sticky top-24 self-start max-h-[85vh] overflow-y-auto custom-scrollbar shrink-0">
          <div className="border-b border-[#e3bf17]/30 pb-1 mb-1">
            <h3 className="font-jersey text-base text-[#e3bf17]/60 tracking-widest uppercase">
              TARGET_INDEX
            </h3>
          </div>
          
          {tickerDashboards.map((ticker, index) => {
            const symbol = ticker.title.split(' ')[0]; 
            return (
              <button 
                key={index}
                onClick={() => scrollToTicker(symbol)}
                className="group text-left border border-[#e3bf17]/20 bg-black/60 backdrop-blur-md py-1.5 px-3 hover:bg-[#e3bf17]/10 hover:border-[#e3bf17]/60 transition-all flex items-center justify-between overflow-hidden shrink-0"
              >
                <span className="font-jersey text-lg text-[#e3bf17] uppercase group-hover:tracking-wider transition-all whitespace-nowrap">
                  {symbol}
                </span>
                <span className="font-mono text-[9px] text-[#e3bf17]/40 opacity-0 group-hover:opacity-100 transition-opacity ml-2 shrink-0">
                  LOCATE
                </span>
              </button>
            );
          })}
        </div>

        {/* MIDDLE COLUMN: Main Content */}
        <div className="flex-1 space-y-8 min-w-0">
          <div className="border-l-4 border-[#e3bf17] bg-[#e3bf17]/5 p-8 backdrop-blur-md">
            <h1 className="font-jersey text-8xl text-[#e3bf17] uppercase m-0 leading-none">
              REPORT_SUMMARY
            </h1>
            <p className="font-jersey text-2xl text-[#e3bf17]/60 mt-4 tracking-[0.1em] uppercase">
              SYSTEM_ANALYSIS_LOG // ARCHIVE_DATA_LOADED
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <div className="border border-[#e3bf17]/20 p-5 bg-black/40">
              <p className="font-jersey text-sm text-[#e3bf17]/40 uppercase mb-2">Algorithm_Win_Rate</p>
              <p className="font-jersey text-4xl text-[#e3bf17]">0.0%</p>
            </div>
            <div className="border border-[#e3bf17]/20 p-5 bg-black/40">
              <p className="font-jersey text-sm text-[#e3bf17]/40 uppercase mb-2">Profit_Factor</p>
              <p className="font-jersey text-4xl text-zinc-600">N/A</p>
            </div>
            <div className="border border-[#e3bf17]/20 p-5 bg-black/40">
              <p className="font-jersey text-sm text-[#e3bf17]/40 uppercase mb-2">System_State</p>
              <p className="font-jersey text-4xl text-[#e3bf17] animate-pulse uppercase">Idle</p>
            </div>
          </div>

          {/* --- DYNAMIC TICKER DASHBOARDS --- */}
          {tickerDashboards.map((ticker, index) => {
            const symbol = ticker.title.split(' ')[0];
            return (
              <div 
                id={`ticker-${symbol}`} 
                key={index} 
                className="border border-[#e3bf17]/20 p-8 bg-black/40 rounded-3xl mb-16 relative overflow-hidden scroll-mt-32"
              >
                <div className="absolute top-0 right-0 w-64 h-64 bg-[#e3bf17]/5 blur-[100px] rounded-full pointer-events-none"></div>

                <h2 className="font-jersey text-4xl text-[#e3bf17] uppercase mb-8 border-b border-[#e3bf17]/10 pb-4 relative z-10 flex items-center gap-4">
                  <span className="bg-[#e3bf17] text-black px-4 py-1 rounded-sm text-3xl">TARGET</span>
                  {ticker.title}
                </h2>

                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 relative z-10">
                  
                  {/* Left Side: Stats, Description, & Trades */}
                  <div className="space-y-6">
                    <div className="space-y-4">
                      {ticker.spyLine && (
                        <div className="flex justify-between items-center bg-black/80 border border-[#e3bf17]/30 p-4 rounded-lg shadow-lg">
                          <span className="font-jersey text-xl text-[#e3bf17]/60 tracking-wider uppercase">{ticker.spyLine.split(':')[0]}</span>
                          <span className="font-mono text-2xl text-[#e3bf17] font-bold">{ticker.spyLine.split(':')[1]}</span>
                        </div>
                      )}
                      {ticker.tickerLine && (
                        <div className="flex justify-between items-center bg-[#e3bf17]/10 border border-[#e3bf17]/50 p-4 rounded-lg shadow-lg">
                          <span className="font-jersey text-xl text-[#e3bf17]/80 tracking-wider uppercase">{ticker.tickerLine.split(':')[0]}</span>
                          <span className="font-mono text-2xl text-[#e3bf17] font-bold">{ticker.tickerLine.split(':')[1]}</span>
                        </div>
                      )}
                      {ticker.description && (
                        <div className="bg-black/40 border-l-4 border-[#e3bf17]/40 p-5 rounded-r-lg mt-4 shadow-inner">
                          <p className="font-mono text-sm text-[#e3bf17]/80 leading-relaxed m-0">{ticker.description}</p>
                        </div>
                      )}
                    </div>

                    {ticker.bestTrades && (
                      <div className="prose prose-invert prose-yellow max-w-none bg-black/50 p-5 border-l-4 border-green-500 border-y border-r border-[#e3bf17]/10 rounded-r-xl shadow-lg">
                        <ReactMarkdown components={sharedMarkdownComponents}>{ticker.bestTrades}</ReactMarkdown>
                      </div>
                    )}

                    {ticker.worstTrades && (
                      <div className="prose prose-invert prose-yellow max-w-none bg-black/50 p-5 border-l-4 border-red-500 border-y border-r border-[#e3bf17]/10 rounded-r-xl shadow-lg">
                        <ReactMarkdown components={sharedMarkdownComponents}>{ticker.worstTrades}</ReactMarkdown>
                      </div>
                    )}
                  </div>

                  {/* Right Side: Designed Terminal Data Grids */}
                  <div className="space-y-0">
                    {ticker.walkForward && (
                      <div className="prose prose-invert prose-yellow max-w-none">
                        <ReactMarkdown components={sharedMarkdownComponents}>{ticker.walkForward}</ReactMarkdown>
                      </div>
                    )}
                    {ticker.returnDist && (
                      <div className="prose prose-invert prose-yellow max-w-none">
                        <ReactMarkdown components={sharedMarkdownComponents}>{ticker.returnDist}</ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })}
          {/* ------------------------------------ */}

          {/* FULL MARKDOWN TEXT RENDERING */}
          <div className="border border-[#e3bf17]/20 p-12 bg-black/40 rounded-br-3xl">
            <div className="prose prose-invert prose-yellow max-w-none">
              <ReactMarkdown components={sharedMarkdownComponents}>
                {rawContent}
              </ReactMarkdown>
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN */}
        <div className="hidden lg:flex w-44 flex-col gap-1 sticky top-24 self-start max-h-[85vh] overflow-y-auto custom-scrollbar shrink-0">
          <div className="border-b border-[#e3bf17]/30 pb-1 mb-1">
            <h3 className="font-jersey text-base text-[#e3bf17]/60 tracking-widest uppercase">
              DATA_INDEX
            </h3>
          </div>
          
          {categories.map((category, index) => (
            <button 
              key={index}
              onClick={() => scrollToSection(category)}
              className="group text-left border border-[#e3bf17]/20 bg-black/60 backdrop-blur-md py-1.5 px-3 hover:bg-[#e3bf17]/10 hover:border-[#e3bf17]/60 transition-all flex items-center justify-between overflow-hidden shrink-0"
            >
              <span className="font-jersey text-[16px] text-[#e3bf17] uppercase group-hover:tracking-wider transition-all whitespace-nowrap">
                {category}
              </span>
              <span className="font-mono text-[9px] text-[#e3bf17]/40 opacity-0 group-hover:opacity-100 transition-opacity ml-2 shrink-0">
                JUMP
              </span>
            </button>
          ))}
        </div>

      </div>
    </motion.div>
  );
};