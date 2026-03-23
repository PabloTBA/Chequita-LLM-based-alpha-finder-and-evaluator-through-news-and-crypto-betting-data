import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

const API = 'http://localhost:8000';

type Status = 'idle' | 'running' | 'done' | 'error';

export const Generate = () => {
  const [status, setStatus]       = useState<Status>('idle');
  const [logs, setLogs]           = useState<string[]>([]);
  const [jobId, setJobId]         = useState<string | null>(null);
  const [reportPath, setReportPath] = useState<string | null>(null);
  const [dots, setDots]           = useState('');
  const logRef                    = useRef<HTMLDivElement>(null);

  // Animate dots while running
  useEffect(() => {
    if (status !== 'running') { setDots(''); return; }
    const t = setInterval(() => setDots((p: string) => p.length >= 3 ? '' : p + '.'), 500);
    return () => clearInterval(t);
  }, [status]);

  // Poll for status while running
  useEffect(() => {
    if (!jobId || status !== 'running') return;
    const t = setInterval(async () => {
      try {
        const res  = await fetch(`${API}/api/status/${jobId}`);
        const data = await res.json();
        setLogs(data.logs ?? []);
        if (data.status === 'done') {
          setStatus('done');
          setReportPath(data.summary_path ?? data.report_path ?? null);
        } else if (data.status === 'error') {
          setStatus('error');
        }
      } catch {
        // server not reachable yet — keep polling
      }
    }, 2000);
    return () => clearInterval(t);
  }, [jobId, status]);

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logs]);

  const handleGenerate = async () => {
    setStatus('running');
    setLogs([]);
    setJobId(null);
    setReportPath(null);
    try {
      const res  = await fetch(`${API}/api/run`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ days: 7, max_tickers: 5, min_volume: 10000, max_markets: 30 }),
      });
      const data = await res.json();
      setJobId(data.job_id);
    } catch {
      setLogs(['ERROR: Could not connect to API server.', 'Make sure api_server.py is running: python api_server.py']);
      setStatus('error');
    }
  };

  const handleReset = () => {
    setStatus('idle');
    setLogs([]);
    setJobId(null);
    setReportPath(null);
  };

  return (
    <motion.div
      onClick={(e: React.MouseEvent) => e.stopPropagation()}
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-20 rounded-2xl flex flex-col items-center justify-center"
    >
      {status === 'idle' && (
        <>
          <h2 className="font-jersey text-7xl text-[#e3bf17] mb-8 uppercase animate-neon-buzz text-center">
            System_Ready
          </h2>
          <button
            onClick={handleGenerate}
            className="group relative px-12 py-4 border-2 border-[#e3bf17] overflow-hidden transition-all hover:shadow-[0_0_20px_rgba(227,191,23,0.6)]"
          >
            <span className="relative z-10 font-jersey text-4xl text-[#e3bf17] group-hover:text-black transition-colors duration-300 uppercase">
              Execute_Generation
            </span>
            <div className="absolute inset-0 bg-[#e3bf17] translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
          </button>
        </>
      )}

      {status === 'running' && (
        <div className="flex flex-col items-center w-full">
          <h2 className="font-jersey text-7xl text-[#e3bf17] mb-4 uppercase w-[600px] text-center">
            Generating{dots}
          </h2>
          <p className="font-mono text-xs text-[#e3bf17]/40 tracking-[0.5em] uppercase animate-pulse mt-2 mb-6">
            Processing_Neural_Weights // Flux_Stable
          </p>
          {logs.length > 0 && (
            <div
              ref={logRef}
              className="w-full max-h-64 overflow-y-auto bg-black/80 border border-[#e3bf17]/20 rounded p-4 font-mono text-xs text-[#e3bf17]/60 space-y-0.5"
            >
              {logs.slice(-60).map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all leading-5">{line}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {status === 'done' && (
        <div className="flex flex-col items-center gap-6 w-full">
          <h2 className="font-jersey text-7xl text-[#e3bf17] uppercase text-center">
            Complete
          </h2>
          {reportPath && (
            <p className="font-mono text-sm text-[#e3bf17]/70 text-center">
              Report saved → <span className="text-[#e3bf17]">{reportPath}</span>
            </p>
          )}
          <button
            onClick={handleReset}
            className="group relative px-10 py-3 border-2 border-[#e3bf17] overflow-hidden transition-all hover:shadow-[0_0_20px_rgba(227,191,23,0.6)]"
          >
            <span className="relative z-10 font-jersey text-3xl text-[#e3bf17] group-hover:text-black transition-colors duration-300 uppercase">
              Run_Again
            </span>
            <div className="absolute inset-0 bg-[#e3bf17] translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
          </button>
        </div>
      )}

      {status === 'error' && (
        <div className="flex flex-col items-center gap-6 w-full">
          <h2 className="font-jersey text-7xl text-red-500 uppercase text-center">
            Error
          </h2>
          {logs.length > 0 && (
            <div className="w-full max-h-48 overflow-y-auto bg-black/80 border border-red-500/30 rounded p-4 font-mono text-xs text-red-400/80 space-y-0.5">
              {logs.slice(-20).map((line, i) => (
                <div key={i} className="whitespace-pre-wrap break-all leading-5">{line}</div>
              ))}
            </div>
          )}
          <button
            onClick={handleReset}
            className="group relative px-10 py-3 border-2 border-red-500 overflow-hidden transition-all hover:shadow-[0_0_20px_rgba(239,68,68,0.6)]"
          >
            <span className="relative z-10 font-jersey text-3xl text-red-500 group-hover:text-white transition-colors duration-300 uppercase">
              Retry
            </span>
            <div className="absolute inset-0 bg-red-500 translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
          </button>
        </div>
      )}
    </motion.div>
  );
};
