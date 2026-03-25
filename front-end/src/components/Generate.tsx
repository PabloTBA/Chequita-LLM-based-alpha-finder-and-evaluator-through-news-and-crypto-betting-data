import { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

type RunStatus = 'idle' | 'starting' | 'running' | 'done' | 'error';

export const Generate = () => {
  const [status, setStatus]   = useState<RunStatus>('idle');
  const [logs, setLogs]       = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string>('');
  const [dots, setDots]       = useState('');
  const logEndRef             = useRef<HTMLDivElement>(null);
  const esRef                 = useRef<EventSource | null>(null);

  // Animate "..." when running
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (status === 'starting' || status === 'running') {
      interval = setInterval(() => {
        setDots(prev => (prev.length >= 3 ? '' : prev + '.'));
      }, 500);
    } else {
      setDots('');
    }
    return () => clearInterval(interval);
  }, [status]);

  // Auto-scroll log pane
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Cleanup SSE on unmount
  useEffect(() => {
    return () => esRef.current?.close();
  }, []);

  const startGeneration = async () => {
    setStatus('starting');
    setLogs([]);
    setErrorMsg('');

    // 1. POST /api/run — triggers the pipeline
    try {
      const res = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days: 14, max_tickers: 15 }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detail.detail ?? res.statusText);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setStatus('error');
      return;
    }

    // 2. Open SSE stream — receives every print() line from the pipeline
    setStatus('running');
    const es = new EventSource('/api/logs');
    esRef.current = es;

    es.onmessage = (event) => {
      const line: string = event.data;
      if (line === '[DONE]') {
        es.close();
        setStatus('done');
        return;
      }
      setLogs(prev => [...prev, line]);
    };

    es.onerror = () => {
      es.close();
      // Only mark as error if still running (server may close stream normally)
      setStatus(prev => (prev === 'running' ? 'error' : prev));
    };
  };

  // ── Idle state ──────────────────────────────────────────────────────────────
  if (status === 'idle') {
    return (
      <motion.div
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-20 rounded-2xl flex flex-col items-center justify-center"
      >
        <h2 className="font-jersey text-7xl text-[#e3bf17] mb-8 uppercase animate-neon-buzz text-center">
          System_Ready
        </h2>
        <button
          onClick={startGeneration}
          className="group relative px-12 py-4 border-2 border-[#e3bf17] overflow-hidden transition-all hover:shadow-[0_0_20px_rgba(227,191,23,0.6)]"
        >
          <span className="relative z-10 font-jersey text-4xl text-[#e3bf17] group-hover:text-black transition-colors duration-300 uppercase">
            Execute_Generation
          </span>
          <div className="absolute inset-0 bg-[#e3bf17] translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
        </button>
        <p className="font-mono text-[10px] text-[#e3bf17]/30 tracking-[0.4em] uppercase mt-6">
          days=14 // max_tickers=15
        </p>
      </motion.div>
    );
  }

  // ── Running / starting state ─────────────────────────────────────────────────
  if (status === 'starting' || status === 'running') {
    return (
      <motion.div
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-10 rounded-2xl flex flex-col items-center"
      >
        <h2 className="font-jersey text-5xl text-[#e3bf17] mb-6 uppercase text-center">
          Generating{dots}
        </h2>
        <p className="font-mono text-[10px] text-[#e3bf17]/40 tracking-[0.5em] uppercase animate-pulse mb-6">
          Pipeline_Active // Streaming_Logs
        </p>

        {/* Log pane */}
        <div className="w-full bg-black/80 border border-[#e3bf17]/20 rounded-xl p-4 h-96 overflow-y-auto font-mono text-xs text-[#e3bf17]/70 leading-relaxed">
          {logs.length === 0 ? (
            <span className="text-[#e3bf17]/30 animate-pulse">Waiting for pipeline output...</span>
          ) : (
            logs.map((line, i) => (
              <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      </motion.div>
    );
  }

  // ── Done state ───────────────────────────────────────────────────────────────
  if (status === 'done') {
    return (
      <motion.div
        onClick={(e) => e.stopPropagation()}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, ease: 'easeOut' }}
        className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-[#e3bf17]/30 p-10 rounded-2xl flex flex-col items-center"
      >
        <h2 className="font-jersey text-5xl text-[#e3bf17] mb-2 uppercase text-center">
          Generation_Complete
        </h2>
        <p className="font-mono text-[10px] text-[#e3bf17]/40 tracking-[0.5em] uppercase mb-6">
          View results in Analysis_Center
        </p>

        {/* Log pane — scrollable recap */}
        <div className="w-full bg-black/80 border border-[#e3bf17]/20 rounded-xl p-4 h-72 overflow-y-auto font-mono text-xs text-[#e3bf17]/60 leading-relaxed mb-6">
          {logs.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
          ))}
          <div ref={logEndRef} />
        </div>

        <button
          onClick={() => { setStatus('idle'); setLogs([]); }}
          className="group relative px-10 py-3 border border-[#e3bf17]/50 overflow-hidden transition-all hover:shadow-[0_0_15px_rgba(227,191,23,0.4)]"
        >
          <span className="relative z-10 font-jersey text-2xl text-[#e3bf17] group-hover:text-black transition-colors duration-300 uppercase">
            Run_Again
          </span>
          <div className="absolute inset-0 bg-[#e3bf17] translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
        </button>
      </motion.div>
    );
  }

  // ── Error state ──────────────────────────────────────────────────────────────
  return (
    <motion.div
      onClick={(e) => e.stopPropagation()}
      initial={{ opacity: 0, y: 50 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: 'easeOut' }}
      className="max-w-5xl w-full bg-black/60 backdrop-blur-xl border border-red-500/40 p-10 rounded-2xl flex flex-col items-center"
    >
      <h2 className="font-jersey text-5xl text-red-400 mb-4 uppercase text-center">
        Pipeline_Error
      </h2>
      {errorMsg && (
        <p className="font-mono text-xs text-red-400/80 mb-6 text-center max-w-xl break-all">
          {errorMsg}
        </p>
      )}

      {/* Show any log lines captured before the error */}
      {logs.length > 0 && (
        <div className="w-full bg-black/80 border border-red-500/20 rounded-xl p-4 h-64 overflow-y-auto font-mono text-xs text-[#e3bf17]/60 leading-relaxed mb-6">
          {logs.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all">{line}</div>
          ))}
        </div>
      )}

      <p className="font-mono text-[10px] text-[#e3bf17]/30 tracking-[0.4em] uppercase mb-6">
        Make sure api_server.py is running on port 8000
      </p>

      <button
        onClick={() => { setStatus('idle'); setLogs([]); setErrorMsg(''); }}
        className="group relative px-10 py-3 border border-[#e3bf17]/50 overflow-hidden transition-all hover:shadow-[0_0_15px_rgba(227,191,23,0.4)]"
      >
        <span className="relative z-10 font-jersey text-2xl text-[#e3bf17] group-hover:text-black transition-colors duration-300 uppercase">
          Retry
        </span>
        <div className="absolute inset-0 bg-[#e3bf17] translate-y-full group-hover:translate-y-0 transition-transform duration-300 -z-0" />
      </button>
    </motion.div>
  );
};
