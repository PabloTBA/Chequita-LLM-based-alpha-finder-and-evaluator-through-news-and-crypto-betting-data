import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';
import {
  parseTraderSummary,
  type TraderData,
  type TickerSection,
} from '../lib/mdParser';

interface Props { onBack: () => void; }

// ── Shared style tokens ──────────────────────────────────────────────────────
const GOLD    = '#e3bf17';
const GOLD_DIM = 'rgba(227,191,23,0.5)';
const GOLD_BG  = 'rgba(227,191,23,0.08)';

// ── Sub-components ───────────────────────────────────────────────────────────

function Chip({ label, color }: { label: string; color: 'green' | 'red' | 'gold' }) {
  const cls = {
    green: 'border-green-500/50 text-green-400 bg-green-500/10',
    red:   'border-red-500/50   text-red-400   bg-red-500/10',
    gold:  'border-[#e3bf17]/50 text-[#e3bf17] bg-[#e3bf17]/10',
  }[color];
  return (
    <span className={`font-mono text-[10px] tracking-widest px-3 py-1 border rounded-full uppercase ${cls}`}>
      {label}
    </span>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-[#e3bf17]/20 rounded-xl p-4 bg-black/40 flex flex-col gap-1">
      <span className="font-mono text-[9px] tracking-[0.3em] text-[#e3bf17]/40 uppercase">{label}</span>
      <span className="font-jersey text-2xl text-[#e3bf17]">{value || '—'}</span>
    </div>
  );
}

function MdTable({ rows }: { rows: Record<string, string>[] }) {
  if (!rows.length) return null;
  const headers = Object.keys(rows[0]);
  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-xs border-collapse">
        <thead>
          <tr>
            {headers.map(h => (
              <th key={h} className="text-left text-[#e3bf17]/50 border-b border-[#e3bf17]/20 pb-2 pr-6 uppercase tracking-widest">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-[#e3bf17]/10 hover:bg-[#e3bf17]/5 transition-colors">
              {headers.map(h => (
                <td key={h} className="py-2 pr-6 text-[#e3bf17]/70 align-top">{row[h]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiagScorecard({ rows }: { rows: { metric: string; value: string; floor: string; pass: string }[] }) {
  return (
    <div className="grid grid-cols-1 gap-2">
      {rows.map((r, i) => (
        <div key={i} className="flex items-center justify-between border border-[#e3bf17]/10 rounded-lg px-4 py-2">
          <span className="font-mono text-xs text-[#e3bf17]/70 flex-1">{r.metric}</span>
          <span className="font-mono text-xs text-[#e3bf17] w-24 text-right">{r.value}</span>
          <span className="font-mono text-[10px] text-[#e3bf17]/40 w-28 text-right">{r.floor}</span>
          <span className="ml-4 text-base">{r.pass.includes('✅') ? '✅' : r.pass.includes('❌') ? '❌' : r.pass}</span>
        </div>
      ))}
    </div>
  );
}

function EquityChart({ data }: { data: { trade: number; p5: number; median: number; p95: number }[] }) {
  if (!data.length) return (
    <p className="font-mono text-xs text-[#e3bf17]/30 text-center py-8">No equity band data</p>
  );

  const fmt = (v: number) => `$${(v / 1000).toFixed(0)}k`;

  return (
    <ResponsiveContainer width="100%" height={260}>
      <AreaChart data={data} margin={{ top: 10, right: 20, left: 10, bottom: 0 }}>
        <defs>
          <linearGradient id="bandGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%"  stopColor={GOLD} stopOpacity={0.2} />
            <stop offset="95%" stopColor={GOLD} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(227,191,23,0.08)" />
        <XAxis
          dataKey="trade"
          tick={{ fill: GOLD_DIM, fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: GOLD_DIM }}
          label={{ value: 'Trade #', position: 'insideBottomRight', fill: GOLD_DIM, fontSize: 10, offset: -5 }}
        />
        <YAxis
          tickFormatter={fmt}
          tick={{ fill: GOLD_DIM, fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: GOLD_DIM }}
          width={48}
        />
        <Tooltip
          contentStyle={{ background: '#0a0a0a', border: `1px solid ${GOLD_DIM}`, borderRadius: 8 }}
          labelStyle={{ color: GOLD, fontSize: 11, marginBottom: 4 }}
          itemStyle={{ color: GOLD_DIM, fontSize: 11 }}
          formatter={(v: number, name: string) => [fmt(v), name]}
          labelFormatter={(t) => `Trade #${t}`}
        />
        {/* P95 top bound — light fill area from bottom */}
        <Area type="monotone" dataKey="p95"    stroke={GOLD}     strokeWidth={1.5} fill="url(#bandGrad)" strokeDasharray="4 2" name="P95 (best 5%)" dot={false} />
        {/* Median — solid bright line, no fill */}
        <Area type="monotone" dataKey="median" stroke={GOLD}     strokeWidth={2.5} fill="none" name="Median"        dot={false} />
        {/* P5 lower bound — dimmed, no fill */}
        <Area type="monotone" dataKey="p5"     stroke={GOLD_DIM} strokeWidth={1.5} fill="black" fillOpacity={0} strokeDasharray="4 2" name="P5 (worst 5%)" dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function TickerCard({ t, expanded, onToggle }: {
  t: TickerSection;
  expanded: boolean;
  onToggle: () => void;
}) {
  const isActive = t.status.toLowerCase().includes('active');

  return (
    <div className="border border-[#e3bf17]/25 rounded-2xl overflow-hidden">
      {/* Header row — always visible */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-8 py-5 hover:bg-[#e3bf17]/5 transition-colors text-left"
      >
        <div className="flex items-center gap-5">
          <span className="font-jersey text-4xl text-[#e3bf17]">{t.ticker}</span>
          <Chip label={t.regime}   color="gold" />
          <Chip label={t.strategy} color="gold" />
          {isActive && <Chip label="ACTIVE" color="green" />}
        </div>
        <span className="font-mono text-[#e3bf17]/40 text-sm">{expanded ? '▲ collapse' : '▼ expand'}</span>
      </button>

      <AnimatePresence initial={false}>
        {expanded && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-8 pb-8 space-y-8 border-t border-[#e3bf17]/15">

              {/* Entry Signal */}
              <section className="pt-6">
                <h4 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase mb-4">
                  Entry Signal
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  <StatCard label="Entry Price"    value={t.entryPrice} />
                  <StatCard label="Stop Loss"      value={t.stopLoss} />
                  <StatCard label="Stop Distance"  value={t.stopDistance} />
                  <StatCard label="Position Size"  value={t.positionSize} />
                  <StatCard label="Dollar Risk"    value={t.dollarRisk} />
                  <StatCard label="ATR₁₄"          value={t.atr14} />
                </div>
              </section>

              {/* Screening Data */}
              {t.screeningRows.length > 0 && (
                <section>
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
                      Screening Data
                    </h4>
                    {t.screenerVerdict && (
                      <Chip
                        label={`Verdict: ${t.screenerVerdict}`}
                        color={t.screenerVerdict === 'BUY' ? 'green' : 'gold'}
                      />
                    )}
                  </div>
                  <MdTable rows={t.screeningRows} />
                </section>
              )}

              {/* Equity Confidence Band */}
              {t.equityBand.length > 0 && (
                <section>
                  <h4 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase mb-4">
                    Monte Carlo — Equity Confidence Band (10,000 sims)
                  </h4>
                  <div className="bg-black/40 border border-[#e3bf17]/15 rounded-xl p-4">
                    <EquityChart data={t.equityBand} />
                    <div className="flex gap-6 mt-3 justify-center">
                      {[
                        { label: 'P95 Best 5%',  style: { borderTop: `2px dashed ${GOLD}` } },
                        { label: 'Median',        style: { borderTop: `2.5px solid ${GOLD}` } },
                        { label: 'P5 Worst 5%',  style: { borderTop: `2px dashed ${GOLD_DIM}` } },
                      ].map(({ label, style }) => (
                        <div key={label} className="flex items-center gap-2">
                          <div className="w-8" style={style} />
                          <span className="font-mono text-[10px] text-[#e3bf17]/50">{label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {/* Diagnostic Scorecard */}
              {t.diagnostics.length > 0 && (
                <section>
                  <h4 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase mb-4">
                    Diagnostic Scorecard
                  </h4>
                  <DiagScorecard rows={t.diagnostics} />
                </section>
              )}

              {/* Backtest summary */}
              {(t.backtestReturn || t.backtestWinRate) && (
                <section>
                  <h4 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase mb-4">
                    Backtest Performance
                  </h4>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <StatCard label="Net Return"  value={t.backtestReturn} />
                    <StatCard label="Win Rate"    value={t.backtestWinRate} />
                    <StatCard label="Trade Count" value={t.tradeCount} />
                    <StatCard label="Max Drawdown" value={t.maxDrawdown} />
                  </div>
                </section>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export const TraderSummary = ({ onBack }: Props) => {
  const [state, setState] = useState<'loading' | 'no-data' | 'error' | 'loaded'>('loading');
  const [data,  setData]  = useState<TraderData | null>(null);
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);

  useEffect(() => {
    fetch('/api/summary')
      .then(async r => {
        if (r.status === 404) { setState('no-data'); return; }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const md = await r.text();
        setData(parseTraderSummary(md));
        setState('loaded');
      })
      .catch(() => setState('error'));
  }, []);

  return (
    <motion.div
      initial={{ x: '100vw' }}
      animate={{ x: 0 }}
      exit={{ x: '100vw' }}
      transition={{ type: 'spring', damping: 25, stiffness: 120 }}
      className="absolute inset-0 flex flex-col bg-black/95 backdrop-blur-3xl z-[100] overflow-y-auto"
    >
      {/* Nav bar */}
      <div className="sticky top-0 z-10 bg-black/80 backdrop-blur border-b border-[#e3bf17]/15 px-10 py-5 flex items-center justify-between shrink-0">
        <button
          onClick={onBack}
          className="font-jersey text-2xl text-[#e3bf17] hover:tracking-[0.2em] transition-all uppercase flex items-center gap-3 group"
        >
          <span className="group-hover:-translate-x-1 transition-transform">{'<<<'}</span>
          Return_to_Hub
        </button>
        <span className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/40 uppercase">
          Trader_Summary_v2.0
        </span>
      </div>

      <div className="flex-1 px-10 py-8 max-w-6xl mx-auto w-full space-y-8">

        {/* ── Loading ── */}
        {state === 'loading' && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <p className="font-mono text-[#e3bf17]/60 text-sm animate-pulse tracking-widest">
              FETCHING_DATA_STREAM...
            </p>
          </div>
        )}

        {/* ── No data ── */}
        {state === 'no-data' && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <p className="font-mono text-[#e3bf17]/60 text-sm animate-pulse">[!] NO_ACTIVE_DATA_STREAM</p>
            <p className="font-mono text-[#e3bf17]/30 text-xs tracking-widest">
              Run the pipeline via Generate_Module first
            </p>
          </div>
        )}

        {/* ── Error ── */}
        {state === 'error' && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <p className="font-mono text-red-400/80 text-sm">[!] STREAM_ERROR — check api_server.py</p>
          </div>
        )}

        {/* ── Loaded ── */}
        {state === 'loaded' && data && (
          <>
            {/* Header */}
            <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-6">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div>
                  <h2 className="font-jersey text-6xl text-[#e3bf17] uppercase">
                    TRADER_SUMMARY
                  </h2>
                  <p className="font-mono text-xs text-[#e3bf17]/40 tracking-widest mt-1">
                    {data.header.date}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <Chip
                    label={`${data.header.tickersPassed} ticker${data.header.tickersPassed !== 1 ? 's' : ''} passed`}
                    color={data.header.tickersPassed > 0 ? 'green' : 'red'}
                  />
                  <Chip
                    label={`Bias: ${data.header.marketBias}`}
                    color="gold"
                  />
                  {data.header.spyBenchmark && (
                    <span className="font-mono text-[10px] text-[#e3bf17]/40">
                      SPY benchmark: {data.header.spyBenchmark}
                    </span>
                  )}
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">
                    Favoured Sectors
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {data.header.favouredSectors.map(s => (
                      <Chip key={s} label={s} color="green" />
                    ))}
                  </div>
                </div>
                <div>
                  <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">
                    Avoid Sectors
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {data.header.avoidSectors.map(s => (
                      <Chip key={s} label={s} color="red" />
                    ))}
                  </div>
                </div>
              </div>

              {data.header.keyRisks && (
                <div className="border-t border-[#e3bf17]/10 pt-4">
                  <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-1">
                    Key Risks
                  </p>
                  <p className="font-mono text-xs text-[#e3bf17]/60 leading-relaxed">
                    {data.header.keyRisks}
                  </p>
                </div>
              )}
            </div>

            {/* Ticker deep-dives */}
            {data.tickers.length === 0 ? (
              <div className="border border-[#e3bf17]/15 rounded-2xl p-10 text-center">
                <p className="font-mono text-[#e3bf17]/40 text-sm">
                  No tickers passed all 3 stages today. No action required.
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
                  Qualified Tickers ({data.tickers.length})
                </h3>
                {data.tickers.map(t => (
                  <TickerCard
                    key={t.ticker}
                    t={t}
                    expanded={expandedTicker === t.ticker}
                    onToggle={() =>
                      setExpandedTicker(prev => prev === t.ticker ? null : t.ticker)
                    }
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
};
