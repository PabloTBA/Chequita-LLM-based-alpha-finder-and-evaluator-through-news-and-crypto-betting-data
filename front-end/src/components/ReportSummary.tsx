import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { parseReport, type ReportData } from '../lib/mdParser';

interface Props { onBack: () => void; }

// ── Shared style tokens ──────────────────────────────────────────────────────
const GOLD     = '#e3bf17';
const GOLD_DIM = 'rgba(227,191,23,0.5)';
const GREEN    = '#4ade80';
const RED      = '#f87171';
const YELLOW   = '#fbbf24';

// ── Sub-components ───────────────────────────────────────────────────────────

function Chip({ label, color }: { label: string; color: 'green' | 'red' | 'gold' | 'yellow' }) {
  const cls = {
    green:  'border-green-500/50  text-green-400  bg-green-500/10',
    red:    'border-red-500/50    text-red-400    bg-red-500/10',
    gold:   'border-[#e3bf17]/50  text-[#e3bf17]  bg-[#e3bf17]/10',
    yellow: 'border-yellow-500/50 text-yellow-400 bg-yellow-500/10',
  }[color];
  return (
    <span className={`font-mono text-[10px] tracking-widest px-3 py-1 border rounded-full uppercase ${cls}`}>
      {label}
    </span>
  );
}

function verdictColor(v: string): 'green' | 'red' | 'yellow' | 'gold' {
  const u = v.toUpperCase();
  if (u === 'BUY')   return 'green';
  if (u === 'AVOID') return 'red';
  if (u === 'WATCH') return 'yellow';
  return 'gold';
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const col = verdictColor(verdict);
  const textCls = {
    green:  'text-green-400',
    red:    'text-red-400',
    yellow: 'text-yellow-400',
    gold:   'text-[#e3bf17]',
  }[col];
  const bgCls = {
    green:  'bg-green-500/15 border-green-500/40',
    red:    'bg-red-500/15   border-red-500/40',
    yellow: 'bg-yellow-500/15 border-yellow-500/40',
    gold:   'bg-[#e3bf17]/10 border-[#e3bf17]/40',
  }[col];
  return (
    <span className={`inline-block font-mono text-[10px] tracking-widest px-3 py-1 border rounded-full uppercase font-bold ${textCls} ${bgCls}`}>
      {verdict}
    </span>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-[#e3bf17]/20 rounded-xl p-4 bg-black/40">
      <p className="font-mono text-[9px] tracking-[0.3em] text-[#e3bf17]/40 uppercase mb-1">{label}</p>
      <p className="font-jersey text-2xl text-[#e3bf17]">{value || '—'}</p>
    </div>
  );
}

function VerdictBarChart({ data }: { data: ReportData }) {
  const counts = {
    BUY:   data.tickers.filter(t => t.verdict.toUpperCase() === 'BUY').length,
    WATCH: data.tickers.filter(t => t.verdict.toUpperCase() === 'WATCH').length,
    AVOID: data.tickers.filter(t => t.verdict.toUpperCase() === 'AVOID').length,
  };
  const chartData = [
    { name: 'BUY',   value: counts.BUY,   fill: GREEN },
    { name: 'WATCH', value: counts.WATCH, fill: YELLOW },
    { name: 'AVOID', value: counts.AVOID, fill: RED },
  ].filter(d => d.value > 0);

  if (!chartData.length) return null;

  return (
    <ResponsiveContainer width="100%" height={120}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
        <XAxis
          type="number"
          tick={{ fill: GOLD_DIM, fontSize: 10 }}
          tickLine={false}
          axisLine={{ stroke: GOLD_DIM }}
          allowDecimals={false}
        />
        <YAxis
          type="category"
          dataKey="name"
          tick={{ fill: GOLD_DIM, fontSize: 11, fontFamily: 'monospace' }}
          tickLine={false}
          axisLine={false}
          width={44}
        />
        <Tooltip
          contentStyle={{ background: '#0a0a0a', border: `1px solid ${GOLD_DIM}`, borderRadius: 8 }}
          labelStyle={{ color: GOLD, fontSize: 11 }}
          itemStyle={{ color: GOLD_DIM, fontSize: 11 }}
          formatter={(v: number) => [v, 'tickers']}
        />
        <Bar dataKey="value" radius={4} barSize={22}>
          {chartData.map((entry, i) => (
            <Cell key={i} fill={entry.fill} fillOpacity={0.75} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function TickersTable({ tickers }: { tickers: ReportData['tickers'] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  return (
    <div className="space-y-1">
      {tickers.map(t => (
        <div key={t.ticker} className="border border-[#e3bf17]/15 rounded-xl overflow-hidden">
          <button
            onClick={() => setExpanded(prev => prev === t.ticker ? null : t.ticker)}
            className="w-full flex items-center gap-5 px-5 py-3 hover:bg-[#e3bf17]/5 transition-colors text-left"
          >
            <span className="font-jersey text-2xl text-[#e3bf17] w-16">{t.ticker}</span>
            <VerdictBadge verdict={t.verdict} />
            <span className="font-mono text-xs text-[#e3bf17]/40 flex-1 truncate ml-2">
              {t.reasoning.slice(0, 90)}{t.reasoning.length > 90 ? '…' : ''}
            </span>
            <span className="font-mono text-[10px] text-[#e3bf17]/30 shrink-0">
              {expanded === t.ticker ? '▲' : '▼'}
            </span>
          </button>
          {expanded === t.ticker && (
            <div className="px-5 pb-4 border-t border-[#e3bf17]/10">
              <p className="font-mono text-xs text-[#e3bf17]/60 leading-relaxed pt-3">{t.reasoning}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function RegimeTable({ rows }: { rows: ReportData['regimes'] }) {
  if (!rows.length) return null;
  const regimeColors: Record<string, string> = {
    Trending:     'text-green-400',
    'Mean-Rev':   'text-yellow-400',
    'Event-Driv': 'text-blue-400',
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full font-mono text-xs border-collapse">
        <thead>
          <tr>
            {['Ticker', 'Regime', 'Hurst', 'ATR/Price'].map(h => (
              <th key={h} className="text-left text-[#e3bf17]/50 border-b border-[#e3bf17]/20 pb-2 pr-6 uppercase tracking-widest">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const colorKey = Object.keys(regimeColors).find(k => r.regime.startsWith(k)) ?? '';
            const regimeClass = regimeColors[colorKey] ?? 'text-[#e3bf17]/70';
            return (
              <tr key={i} className="border-b border-[#e3bf17]/10 hover:bg-[#e3bf17]/5 transition-colors">
                <td className="py-2 pr-6 text-[#e3bf17] font-bold">{r.ticker}</td>
                <td className={`py-2 pr-6 ${regimeClass}`}>{r.regime}</td>
                <td className="py-2 pr-6 text-[#e3bf17]/70">{r.hurst}</td>
                <td className="py-2 pr-6 text-[#e3bf17]/70">{r.atrPct}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export const ReportSummary = ({ onBack }: Props) => {
  const [state, setState] = useState<'loading' | 'no-data' | 'error' | 'loaded'>('loading');
  const [data,  setData]  = useState<ReportData | null>(null);

  useEffect(() => {
    fetch('/api/report')
      .then(async r => {
        if (r.status === 404) { setState('no-data'); return; }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const md = await r.text();
        setData(parseReport(md));
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
          Report_Summary_v2.0
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
            {/* Executive Summary */}
            <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-6">
              <div className="flex items-start justify-between flex-wrap gap-4">
                <div>
                  <h2 className="font-jersey text-6xl text-[#e3bf17] uppercase">
                    Report_Summary
                  </h2>
                  <p className="font-mono text-xs text-[#e3bf17]/40 tracking-widest mt-1">
                    {data.exec.runDate}  {data.exec.newsWindow && `· ${data.exec.newsWindow}`}
                  </p>
                </div>
                <Chip label={`Bias: ${data.exec.marketBias}`} color="gold" />
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <StatCard label="Articles Analysed" value={data.exec.articlesAnalysed} />
                <StatCard label="Buy Signals"  value={String(data.exec.buyCandidates.length)} />
                <StatCard label="Watch List"   value={String(data.exec.watchList.length)} />
                <StatCard label="Avoid"        value={String(data.exec.avoidList.length)} />
              </div>

              {data.exec.buyCandidates.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {data.exec.buyCandidates.map(s => <Chip key={s} label={s} color="green" />)}
                  {data.exec.watchList.map(s => <Chip key={s} label={s} color="yellow" />)}
                  {data.exec.avoidList.map(s => <Chip key={s} label={s} color="red" />)}
                </div>
              )}
            </div>

            {/* Verdict Distribution */}
            {data.tickers.length > 0 && (
              <div className="border border-[#e3bf17]/25 rounded-2xl p-8">
                <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase mb-6">
                  Verdict Distribution
                </h3>
                <VerdictBarChart data={data} />
              </div>
            )}

            {/* Macro Environment */}
            {(data.macroSummary || data.favouredSectors.length > 0) && (
              <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-5">
                <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
                  Macro Environment
                </h3>

                {data.macroSummary && (
                  <blockquote className="border-l-2 border-[#e3bf17]/40 pl-4 font-mono text-sm text-[#e3bf17]/60 italic leading-relaxed">
                    {data.macroSummary}
                  </blockquote>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {data.favouredSectors.length > 0 && (
                    <div>
                      <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">
                        Favoured
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {data.favouredSectors.map(s => <Chip key={s} label={s} color="green" />)}
                      </div>
                    </div>
                  )}
                  {data.avoidSectors.length > 0 && (
                    <div>
                      <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">
                        Avoid
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {data.avoidSectors.map(s => <Chip key={s} label={s} color="red" />)}
                      </div>
                    </div>
                  )}
                </div>

                {data.exec.keyRisks && (
                  <div className="border-t border-[#e3bf17]/10 pt-4">
                    <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-1">
                      Key Risks
                    </p>
                    <p className="font-mono text-xs text-[#e3bf17]/60 leading-relaxed">
                      {data.exec.keyRisks}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Shortlisted Tickers */}
            {data.tickers.length > 0 && (
              <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-5">
                <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
                  Shortlisted Tickers ({data.tickers.length})
                </h3>
                <TickersTable tickers={data.tickers} />
              </div>
            )}

            {/* Regime Classification */}
            {data.regimes.length > 0 && (
              <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-5">
                <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
                  Regime Classification
                </h3>
                <RegimeTable rows={data.regimes} />
              </div>
            )}
          </>
        )}
      </div>
    </motion.div>
  );
};
