import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine, LabelList,
} from 'recharts';
import { parseReport, type ReportData, type StrategyTicker } from '../lib/mdParser';

interface Props { onBack: () => void; }

type Section = 'overview' | 'tickers' | 'regime' | 'strategy';

// ── Style tokens ──────────────────────────────────────────────────────────────
const GOLD     = '#e3bf17';
const GOLD_DIM = 'rgba(227,191,23,0.5)';
const GREEN    = '#4ade80';
const RED      = '#f87171';
const YELLOW   = '#fbbf24';

// ── Shared primitives ─────────────────────────────────────────────────────────

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
  const map = {
    green:  { text: 'text-green-400',  bg: 'bg-green-500/15  border-green-500/40' },
    red:    { text: 'text-red-400',    bg: 'bg-red-500/15    border-red-500/40' },
    yellow: { text: 'text-yellow-400', bg: 'bg-yellow-500/15 border-yellow-500/40' },
    gold:   { text: 'text-[#e3bf17]',  bg: 'bg-[#e3bf17]/10  border-[#e3bf17]/40' },
  }[col];
  return (
    <span className={`inline-block font-mono text-[10px] tracking-widest px-3 py-1 border rounded-full uppercase font-bold ${map.text} ${map.bg}`}>
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

// ── Tab bar ───────────────────────────────────────────────────────────────────

function TabBar({ tabs, active, onSelect, small }: {
  tabs: { id: string; label: string }[];
  active: string;
  onSelect: (id: string) => void;
  small?: boolean;
}) {
  const base = small
    ? 'font-mono text-[9px] tracking-[0.25em] px-3 py-2'
    : 'font-mono text-[10px] tracking-[0.3em] px-5 py-3';
  return (
    <div className="flex overflow-x-auto gap-0 border-b border-[#e3bf17]/15 scrollbar-hide">
      {tabs.map(t => (
        <button
          key={t.id}
          onClick={() => onSelect(t.id)}
          className={`${base} whitespace-nowrap uppercase transition-all border-b-2 -mb-px shrink-0 ${
            active === t.id
              ? 'text-[#e3bf17] border-[#e3bf17]'
              : 'text-[#e3bf17]/35 border-transparent hover:text-[#e3bf17]/60 hover:border-[#e3bf17]/30'
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

// ── Charts ────────────────────────────────────────────────────────────────────

function HurstChart({ regimes }: { regimes: ReportData['regimes'] }) {
  if (!regimes.length) return null;
  const data = regimes.map(r => ({ ticker: r.ticker, hurst: parseFloat(r.hurst) || 0 }));
  const h = Math.max(180, data.length * 28);
  return (
    <div className="bg-black/40 border border-[#e3bf17]/15 rounded-xl p-4">
      <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-3">
        Hurst Exponent (H &gt; 0.5 = trending, H &gt; 0.7 = strongly trending)
      </p>
      <ResponsiveContainer width="100%" height={h}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 50, top: 4, bottom: 4 }}>
          <XAxis type="number" domain={[0, 1]} tick={{ fill: GOLD_DIM, fontSize: 9 }}
            tickLine={false} axisLine={{ stroke: GOLD_DIM }} />
          <YAxis type="category" dataKey="ticker" width={44}
            tick={{ fill: GOLD_DIM, fontSize: 10, fontFamily: 'monospace' }}
            tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: '#0a0a0a', border: `1px solid ${GOLD_DIM}`, borderRadius: 8 }}
            labelStyle={{ color: GOLD, fontSize: 11 }}
            itemStyle={{ color: GOLD_DIM, fontSize: 11 }}
            formatter={(v: number) => [v.toFixed(3), 'Hurst']}
          />
          <ReferenceLine x={0.5} stroke={RED}    strokeDasharray="3 3" strokeOpacity={0.5} />
          <ReferenceLine x={0.7} stroke={YELLOW} strokeDasharray="3 3" strokeOpacity={0.5} />
          <Bar dataKey="hurst" radius={3} barSize={16} fill={GOLD} fillOpacity={0.7}>
            <LabelList dataKey="hurst" position="right"
              formatter={(v: number) => v.toFixed(3)}
              style={{ fill: GOLD_DIM, fontSize: 9, fontFamily: 'monospace' }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex gap-6 mt-2 justify-center">
        {[
          { label: 'Random walk (0.5)', color: RED },
          { label: 'Strong trend (0.7)', color: YELLOW },
        ].map(({ label, color }) => (
          <div key={label} className="flex items-center gap-2">
            <div className="w-6 border-t-2 border-dashed" style={{ borderColor: color, opacity: 0.5 }} />
            <span className="font-mono text-[9px] text-[#e3bf17]/40">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function AtrChart({ regimes }: { regimes: ReportData['regimes'] }) {
  if (!regimes.length) return null;
  const data = regimes.map(r => ({
    ticker: r.ticker,
    atrPct: parseFloat(r.atrPct.replace('%', '')) || 0,
  }));
  const h = Math.max(180, data.length * 28);
  return (
    <div className="bg-black/40 border border-[#e3bf17]/15 rounded-xl p-4">
      <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-3">
        ATR / Price % (daily volatility — higher = wider stops needed)
      </p>
      <ResponsiveContainer width="100%" height={h}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 50, top: 4, bottom: 4 }}>
          <XAxis type="number" tick={{ fill: GOLD_DIM, fontSize: 9 }}
            tickLine={false} axisLine={{ stroke: GOLD_DIM }}
            tickFormatter={(v: number) => `${v}%`} />
          <YAxis type="category" dataKey="ticker" width={44}
            tick={{ fill: GOLD_DIM, fontSize: 10, fontFamily: 'monospace' }}
            tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: '#0a0a0a', border: `1px solid ${GOLD_DIM}`, borderRadius: 8 }}
            labelStyle={{ color: GOLD, fontSize: 11 }}
            itemStyle={{ color: GOLD_DIM, fontSize: 11 }}
            formatter={(v: number) => [`${v.toFixed(2)}%`, 'ATR/Price']}
          />
          <Bar dataKey="atrPct" radius={3} barSize={16} fill={GOLD} fillOpacity={0.55}>
            <LabelList dataKey="atrPct" position="right"
              formatter={(v: number) => `${v.toFixed(2)}%`}
              style={{ fill: GOLD_DIM, fontSize: 9, fontFamily: 'monospace' }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
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
    <ResponsiveContainer width="100%" height={110}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 0, right: 30, top: 0, bottom: 0 }}>
        <XAxis type="number" tick={{ fill: GOLD_DIM, fontSize: 10 }}
          tickLine={false} axisLine={{ stroke: GOLD_DIM }} allowDecimals={false} />
        <YAxis type="category" dataKey="name" width={44}
          tick={{ fill: GOLD_DIM, fontSize: 11, fontFamily: 'monospace' }}
          tickLine={false} axisLine={false} />
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

// ── Section views ─────────────────────────────────────────────────────────────

function OverviewSection({ data }: { data: ReportData }) {
  return (
    <div className="space-y-6">
      {/* Executive summary */}
      <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-6">
        <div className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h2 className="font-jersey text-5xl text-[#e3bf17] uppercase">Executive Summary</h2>
            <p className="font-mono text-xs text-[#e3bf17]/40 tracking-widest mt-1">
              {data.exec.runDate}{data.exec.newsWindow ? ` · ${data.exec.newsWindow}` : ''}
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

        {data.tickers.length > 0 && (
          <div className="space-y-2">
            <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase">
              Verdict Distribution
            </p>
            <VerdictBarChart data={data} />
            <div className="flex flex-wrap gap-2 mt-2">
              {data.exec.buyCandidates.map(s => <Chip key={s} label={s} color="green" />)}
              {data.exec.watchList.map(s => <Chip key={s} label={s} color="yellow" />)}
              {data.exec.avoidList.map(s => <Chip key={s} label={s} color="red" />)}
            </div>
          </div>
        )}

        {data.exec.topThemes && (
          <div className="border-t border-[#e3bf17]/10 pt-4">
            <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-1">Top Themes</p>
            <p className="font-mono text-xs text-[#e3bf17]/60 leading-relaxed">{data.exec.topThemes}</p>
          </div>
        )}
      </div>

      {/* Macro environment */}
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
                <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">Favoured</p>
                <div className="flex flex-wrap gap-2">
                  {data.favouredSectors.map(s => <Chip key={s} label={s} color="green" />)}
                </div>
              </div>
            )}
            {data.avoidSectors.length > 0 && (
              <div>
                <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">Avoid</p>
                <div className="flex flex-wrap gap-2">
                  {data.avoidSectors.map(s => <Chip key={s} label={s} color="red" />)}
                </div>
              </div>
            )}
          </div>
          {data.exec.keyRisks && (
            <div className="border-t border-[#e3bf17]/10 pt-4">
              <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-1">Key Risks</p>
              <p className="font-mono text-xs text-[#e3bf17]/60 leading-relaxed">{data.exec.keyRisks}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TickersSection({ data }: { data: ReportData }) {
  const [active, setActive] = useState(data.tickers[0]?.ticker ?? '');

  const tickerTabs = data.tickers.map(t => ({ id: t.ticker, label: t.ticker }));
  const selected   = data.tickers.find(t => t.ticker === active);
  const regime     = data.regimes.find(r => r.ticker === active);
  const strategy   = data.strategies.find(s => s.ticker === active);
  const col        = selected ? verdictColor(selected.verdict) : 'gold';

  const verdictBgMap = {
    green:  'bg-green-500/10  border-green-500/25',
    red:    'bg-red-500/10    border-red-500/25',
    yellow: 'bg-yellow-500/10 border-yellow-500/25',
    gold:   'bg-[#e3bf17]/8   border-[#e3bf17]/25',
  };

  return (
    <div className="space-y-4">
      <TabBar tabs={tickerTabs} active={active} onSelect={setActive} small />

      {selected && (
        <div className={`border rounded-2xl p-6 space-y-5 ${verdictBgMap[col]}`}>
          <div className="flex items-center gap-4 flex-wrap">
            <span className="font-jersey text-5xl text-[#e3bf17]">{selected.ticker}</span>
            <VerdictBadge verdict={selected.verdict} />
            {regime && <Chip label={regime.regime} color="gold" />}
            {strategy && <Chip label={strategy.strategy} color="gold" />}
          </div>

          <div>
            <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">Reasoning</p>
            <p className="font-mono text-sm text-[#e3bf17]/70 leading-relaxed">{selected.reasoning}</p>
          </div>

          {regime && (
            <div>
              <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">Regime Data</p>
              <MdTable rows={[{
                Ticker: regime.ticker,
                Regime: regime.regime,
                'Hurst Exponent': regime.hurst,
                'ATR / Price': regime.atrPct,
              }]} />
            </div>
          )}

          {strategy && (
            <div>
              <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">Entry Signal</p>
              <div className={`font-mono text-xs px-4 py-3 rounded-lg border ${
                strategy.entryStatus.includes('ACTIVE')
                  ? 'text-green-400 border-green-500/30 bg-green-500/5'
                  : 'text-[#e3bf17]/50 border-[#e3bf17]/15 bg-black/30'
              }`}>
                {strategy.entryStatus || 'No entry signal data'}
                {strategy.entryDetails && (
                  <div className="mt-1 text-[#e3bf17]/40 text-[10px]">{strategy.entryDetails}</div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RegimeSection({ data }: { data: ReportData }) {
  return (
    <div className="space-y-6">
      <div className="border border-[#e3bf17]/25 rounded-2xl p-8 space-y-4">
        <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
          Regime Classification Table
        </h3>
        <MdTable rows={data.regimes.map(r => ({
          Ticker: r.ticker,
          Regime: r.regime,
          'Hurst': r.hurst,
          'ATR/Price': r.atrPct,
        }))} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="border border-[#e3bf17]/25 rounded-2xl p-6 space-y-4">
          <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
            Hurst Exponent by Ticker
          </h3>
          <HurstChart regimes={data.regimes} />
        </div>

        <div className="border border-[#e3bf17]/25 rounded-2xl p-6 space-y-4">
          <h3 className="font-mono text-[10px] tracking-[0.4em] text-[#e3bf17]/50 uppercase">
            Daily Volatility (ATR/Price) by Ticker
          </h3>
          <AtrChart regimes={data.regimes} />
        </div>
      </div>
    </div>
  );
}

function StrategyDetail({ strat }: { strat: StrategyTicker }) {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4 flex-wrap">
        <span className="font-jersey text-5xl text-[#e3bf17]">{strat.ticker}</span>
        <Chip label={strat.strategy} color="gold" />
        <Chip label={strat.regime} color="gold" />
      </div>

      {strat.reasoning && (
        <div>
          <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-1">LLM Reasoning</p>
          <p className="font-mono text-xs text-[#e3bf17]/60 leading-relaxed">{strat.reasoning}</p>
        </div>
      )}

      {/* Entry conditions + exit rules in two columns */}
      {(strat.entryConditions.length > 0 || strat.exitRules.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {strat.entryConditions.length > 0 && (
            <div className="border border-[#e3bf17]/15 rounded-xl p-4">
              <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-3">
                Entry Conditions
              </p>
              <ul className="space-y-1">
                {strat.entryConditions.map((c, i) => (
                  <li key={i} className="font-mono text-[10px] text-[#e3bf17]/60 flex gap-2">
                    <span className="text-[#e3bf17]/30 shrink-0">›</span>{c}
                  </li>
                ))}
              </ul>
              {strat.positionSizing && (
                <div className="border-t border-[#e3bf17]/10 pt-2 mt-2">
                  <p className="font-mono text-[9px] text-[#e3bf17]/40 uppercase mb-1">Sizing</p>
                  <p className="font-mono text-[10px] text-[#e3bf17]/60">{strat.positionSizing}</p>
                </div>
              )}
            </div>
          )}
          {strat.exitRules.length > 0 && (
            <div className="border border-[#e3bf17]/15 rounded-xl p-4">
              <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-3">
                Exit Rules (priority order)
              </p>
              <ol className="space-y-1 list-none">
                {strat.exitRules.map((r, i) => (
                  <li key={i} className="font-mono text-[10px] text-[#e3bf17]/60 flex gap-2">
                    <span className="text-[#e3bf17]/30 shrink-0 w-4">{i + 1}.</span>{r}
                  </li>
                ))}
              </ol>
            </div>
          )}
        </div>
      )}

      {/* Adjusted Parameters table */}
      {strat.adjustedParams.length > 0 && (
        <div className="border border-[#e3bf17]/25 rounded-xl p-5 space-y-3">
          <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase">
            Adjusted Parameters
          </p>
          <MdTable rows={strat.adjustedParams.map(p => ({ Parameter: p.parameter, Value: p.value }))} />
        </div>
      )}

      {/* LLM adjustments */}
      {strat.llmAdjustments.length > 0 && (
        <div className="border border-[#e3bf17]/15 rounded-xl p-5 space-y-2">
          <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase mb-2">
            LLM Parameter Adjustments
          </p>
          <ul className="space-y-2">
            {strat.llmAdjustments.map((a, i) => (
              <li key={i} className="font-mono text-[10px] text-[#e3bf17]/60 flex gap-2 leading-relaxed">
                <span className="text-[#e3bf17]/30 shrink-0">→</span>{a}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Entry signal */}
      <div className="border border-[#e3bf17]/15 rounded-xl p-5 space-y-2">
        <p className="font-mono text-[9px] tracking-[0.35em] text-[#e3bf17]/40 uppercase">
          Current Entry Signal
        </p>
        <div className={`font-mono text-xs px-4 py-3 rounded-lg border ${
          strat.entryStatus.includes('ACTIVE')
            ? 'text-green-400 border-green-500/30 bg-green-500/5'
            : 'text-[#e3bf17]/50 border-[#e3bf17]/15 bg-black/30'
        }`}>
          {strat.entryStatus || '—'}
          {strat.entryDetails && (
            <div className="mt-1 text-[#e3bf17]/40 text-[10px]">{strat.entryDetails}</div>
          )}
        </div>
      </div>
    </div>
  );
}

function StrategySection({ data }: { data: ReportData }) {
  const [active, setActive] = useState(data.strategies[0]?.ticker ?? '');

  if (!data.strategies.length) {
    return (
      <div className="border border-[#e3bf17]/15 rounded-2xl p-10 text-center">
        <p className="font-mono text-[#e3bf17]/40 text-sm">No strategy parameters found in report.</p>
      </div>
    );
  }

  const tabs     = data.strategies.map(s => ({ id: s.ticker, label: s.ticker }));
  const selected = data.strategies.find(s => s.ticker === active);

  return (
    <div className="space-y-4">
      <TabBar tabs={tabs} active={active} onSelect={setActive} small />
      {selected && (
        <div className="border border-[#e3bf17]/25 rounded-2xl p-6">
          <StrategyDetail strat={selected} />
        </div>
      )}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export const ReportSummary = ({ onBack }: Props) => {
  const [state,   setState]   = useState<'loading' | 'no-data' | 'error' | 'loaded'>('loading');
  const [data,    setData]    = useState<ReportData | null>(null);
  const [section, setSection] = useState<Section>('overview');

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

  const sectionTabs = data ? [
    { id: 'overview',  label: 'Overview' },
    { id: 'tickers',   label: `Tickers (${data.tickers.length})` },
    { id: 'regime',    label: `Regime (${data.regimes.length})` },
    { id: 'strategy',  label: `Strategy (${data.strategies.length})` },
  ] : [];

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

      {/* Section tab bar (sticky below nav) */}
      {state === 'loaded' && data && (
        <div className="sticky top-[73px] z-10 bg-black/90 backdrop-blur px-10">
          <TabBar tabs={sectionTabs} active={section} onSelect={id => setSection(id as Section)} />
        </div>
      )}

      <div className="flex-1 px-10 py-8 max-w-6xl mx-auto w-full">

        {state === 'loading' && (
          <div className="flex flex-col items-center justify-center h-64">
            <p className="font-mono text-[#e3bf17]/60 text-sm animate-pulse tracking-widest">
              FETCHING_DATA_STREAM...
            </p>
          </div>
        )}

        {state === 'no-data' && (
          <div className="flex flex-col items-center justify-center h-64 gap-4">
            <p className="font-mono text-[#e3bf17]/60 text-sm animate-pulse">[!] NO_ACTIVE_DATA_STREAM</p>
            <p className="font-mono text-[#e3bf17]/30 text-xs tracking-widest">
              Run the pipeline via Generate_Module first
            </p>
          </div>
        )}

        {state === 'error' && (
          <div className="flex flex-col items-center justify-center h-64">
            <p className="font-mono text-red-400/80 text-sm">[!] STREAM_ERROR — check api_server.py</p>
          </div>
        )}

        {state === 'loaded' && data && (
          <div className="pt-6">
            {section === 'overview'  && <OverviewSection  data={data} />}
            {section === 'tickers'   && <TickersSection   data={data} />}
            {section === 'regime'    && <RegimeSection    data={data} />}
            {section === 'strategy'  && <StrategySection  data={data} />}
          </div>
        )}
      </div>
    </motion.div>
  );
};
