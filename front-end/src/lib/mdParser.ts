/** Parse all markdown tables in a text block into arrays of row objects */
export function findTables(text: string): Record<string, string>[][] {
  const tables: Record<string, string>[][] = [];
  const lines = text.split('\n');
  let block: string[] = [];

  for (const raw of lines) {
    const line = raw.trim();
    if (line.startsWith('|')) {
      block.push(line);
    } else if (block.length > 0) {
      if (block.length >= 3) tables.push(parseTableBlock(block));
      block = [];
    }
  }
  if (block.length >= 3) tables.push(parseTableBlock(block));
  return tables;
}

function parseTableBlock(lines: string[]): Record<string, string>[] {
  const headers = lines[0].split('|').slice(1, -1).map(h => h.trim());
  return lines.slice(2).map(row => {
    const cells = row.split('|').slice(1, -1).map(c => c.trim());
    const obj: Record<string, string> = {};
    headers.forEach((h, i) => { obj[h] = cells[i] ?? ''; });
    return obj;
  });
}

/** Extract the text between two substrings */
export function slice(text: string, from: string, to?: string): string {
  const start = text.indexOf(from);
  if (start === -1) return '';
  const body = text.slice(start + from.length);
  if (!to) return body.trim();
  const end = body.indexOf(to);
  return (end === -1 ? body : body.slice(0, end)).trim();
}

/** Strip markdown bold/italic markers */
export function strip(s: string): string {
  return s.replace(/\*\*/g, '').replace(/\*/g, '').replace(/_/g, '').trim();
}

/** Parse a numeric value from strings like "$100,000", "5.47%", "0.22×" */
export function num(s: string): number {
  return parseFloat(s.replace(/[$,%×]/g, '').replace(/,/g, '')) || 0;
}

// ── TraderSummary types & parser ─────────────────────────────────────────────

export interface TraderHeader {
  date: string;
  tickersPassed: number;
  spyBenchmark: string;
  marketBias: string;
  favouredSectors: string[];
  avoidSectors: string[];
  keyRisks: string;
}

export interface EquityPoint {
  trade: number;
  p5: number;
  median: number;
  p95: number;
}

export interface DiagnosticRow {
  metric: string;
  value: string;
  floor: string;
  pass: string;
}

export interface TickerSection {
  ticker: string;
  strategy: string;
  regime: string;
  status: string;
  entryPrice: string;
  stopLoss: string;
  stopDistance: string;
  positionSize: string;
  dollarRisk: string;
  atr14: string;
  screenerVerdict: string;
  screeningRows: Record<string, string>[];
  equityBand: EquityPoint[];
  mcSummaryRows: Record<string, string>[];
  mcRiskRows: Record<string, string>[];
  diagnostics: DiagnosticRow[];
  wfSplitsRows: Record<string, string>[];
  backtestRows: Record<string, string>[];
  tradeLogRows: Record<string, string>[];
  backtestReturn: string;
  backtestWinRate: string;
  tradeCount: string;
  maxDrawdown: string;
  sharpe: string;
}

export interface TraderData {
  header: TraderHeader;
  tickers: TickerSection[];
  rawMd: string;
}

export function parseTraderSummary(md: string): TraderData {
  const header = parseTraderHeader(md);
  const tickers = parseTickerSections(md);
  return { header, tickers, rawMd: md };
}

function parseTraderHeader(md: string): TraderHeader {
  return {
    date: md.match(/# Trader Summary — (.+)/)?.[1]?.trim() ?? '',
    tickersPassed: parseInt(md.match(/\*\*(\d+) ticker\(s\) passed/)?.[1] ?? '0'),
    spyBenchmark: md.match(/SPY benchmark: ([^\s|\\n]+)/)?.[1]?.trim() ?? '',
    marketBias: md.match(/\*\*Market bias:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
    favouredSectors: (md.match(/\*\*Favoured sectors:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    avoidSectors: (md.match(/\*\*Avoid sectors:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    keyRisks: md.match(/\*\*Key risks:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
  };
}

function parseTickerSections(md: string): TickerSection[] {
  // Each deep-dive ticker section starts after "---" followed by "## TICKER"
  const parts = md.split(/\n---+\n(?=## [A-Z])/);

  return parts.slice(1).flatMap(part => {
    const hdrMatch = part.match(/^## ([A-Z]+(?:-[A-Z]+)?) — ([^|]+)\| (.+)/m);
    if (!hdrMatch) return [];

    const ticker   = hdrMatch[1].trim();
    const strategy = hdrMatch[2].trim();
    const regime   = hdrMatch[3].trim();

    // Section 1 — Entry Signal
    const sec1 = slice(part, '### 1. Entry Signal', '### 2.');
    const entryTables = findTables(sec1);
    const entryMap: Record<string, string> = {};
    (entryTables[0] ?? []).forEach(r => { entryMap[r['Field'] ?? ''] = r['Value'] ?? ''; });

    // Section 2 — Screening data
    const sec2 = slice(part, '### 2. Why This Ticker Was Selected', '### 3.');
    const screenTables = findTables(sec2);
    const screeningRows = screenTables[0] ?? [];

    // Section 4 — Monte Carlo
    const sec4 = slice(part, '### 4. Monte Carlo Risk Profile', '### 5.');
    const mc4Tables = findTables(sec4);
    // Equity band: the table with 'Trade #' column
    const bandTable    = mc4Tables.find(t => t[0]?.['Trade #'] !== undefined) ?? [];
    // Summary metrics: first table that has P5/Median/P95 columns (not equity band)
    const mcSummaryRows = mc4Tables.find(
      t => t[0] !== undefined && 'Metric' in t[0] && !('Trade #' in t[0])
    ) ?? [];
    // Risk metrics: the table with 'Risk metric' column
    const mcRiskRows    = mc4Tables.find(t => t[0]?.['Risk metric'] !== undefined) ?? [];

    const equityBand: EquityPoint[] = bandTable.map(r => ({
      trade:  parseInt(r['Trade #'] ?? '0'),
      p5:     num(r['P5 ($)'] ?? '0'),
      median: num(r['Median ($)'] ?? '0'),
      p95:    num(r['P95 ($)'] ?? '0'),
    })).filter(p => !isNaN(p.trade) && p.trade >= 0);

    // Section 5 — Diagnostic Scorecard
    const sec5 = slice(part, '### 5. Diagnostic Scorecard', '### 6.');
    const diagTables = findTables(sec5);
    // Main scorecard: the table with Metric/Value/Floor/Pass columns
    const scorecardTable = diagTables.find(t => t[0]?.['Pass'] !== undefined) ?? diagTables[0] ?? [];
    const diagnostics: DiagnosticRow[] = scorecardTable.map(r => ({
      metric: r['Metric'] ?? '',
      value:  r['Value']  ?? '',
      floor:  r['Floor']  ?? '',
      pass:   r['Pass']   ?? '',
    }));
    // Walk-forward splits: the table with IS/OOS columns
    const wfSplitsRows = diagTables.find(t => t[0]?.['IS/OOS'] !== undefined) ?? [];

    // Section 6 — Backtest
    const sec6 = slice(part, '### 6. Backtest Performance');
    const bt6Tables = findTables(sec6);
    // Main metrics: first Metric/Value table
    const backtestRows = bt6Tables.find(t => t[0]?.['Metric'] !== undefined) ?? bt6Tables[0] ?? [];
    // Trade log: table with Entry/Exit columns
    const tradeLogRows = bt6Tables.find(t => t[0]?.['Entry'] !== undefined) ?? [];

    const btMap: Record<string, string> = {};
    backtestRows.forEach(r => { btMap[r['Metric'] ?? ''] = r['Value'] ?? ''; });

    return [{
      ticker,
      strategy,
      regime,
      status:        part.match(/\*\*Status:\*\*\s*(.+)/)?.[1]?.trim() ?? '',
      entryPrice:    entryMap['Entry price']    ?? '',
      stopLoss:      entryMap['Stop loss']      ?? '',
      stopDistance:  entryMap['Stop distance']  ?? '',
      positionSize:  entryMap['Position size']  ?? '',
      dollarRisk:    entryMap['Dollar risk']    ?? '',
      atr14:         entryMap['Current ATR₁₄']  ?? '',
      screenerVerdict: part.match(/\*\*Screener verdict:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
      screeningRows,
      equityBand,
      mcSummaryRows,
      mcRiskRows,
      diagnostics,
      wfSplitsRows,
      backtestRows,
      tradeLogRows,
      backtestReturn: btMap['Net return (after slippage)']
        ? btMap['Net return (after slippage)'].split('|')[0].trim()
        : '',
      backtestWinRate: btMap['Win rate'] ?? '',
      tradeCount:      btMap['Trade count'] ?? '',
      maxDrawdown:     btMap['Max drawdown'] ?? diagnostics.find(d => d.metric === 'Max drawdown')?.value ?? '',
      sharpe:          btMap['Sharpe (RF-adjusted)'] ?? diagnostics.find(d => d.metric.includes('Sharpe'))?.value ?? '',
    }];
  });
}

// ── ReportSummary types & parser ─────────────────────────────────────────────

export interface ReportExecSummary {
  runDate: string;
  newsWindow: string;
  articlesAnalysed: string;
  marketBias: string;
  buyCandidates: string[];
  watchList: string[];
  avoidList: string[];
  topThemes: string;
  keyRisks: string;
}

export interface ReportTicker {
  ticker: string;
  verdict: string;
  reasoning: string;
}

export interface RegimeRow {
  ticker: string;
  regime: string;
  hurst: string;
  atrPct: string;
}

// ── Strategy types ────────────────────────────────────────────────────────────

export interface StrategyAdjParam {
  parameter: string;
  value: string;
}

export interface StrategyTicker {
  ticker: string;
  strategy: string;
  regime: string;
  reasoning: string;
  orderType: string;
  entryConditions: string[];
  positionSizing: string;
  exitRules: string[];
  adjustedParams: StrategyAdjParam[];
  llmAdjustments: string[];
  entryStatus: string;
  entryDetails: string;
}

export interface ReportData {
  exec: ReportExecSummary;
  tickers: ReportTicker[];
  regimes: RegimeRow[];
  strategies: StrategyTicker[];
  macroSummary: string;
  favouredSectors: string[];
  avoidSectors: string[];
  rawMd: string;
}

function parseStrategySection(md: string): StrategyTicker[] {
  const stratSec = slice(md, '## Strategy Parameters');
  if (!stratSec) return [];

  const parts = stratSec.split(/\n(?=### [A-Z])/);

  return parts.flatMap(part => {
    const hdrMatch = part.match(/^### ([A-Z]+(?:[-.][A-Z]+)?) — (.+)/m);
    if (!hdrMatch) return [];

    const ticker   = hdrMatch[1].trim();
    const strategy = hdrMatch[2].trim();
    const regime   = part.match(/\*\*Regime:\*\* ([^\n]+)/)?.[1]?.trim() ?? '';
    const reasoning = part.match(/\*\*Reasoning:\*\* ([^\n]+)/)?.[1]?.trim() ?? '';
    const orderType = part.match(/\*\*Order type:\*\* ([^\n]+)/)?.[1]?.trim() ?? '';

    const entryBlock = slice(part, '**Entry', '**Position sizing');
    const entryConditions = [...entryBlock.matchAll(/^- (.+)$/gm)].map(m => m[1]);

    const positionSizing = part.match(/\*\*Position sizing:\*\* ([^\n]+)/)?.[1]?.trim() ?? '';

    const exitBlock = slice(part, '**Exit rules', '#### Adjusted');
    const exitRules = [...exitBlock.matchAll(/^\d+\. (.+)$/gm)].map(m => strip(m[1]));

    const adjBlock  = slice(part, '#### Adjusted Parameters', '**LLM adjustments');
    const adjTables = findTables(adjBlock);
    const adjustedParams: StrategyAdjParam[] = (adjTables[0] ?? [])
      .map(r => ({ parameter: r['Parameter'] ?? '', value: r['Value'] ?? '' }))
      .filter(p => p.parameter);

    const llmBlock = slice(part, '**LLM adjustments:**', '#### Current Entry Signal');
    const llmAdjustments = [...llmBlock.matchAll(/^- (.+)$/gm)].map(m => m[1]);

    const signalBlock = slice(part, '#### Current Entry Signal');
    const entryStatus  = signalBlock.match(/\*\*Status: (.+?)\*\*/)?.[1]?.trim() ?? '';
    const entryDetails = signalBlock.match(/```\n([\s\S]+?)\n```/)?.[1]?.trim() ?? '';

    return [{ ticker, strategy, regime, reasoning, orderType, entryConditions,
              positionSizing, exitRules, adjustedParams, llmAdjustments, entryStatus, entryDetails }];
  });
}

export function parseReport(md: string): ReportData {
  const execSec = slice(md, '## Executive Summary', '## Macro');

  const exec: ReportExecSummary = {
    runDate:          execSec.match(/\*\*Run date:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
    newsWindow:       execSec.match(/\*\*News window:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
    articlesAnalysed: execSec.match(/\*\*Articles analysed:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
    marketBias:       execSec.match(/\*\*Overall market bias:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
    buyCandidates: (execSec.match(/\*\*Buy candidates[^:]*:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    watchList: (execSec.match(/\*\*Watch[^:]*:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    avoidList: (execSec.match(/\*\*Avoid[^:]*:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    topThemes: execSec.match(/\*\*Top themes:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
    keyRisks:  execSec.match(/\*\*Key risks:\*\* ([^\n]+)/)?.[1]?.trim() ?? '',
  };

  const macroSec = slice(md, '## Macro Environment', '## Shortlisted');
  const macroSummaryMatch = macroSec.match(/^> (.+)/m);

  const tickerSec = slice(md, '## Shortlisted Tickers', '## Regime');
  const tickerTables = findTables(tickerSec);
  const tickers: ReportTicker[] = (tickerTables[0] ?? []).map(r => ({
    ticker:    r['Ticker']   ?? '',
    verdict:   strip(r['Verdict']  ?? ''),
    reasoning: r['Reasoning'] ?? '',
  })).filter(t => t.ticker);

  const regimeSec = slice(md, '## Regime Classification', '## Strategy');
  const regimeTables = findTables(regimeSec);
  const regimes: RegimeRow[] = (regimeTables[0] ?? []).map(r => ({
    ticker:  r['Ticker']    ?? '',
    regime:  r['Regime']    ?? '',
    hurst:   r['Hurst']     ?? '',
    atrPct:  r['ATR/Price'] ?? '',
  })).filter(r => r.ticker);

  return {
    exec,
    tickers,
    regimes,
    strategies: parseStrategySection(md),
    macroSummary: macroSummaryMatch?.[1]?.trim() ?? '',
    favouredSectors: (macroSec.match(/\*\*Favoured sectors:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    avoidSectors: (macroSec.match(/\*\*Avoid sectors:\*\* ([^\n]+)/)?.[1] ?? '')
      .split(',').map(s => s.trim()).filter(Boolean),
    rawMd: md,
  };
}
