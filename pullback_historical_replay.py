#!/usr/bin/env python3
"""READ-ONLY offline historical replay for OneFunded H1 Pullback.

Two variants: original current-candle retracement and historical maximum
retracement. Uses OANDA practice *GET* candles only; never sends orders.

Dependencies: pip install pandas numpy requests
Environment: OANDA_API_KEY and optionally OANDA_BASE_URL
Usage:
  python pullback_historical_replay.py --from 2024-01-01 --to 2026-10-01 --pairs USD_JPY EUR_USD
  python pullback_historical_replay.py --from 2024-01-01 --to 2026-10-01

WARNING: exploratory replay, not an exact simulation of OANDA fills, bid/ask
history, variable spread, portfolio currency conversions, or Render scan timing.
"""
import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

PAIRS = ('EUR_USD GBP_USD USD_JPY USD_CAD AUD_USD NZD_USD USD_CHF '
         'EUR_JPY GBP_JPY AUD_JPY NZD_JPY CAD_JPY CHF_JPY '
         'EUR_GBP EUR_AUD EUR_CAD EUR_CHF EUR_NZD '
         'GBP_AUD GBP_CAD GBP_CHF GBP_NZD AUD_CAD AUD_CHF AUD_NZD '
         'NZD_CAD NZD_CHF CAD_CHF').split()
BASE = os.getenv('OANDA_BASE_URL', 'https://api-fxpractice.oanda.com')


def download(pair, tf, start, end, cache):
    path = cache / f'{pair}_{tf}_{start}_{end}.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['time'])
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df
    key = os.getenv('OANDA_API_KEY')
    if not key:
        raise RuntimeError('Set OANDA_API_KEY for read-only historical candle requests')
    sess = requests.Session()
    sess.headers['Authorization'] = f'Bearer {key}'
    start_ts = pd.Timestamp(start, tz='UTC') - pd.Timedelta(days=115) # >=500 H4 bars for warmup
    end_ts = pd.Timestamp(end, tz='UTC')
    cursor = start_ts
    out = []
    for page in range(100):
        params = {'granularity': tf, 'price': 'M', 'smooth': 'false',
                  'from': cursor.isoformat().replace('+00:00', 'Z'),
                  'count': 5000, 'includeFirst': 'true' if page == 0 else 'false'}
        for attempt in range(5):
            response = sess.get(f'{BASE}/v3/instruments/{pair}/candles',
                                params=params, timeout=45)
            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            break
        else:
            raise RuntimeError(f'Failed to fetch {pair} {tf}: HTTP {response.status_code}')
        candles = response.json().get('candles', [])
        if not candles:
            break
        last = pd.Timestamp(candles[-1]['time'])
        for c in candles:
            t = pd.Timestamp(c['time'])
            if t >= end_ts:
                break
            if c.get('complete') and c.get('mid'):
                mid = c['mid']
                out.append({'time': t, 'open': float(mid['o']),
                            'high': float(mid['h']), 'low': float(mid['l']),
                            'close': float(mid['c'])})
        if last >= end_ts or last <= cursor or len(candles) < 2:
            break
        cursor = last
    if not out:
        raise RuntimeError(f'No candles for {pair} {tf}')
    df = pd.DataFrame(out).drop_duplicates('time').sort_values('time').reset_index(drop=True)
    cache.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def indicators(df):
    df = df.copy()
    df['ema8'] = df.close.ewm(span=8, adjust=False, min_periods=8).mean()
    df['sma200'] = df.close.rolling(200, min_periods=200).mean()
    prev = df.close.shift(1)
    tr = pd.concat([df.high-df.low, (df.high-prev).abs(),
                    (df.low-prev).abs()], axis=1).max(axis=1)
    # Same Wilder SMA-seeded ATR formula as the worker.
    arr = tr.to_numpy(float)
    atr = np.full(len(df), np.nan)
    if len(df) >= 14:
        atr[13] = arr[:14].mean()
        for i in range(14, len(df)):
            atr[i] = (13 * atr[i-1] + arr[i]) / 14
    df['atr14'] = atr
    return df


def trend(df, i):
    if i < 206:
        return None
    r, old = df.iloc[i], df.iloc[i-5]
    atr = r.atr14
    if not np.isfinite(atr) or atr <= 0:
        return None
    gap = (r.ema8-r.sma200)/atr
    slope = ((r.ema8-old.ema8)-(r.sma200-old.sma200))/atr
    if gap >= .20 and slope >= .03 and r.close > r.sma200:
        return 'BULL'
    if gap <= -.20 and slope <= -.03 and r.close < r.sma200:
        return 'BEAR'
    return None


def pivot_indices(df):
    h = df.high.to_numpy()
    l = df.low.to_numpy()
    highs, lows = [], []
    for i in range(3, len(df)-3):
        if h[i] == max(h[i-3:i+4]) and np.sum(h[i-3:i+4] == h[i]) == 1:
            highs.append(i)
        if l[i] == min(l[i-3:i+4]) and np.sum(l[i-3:i+4] == l[i]) == 1:
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


def recent_impulse(df, i, direction, highs, lows):
    # Only pivots confirmed by the time of this closed candle.
    valid_highs = highs[highs <= i-3]
    valid_lows = lows[lows <= i-3]
    if direction == 'BULL':
        for end in valid_highs[::-1]:
            starts = valid_lows[valid_lows < end]
            if not len(starts):
                continue
            start = int(starts[-1])
            size = float(df.high.iloc[end] - df.low.iloc[start])
            atr = float(df.atr14.iloc[end])
            if atr > 0 and size / atr >= 1.5:
                if i-end > 80:
                    return None
                return (start, int(end), max(df.high.iloc[start], df.high.iloc[end]),
                        min(df.low.iloc[start], df.low.iloc[end]), size)
    else:
        for end in valid_lows[::-1]:
            starts = valid_highs[valid_highs < end]
            if not len(starts):
                continue
            start = int(starts[-1])
            size = float(df.high.iloc[start] - df.low.iloc[end])
            atr = float(df.atr14.iloc[end])
            if atr > 0 and size / atr >= 1.5:
                if i-end > 80:
                    return None
                return (start, int(end), max(df.high.iloc[start], df.high.iloc[end]),
                        min(df.low.iloc[start], df.low.iloc[end]), size)
    return None


def confirmation(df, i, direction):
    a, b = df.iloc[i-1], df.iloc[i]
    rng = b.high-b.low
    body = abs(b.close-b.open)
    if rng <= 0:
        return False
    ratio = body/rng
    upper = b.high-max(b.open,b.close)
    lower = min(b.open,b.close)-b.low
    if direction == 'BULL':
        return bool((b.close>b.open and a.close<a.open and b.open<=a.close and b.close>=a.open)
                    or (b.close>b.open and lower>=body and b.close>=b.low+.65*rng)
                    or (b.close>b.open and ratio>=.60 and b.close>a.high)
                    or (b.close>a.high and ratio>=.45))
    return bool((b.close<b.open and a.close>a.open and b.open>=a.close and b.close<=a.open)
                or (b.close<b.open and upper>=body and b.close<=b.low+.35*rng)
                or (b.close<b.open and ratio>=.60 and b.close<a.low)
                or (b.close<a.low and ratio>=.45))


def signals(pair, h1, h4, start, end):
    h1, h4 = indicators(h1), indicators(h4)
    highs, lows = pivot_indices(h1)
    h4_end = (h4.time + pd.Timedelta(hours=4)).to_numpy(dtype='datetime64[ns]')
    all_signals = []
    first = pd.Timestamp(start, tz='UTC')
    last = pd.Timestamp(end, tz='UTC')
    for i in range(500, len(h1)-1):
        now = h1.time.iloc[i]
        if now < first or now >= last:
            continue
        # A completed H1 bar closes 1h after its timestamp.
        j = np.searchsorted(h4_end, (now + pd.Timedelta(hours=1)).to_datetime64(), side='right')-1
        if j < 206:
            continue
        direction = trend(h1, i)
        if direction is None or direction != trend(h4, j):
            continue
        impulse = recent_impulse(h1, i, direction, highs, lows)
        if not impulse:
            continue
        start_i, end_i, hi, lo, size = impulse
        atr = h1.atr14.iloc[i]
        if not np.isfinite(atr) or atr <= 0 or size <= 0:
            continue
        cur = h1.iloc[i]
        current_dist = max(0, hi-cur.low) if direction=='BULL' else max(0, cur.high-lo)
        extreme = h1.iloc[end_i+1:i+1]
        if extreme.empty:
            continue
        max_dist = max(0, hi-extreme.low.min()) if direction=='BULL' else max(0, extreme.high.max()-lo)
        if not confirmation(h1, i, direction):
            continue
        original_struct = cur.low > lo if direction=='BULL' else cur.high < hi
        original_invalid = cur.close <= lo if direction=='BULL' else cur.close >= hi
        original = (original_struct and not original_invalid and
                    .35 <= current_dist/atr <= 1.5 and 20 <= 100*current_dist/size <= 75)
        candidate = (.35 <= max_dist/atr <= 1.5 and 20 <= 100*max_dist/size <= 75
                     and max_dist < size)
        if not (original or candidate):
            continue
        all_signals.append({'pair': pair, 'time': now.isoformat(), 'i': i,
                            'direction': direction, 'cycle': f'{pair}|{direction}|{start_i}|{end_i}',
                            'original': original, 'max_retrace': candidate,
                            'current_retrace_pct': 100*current_dist/size,
                            'max_retrace_pct': 100*max_dist/size,
                            'impulse_low': lo, 'impulse_high': hi, 'atr': atr})
    return all_signals, h1


def simulate(all_signals, histories, variant, spread_pips, max_open=3, rr=1.5):
    # Per-variant global 3-trade cap and one trade per impulse cycle.
    # Entries at next H1 open; static synthetic spread; SL first if both hit.
    entries = []
    for s in all_signals:
        if not s[variant]:
            continue
        df = histories[s['pair']]
        i = s['i']
        if i+1 >= len(df):
            continue
        pip = .01 if s['pair'].endswith('_JPY') else .0001
        half_spread = spread_pips*pip/2
        bull = s['direction']=='BULL'
        entry = float(df.open.iloc[i+1]) + (half_spread if bull else -half_spread)
        sl = s['impulse_low']-.25*s['atr'] if bull else s['impulse_high']+.25*s['atr']
        dist = (entry-sl) if bull else (sl-entry)
        if dist <= 0:
            continue
        tp = entry+(rr*dist if bull else -rr*dist)
        exit_i, realized = None, None
        for k in range(i+1, len(df)):
            candle = df.iloc[k]
            # Model the bid/ask execution side by applying half-spread to exit quotes.
            lo = candle.low - half_spread if bull else candle.low + half_spread
            hi = candle.high - half_spread if bull else candle.high + half_spread
            exit_open = candle.open - half_spread if bull else candle.open + half_spread
            if bull:
                if exit_open <= sl:
                    exit_i, realized = k, (exit_open-entry)/dist; break
                if exit_open >= tp:
                    exit_i, realized = k, (exit_open-entry)/dist; break
                if lo <= sl:
                    exit_i, realized = k, -1.; break
                if hi >= tp:
                    exit_i, realized = k, rr; break
            else:
                if exit_open >= sl:
                    exit_i, realized = k, (entry-exit_open)/dist; break
                if exit_open <= tp:
                    exit_i, realized = k, (entry-exit_open)/dist; break
                if hi >= sl:
                    exit_i, realized = k, -1.; break
                if lo <= tp:
                    exit_i, realized = k, rr; break
        if exit_i is not None:
            entries.append({**s, 'entry_time': df.time.iloc[i+1],
                            'exit_time': df.time.iloc[exit_i], 'entry': entry,
                            'sl': sl, 'tp': tp, 'realized_r': realized})
    entries.sort(key=lambda x:(x['entry_time'], x['pair']))
    accepted, used_cycles = [], set()
    for e in entries:
        if e['cycle'] in used_cycles:
            continue
        active = sum(x['entry_time'] <= e['entry_time'] <= x['exit_time'] for x in accepted)
        if active >= max_open:
            continue
        accepted.append(e)
        used_cycles.add(e['cycle'])
    return accepted


def report(trades):
    if not trades:
        return {'trades': 0, 'wins': 0, 'win_rate': None, 'net_r': 0,
                'expectancy_r': None, 'max_drawdown_r': None}
    vals = np.array([x['realized_r'] for x in sorted(trades, key=lambda x:x['exit_time'])])
    equity = np.r_[0, np.cumsum(vals)]
    dd = equity - np.maximum.accumulate(equity)
    return {'trades': len(vals), 'wins': int(sum(vals>0)),
            'win_rate': round(float(np.mean(vals>0)*100), 2),
            'net_r': round(float(sum(vals)), 3),
            'expectancy_r': round(float(vals.mean()), 4),
            'max_drawdown_r': round(float(-dd.min()), 3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--from', dest='start', default='2024-01-01')
    p.add_argument('--to', dest='end', default='2026-10-01')
    p.add_argument('--pairs', nargs='+', default=PAIRS)
    p.add_argument('--spread-pips', type=float, default=2.0,
                   help='Constant assumed full spread in pips; run sensitivity tests')
    p.add_argument('--max-open', type=int, default=5)
    p.add_argument('--split', default='2026-01-01', help='Out-of-sample start date')
    p.add_argument('--out', default='pullback_replay_results')
    args = p.parse_args()
    cache = Path('pullback_historical_cache')
    all_signals, histories = [], {}
    for pair in args.pairs:
        if pair not in PAIRS:
            raise ValueError(f'Unknown pair {pair}')
        print(f'Fetching/replaying {pair}...', flush=True)
        h1 = download(pair, 'H1', args.start, args.end, cache)
        h4 = download(pair, 'H4', args.start, args.end, cache)
        ss, hist = signals(pair, h1, h4, args.start, args.end)
        histories[pair] = hist
        all_signals.extend(ss)
        print(f'  H1 bars={len(h1)}, H4 bars={len(h4)}, confirmation signals={len(ss)}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_signals).to_csv(out/'signals_both_variants.csv', index=False)
    rows = []
    split = pd.Timestamp(args.split, tz='UTC')
    for spread in sorted(set([1.0, args.spread_pips, 3.0, 5.0])):
        for variant in ['original', 'max_retrace']:
            trades = simulate(all_signals, histories, variant, spread, args.max_open)
            if spread == args.spread_pips:
                pd.DataFrame(trades).to_csv(out/f'{variant}_trades.csv', index=False)
            for period, sample in [('all', trades),
                                   ('in_sample', [x for x in trades if x['entry_time'] < split]),
                                   ('out_of_sample', [x for x in trades if x['entry_time'] >= split])]:
                metrics = report(sample)
                metrics.update({'variant': variant, 'period': period, 'spread_pips': spread})
                rows.append(metrics)
    pd.DataFrame(rows).to_csv(out/'comparison.csv', index=False)
    print(pd.DataFrame(rows).query("period == 'all'").to_string(index=False))
    print(f'Files saved in {out.resolve()}')

if __name__ == '__main__':
    main()
