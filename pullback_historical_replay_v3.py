#!/usr/bin/env python3
"""READ-ONLY historical bid/ask replay for OneFunded H1 Pullback.

Three variants: original current-candle retracement, historical maximum
retracement (alternative entry logic), and strict_original_filter (original
entry rules PLUS maximum historical retracement constraints). Uses OANDA practice
GET candles only; never sends orders. Entry/exit quotes use historical bid/ask
H1 candles rather than an assumed fixed spread.

Dependencies: pip install pandas numpy requests
Environment: OANDA_API_KEY and optionally OANDA_BASE_URL
Usage:
  python pullback_historical_replay_v3.py --from 2024-01-01 --to 2026-10-01 --pairs USD_JPY EUR_USD
  python pullback_historical_replay_v3.py --from 2024-01-01 --to 2026-10-01

WARNING: exploratory replay, not an exact simulation of tick-by-tick fills,
slippage, portfolio currency conversions, swap/financing, or Render scan timing.
H1 bid/ask OHLC cannot resolve whether a stop or target was hit first within
the same candle; the model conservatively assumes stop-first.
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
        # strict_original_filter cannot create a signal absent from the original.
        # In contrast, max_retrace alone is an alternative entry model.
        accepted_signal = (s['original'] and s['max_retrace']) if variant == 'strict_original_filter' else s[variant]
        if not accepted_signal:
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



def download_bidask(pair, start, end, cache):
    """Read-only historical OANDA H1 bid/ask candles; cached separately from mid."""
    path = cache / f'{pair}_H1_BA_{start}_{end}.csv'
    if path.exists():
        df = pd.read_csv(path, parse_dates=['time'])
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df
    key = os.getenv('OANDA_API_KEY')
    if not key:
        raise RuntimeError('Set OANDA_API_KEY in this Terminal session (do not share it)')
    sess = requests.Session()
    sess.headers['Authorization'] = f'Bearer {key}'
    cursor = pd.Timestamp(start, tz='UTC') - pd.Timedelta(days=115)
    end_ts = pd.Timestamp(end, tz='UTC')
    out = []
    for page in range(100):
        params = {'granularity': 'H1', 'price': 'BA', 'smooth': 'false',
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
            raise RuntimeError(f'Unable to fetch {pair} bid/ask: HTTP {response.status_code}')
        candles = response.json().get('candles', [])
        if not candles:
            break
        last = pd.Timestamp(candles[-1]['time'])
        for c in candles:
            t = pd.Timestamp(c['time'])
            if t >= end_ts:
                break
            if not c.get('complete') or not c.get('bid') or not c.get('ask'):
                continue
            bid, ask = c['bid'], c['ask']
            out.append({'time': t,
                        **{f'bid_{k}': float(bid[v]) for k,v in
                           [('open','o'),('high','h'),('low','l'),('close','c')]},
                        **{f'ask_{k}': float(ask[v]) for k,v in
                           [('open','o'),('high','h'),('low','l'),('close','c')]}})
        if last >= end_ts or last <= cursor or len(candles) < 2:
            break
        cursor = last
    if not out:
        raise RuntimeError(f'No OANDA bid/ask candles for {pair}')
    df = pd.DataFrame(out).drop_duplicates('time').sort_values('time').reset_index(drop=True)
    cache.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def simulate_bidask(all_signals, histories, bidasks, variant, max_open=5,
                    rr=1.5, commission_roundtrip_pips=0.0):
    """Use next-H1 bid/ask open for entries and exit-side bid/ask OHLC for exits."""
    candidates = []
    for s in all_signals:
        accepted_signal = (s['original'] and s['max_retrace']) if variant == 'strict_original_filter' else s[variant]
        if not accepted_signal:
            continue
        pair = s['pair']
        df = histories[pair]
        ba = bidasks[pair]
        i = s['i']
        if i+1 >= len(df):
            continue
        start_time = df.time.iloc[i+1]
        # A missing BA bar must NOT be filled from a different hour.
        if start_time not in ba.index:
            continue
        quote = ba.loc[start_time]
        if isinstance(quote, pd.DataFrame):
            raise ValueError(f'Duplicate bid/ask timestamp for {pair} {start_time}')
        pip = .01 if pair.endswith('_JPY') else .0001
        bull = s['direction'] == 'BULL'
        entry = float(quote.ask_open if bull else quote.bid_open)
        spread = (float(quote.ask_open) - float(quote.bid_open))/pip
        if spread < 0:
            raise ValueError(f'Negative historical spread {pair} {start_time}')
        sl = s['impulse_low']-.25*s['atr'] if bull else s['impulse_high']+.25*s['atr']
        dist = entry-sl if bull else sl-entry
        if dist <= 0:
            continue
        tp = entry + rr*dist if bull else entry - rr*dist
        exit_i, realized = None, None
        # OANDA BA OHLC does not give intra-candle event ordering. SL-first.
        for k in range(i+1, len(df)):
            t = df.time.iloc[k]
            if t not in ba.index:
                continue
            q = ba.loc[t]
            if bull:
                exit_open = float(q.bid_open)
                if exit_open <= sl or exit_open >= tp:
                    exit_i, realized = k, (exit_open-entry)/dist
                    break
                if float(q.bid_low) <= sl:
                    exit_i, realized = k, -1.0
                    break
                if float(q.bid_high) >= tp:
                    exit_i, realized = k, rr
                    break
            else:
                exit_open = float(q.ask_open)
                if exit_open >= sl or exit_open <= tp:
                    exit_i, realized = k, (entry-exit_open)/dist
                    break
                if float(q.ask_high) >= sl:
                    exit_i, realized = k, -1.0
                    break
                if float(q.ask_low) <= tp:
                    exit_i, realized = k, rr
                    break
        if exit_i is None:
            continue
        realized -= commission_roundtrip_pips*pip/dist
        candidates.append({**s, 'entry_time': start_time,
                           'exit_time': df.time.iloc[exit_i],
                           'entry': entry, 'sl': sl, 'tp': tp,
                           'entry_spread_pips': spread,
                           'stop_pips': dist/pip,
                           'commission_roundtrip_pips': commission_roundtrip_pips,
                           'realized_r': realized})
    candidates.sort(key=lambda x: (x['entry_time'], x['pair']))
    accepted, used_cycles = [], set()
    for e in candidates:
        if e['cycle'] in used_cycles:
            continue
        active = sum(x['entry_time'] <= e['entry_time'] <= x['exit_time'] for x in accepted)
        if active >= max_open:
            continue
        accepted.append(e)
        used_cycles.add(e['cycle'])
    return accepted


def by_group(trades, group):
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    if group == 'year':
        df['group'] = pd.to_datetime(df.entry_time, utc=True).dt.year.astype(str)
    elif group == 'pair':
        df['group'] = df.pair
    else:
        raise ValueError(group)
    rows = []
    for g, d in df.groupby('group'):
        r = report(d.to_dict('records'))
        r.update({'group': g, 'avg_entry_spread_pips': round(d.entry_spread_pips.mean(),3),
                  'median_entry_spread_pips': round(d.entry_spread_pips.median(),3)})
        rows.append(r)
    return pd.DataFrame(rows)


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
    p = argparse.ArgumentParser(description='Read-only H1 Pullback replay using historical OANDA bid/ask candles')
    p.add_argument('--from', dest='start', default='2024-01-01')
    p.add_argument('--to', dest='end', default='2026-10-01')
    p.add_argument('--pairs', nargs='+', default=PAIRS)
    p.add_argument('--max-open', type=int, default=5)
    p.add_argument('--split', default='2026-01-01')
    p.add_argument('--commission-roundtrip-pips', type=float, default=0.0,
                   help='Optional additional commission equivalent in pips (round-trip); default 0')
    p.add_argument('--out', default='pullback_replay_results_v3')
    args = p.parse_args()
    if args.commission_roundtrip_pips < 0:
        p.error('--commission-roundtrip-pips cannot be negative')
    cache = Path('pullback_historical_cache')
    all_signals, histories, bidasks = [], {}, {}
    for pair in args.pairs:
        if pair not in PAIRS:
            raise ValueError(f'Unknown pair {pair}')
        print(f'Fetching/replaying {pair}...', flush=True)
        h1 = download(pair, 'H1', args.start, args.end, cache)
        h4 = download(pair, 'H4', args.start, args.end, cache)
        ba = download_bidask(pair, args.start, args.end, cache)
        ss, hist = signals(pair, h1, h4, args.start, args.end)
        histories[pair] = hist
        bidasks[pair] = ba.set_index('time', verify_integrity=True)
        all_signals.extend(ss)
        print(f'  H1 mid={len(h1)}, H4 mid={len(h4)}, H1 bid/ask={len(ba)}, signals={len(ss)}', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_signals).to_csv(out/'signals_both_variants.csv', index=False)
    rows, pairs, years = [], [], []
    split = pd.Timestamp(args.split, tz='UTC')
    for variant in ['original', 'max_retrace', 'strict_original_filter']:
        trades = simulate_bidask(all_signals, histories, bidasks, variant, args.max_open,
                                 commission_roundtrip_pips=args.commission_roundtrip_pips)
        pd.DataFrame(trades).to_csv(out/f'{variant}_trades.csv', index=False)
        for period, sample in [('all', trades),
                               ('in_sample', [x for x in trades if x['entry_time'] < split]),
                               ('out_of_sample', [x for x in trades if x['entry_time'] >= split])]:
            metrics = report(sample)
            metrics.update({'variant': variant, 'period': period, 'cost_model': 'historical_bid_ask',
                            'commission_roundtrip_pips': args.commission_roundtrip_pips})
            rows.append(metrics)
        for group_name, destination in [('pair', pairs), ('year', years)]:
            table = by_group(trades, group_name)
            if not table.empty:
                table['variant'] = variant
                destination.append(table)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out/'comparison_historical_bidask.csv', index=False)
    if pairs:
        pd.concat(pairs, ignore_index=True).to_csv(out/'pair_performance.csv', index=False)
    if years:
        pd.concat(years, ignore_index=True).to_csv(out/'yearly_performance.csv', index=False)
    print('\nFULL PERIOD — HISTORICAL BID/ASK (not fixed spread):')
    print(comparison.query("period == 'all'").to_string(index=False))
    print('\nOUT OF SAMPLE:')
    print(comparison.query("period == 'out_of_sample'").to_string(index=False))
    print(f'\nFiles saved in {out.resolve()}')

if __name__ == '__main__':
    main()
