"""Closed-M5 ORB breakout/retest detector. Signal-only; never places orders.

Opening range is 09:30-09:45 America/New_York. The breakout predicate
mirrors the provided Pine Script's three-candle comparisons. We track
breakout direction separately and wait for a later retest candle.
"""
from datetime import datetime, time
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')


def analyze(candles, rr=1.0, sl_buffer=0.20):
    if rr <= 0 or sl_buffer < 0:
        raise ValueError('rr must be positive and buffer nonnegative')
    days = {}
    for row in sorted(candles, key=lambda r: r['time']):
        dt = datetime.fromisoformat(row['time'].replace('Z', '+00:00')).astimezone(NY)
        days.setdefault(dt.date().isoformat(), []).append((dt, row))
    sessions = []
    for day, bars in sorted(days.items()):
        opening = [(dt, c) for dt, c in bars if time(9, 30) <= dt.time() < time(9, 45)]
        # Require all three M5 opening candles, including their expected timestamps.
        if [dt.strftime('%H:%M') for dt, _ in opening] != ['09:30', '09:35', '09:40']:
            continue
        high = max(c['high'] for _, c in opening)
        low = min(c['low'] for _, c in opening)
        width = high - low
        if width <= 0:
            continue
        targets = {'long': {str(p): high + width * p / 100 for p in (50, 100, 150)},
                   'short': {str(p): low - width * p / 100 for p in (50, 100, 150)}}
        events = []
        direction = None
        breakout_index = -1
        after = [(dt, c) for dt, c in bars if dt.time() >= time(9, 45)]
        # Include opening candles in the rolling 3-candle test, but signal only after range completion.
        rolling = opening + after
        for i in range(3, len(rolling)):
            dt, cur = rolling[i]
            if dt.time() < time(9, 45):
                continue
            _, prev = rolling[i - 1]
            _, first = rolling[i - 2]
            bullish = (first['low'] < high and first['close'] > high and
                       prev['low'] > high and prev['close'] > high and
                       cur['close'] > prev['low'] and cur['low'] > high)
            bearish = (first['high'] > low and first['close'] < low and
                       prev['high'] < low and prev['close'] < low and
                       cur['close'] < prev['high'] and cur['high'] < low)
            stamp = dt.isoformat()
            if direction is None:
                if bullish or bearish:
                    direction = 'BUY' if bullish else 'SELL'
                    breakout_index = i
                    events.append({'time': stamp, 'type': 'BREAKOUT_WAIT_RETEST', 'side': direction,
                                   'price': cur['close']})
                continue
            if i <= breakout_index:
                continue
            if direction == 'BUY':
                success = prev['close'] > high and cur['low'] <= high and cur['close'] >= high
                failed = prev['close'] > high and cur['close'] < high
            else:
                success = prev['close'] < low and cur['high'] >= low and cur['close'] <= low
                failed = prev['close'] < low and cur['close'] > low
            if success:
                entry = cur['close']  # analytical candle-close reference, not executable bid/ask
                stop = cur['low'] - sl_buffer if direction == 'BUY' else cur['high'] + sl_buffer
                risk = entry - stop if direction == 'BUY' else stop - entry
                if risk > 0:
                    tp = entry + rr * risk if direction == 'BUY' else entry - rr * risk
                    events.append({'time': stamp, 'type': 'RETEST_ENTRY_CANDIDATE',
                                   'side': direction, 'entry_reference': entry,
                                   'sl': stop, 'tp': tp, 'rr': rr})
                direction = None
            elif failed:
                events.append({'time': stamp, 'type': 'FAILED_RETEST', 'side': direction,
                               'price': cur['close']})
                direction = None
        sessions.append({'date': day, 'orb_high': high, 'orb_low': low,
                         'orb_mid': (high + low) / 2, 'range': width,
                         'targets': targets, 'events': events})
    return sessions
