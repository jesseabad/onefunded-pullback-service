# Gold ORB Trading Bot — signal-only v0.1

OANDA practice account only. No trading endpoints or order submission.

## Render Web Service
Build: `pip install -r requirements.txt`
Start: `gunicorn app:app --bind 0.0.0.0:$PORT`

Set Render Environment variables:
- `ACCOUNT_MODE=DEMO`
- `OANDA_ACCOUNT_ID` (secret)
- `OANDA_API_KEY` (secret)
- `ORB_INSTRUMENT=XAU_USD`
- `ORB_RISK_REWARD=1.0`
- `ORB_SL_BUFFER_PRICE=0.20` (gold price units, **not** pips)
- `ORB_CANDLE_COUNT=500`

Open `/` for health or `/scan` for current ORB signals. This first release
only calculates JSON levels/events, not graphical candlestick charts, scheduled
polling, historical backtests, Google Sheets logs, or order execution.

**Important:** The original Pine Script's opening range is 09:30–09:45 New York.
This version uses exactly three completed M5 candles and checks signals only
on completed candles. Retests are direction-aware and must occur after a
breakout. This deliberately corrects ambiguities in the original indicator.

OANDA midpoint candle-close entries are *references only*. Actual fills and
1R outcomes will differ after bid/ask spread, slippage, and fees.
