"""
OneFunded $5K Pullback Service — V1
READ-ONLY OANDA analysis. It NEVER submits orders.

Pipeline:
H4/H1 trend -> most recent H1 impulse -> Wilder ATR(14) ->
pullback depth -> structure holds -> closed-H1 confirmation ->
fixed-risk trade plan -> Google Sheets log (CONFIRMED only).

Designed to sit beside the existing EMA8/SMA200 cross scanner without modifying it.
"""

import json
import math
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    import gspread
    from google.oauth2.service_account import Credentials
except ImportError:
    gspread = None
    Credentials = None

from tools.config_live_single import API_KEY, ACCOUNT_ID


# ============================================================
# Configuration
# ============================================================

ACCOUNT_MODE = os.environ.get("ACCOUNT_MODE", "DEMO").upper()
BASE_URL = (
    "https://api-fxtrade.oanda.com"
    if ACCOUNT_MODE == "LIVE"
    else "https://api-fxpractice.oanda.com"
)

PAIR_GROUP = os.environ.get("PULLBACK_PAIR_GROUP", "CORE_FOREX").upper()
SCAN_INTERVAL_SECONDS = int(os.environ.get("PULLBACK_SCAN_INTERVAL_SECONDS", "60"))
CANDLE_COUNT = int(os.environ.get("PULLBACK_CANDLE_COUNT", "500"))

EMA_PERIOD = 8
SMA_PERIOD = 200
ATR_PERIOD = 14

# Trend filter: EMA/SMA alignment plus minimum separation and relative slope.
TREND_SLOPE_LOOKBACK = int(os.environ.get("PULLBACK_TREND_SLOPE_LOOKBACK", "5"))
MIN_GAP_ATR = float(os.environ.get("PULLBACK_MIN_GAP_ATR", "0.20"))
MIN_REL_SLOPE_ATR = float(os.environ.get("PULLBACK_MIN_REL_SLOPE_ATR", "0.03"))

# Confirmed pivot settings.
PIVOT_LEFT = int(os.environ.get("PULLBACK_PIVOT_LEFT", "3"))
PIVOT_RIGHT = int(os.environ.get("PULLBACK_PIVOT_RIGHT", "3"))
MIN_IMPULSE_ATR = float(os.environ.get("PULLBACK_MIN_IMPULSE_ATR", "1.50"))
MAX_IMPULSE_AGE_BARS = int(os.environ.get("PULLBACK_MAX_IMPULSE_AGE_BARS", "80"))

# These are deliberately configurable so logged results can tune them.
PULLBACK_MIN_ATR = float(os.environ.get("PULLBACK_MIN_ATR", "0.35"))
PULLBACK_MAX_ATR = float(os.environ.get("PULLBACK_MAX_ATR", "1.50"))
PULLBACK_MIN_RETRACE_PCT = float(os.environ.get("PULLBACK_MIN_RETRACE_PCT", "20"))
PULLBACK_MAX_RETRACE_PCT = float(os.environ.get("PULLBACK_MAX_RETRACE_PCT", "75"))
EXTENDED_ATR = float(os.environ.get("PULLBACK_EXTENDED_ATR", "1.00"))

# Structural stop + ATR buffer.
SL_ATR_BUFFER = float(os.environ.get("PULLBACK_SL_ATR_BUFFER", "0.25"))
RISK_CAD = float(os.environ.get("PULLBACK_RISK_CAD", "25.00"))

# OneFunded analysis sheet.
ONEFUNDED_SHEET_ID = os.environ.get("ONEFUNDED_SHEET_ID", "").strip()
GSHEET_SERVICE_KEY = os.environ.get("GSHEET_SERVICE_KEY", "").strip()
GSHEET_WORKSHEET = os.environ.get("PULLBACK_GSHEET_WORKSHEET", "Pullback_Setups").strip()
GSHEET_LOG_ENABLED = os.environ.get("PULLBACK_GSHEET_LOG_ENABLED", "true").lower() in (
    "1", "true", "yes", "on"
)

PAIR_GROUPS = {
    "MAJORS": [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD",
        "AUD_USD", "NZD_USD", "USD_CHF",
    ],
    "CORE_FOREX": [
        "EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD",
        "AUD_USD", "NZD_USD", "USD_CHF",
        "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "CAD_JPY", "CHF_JPY",
        "EUR_GBP", "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_NZD",
        "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD",
        "AUD_CAD", "AUD_CHF", "AUD_NZD",
        "NZD_CAD", "NZD_CHF", "CAD_CHF",
    ],
}
if PAIR_GROUP not in PAIR_GROUPS:
    raise ValueError(f"Unknown PULLBACK_PAIR_GROUP={PAIR_GROUP!r}")

PAIRS = PAIR_GROUPS[PAIR_GROUP]
TIMEFRAME_MAP = {"H1": "H1", "H4": "H4"}

SESSION = requests.Session()
SESSION.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
})
INSTRUMENT_CACHE = {}

GSHEET_CLIENT = None
GSHEET_TAB = None
LOGGED_SETUP_IDS = set()

GSHEET_COLUMNS = [
    "setup_id", "confirmed_time", "pair", "direction",
    "h4_trend", "h1_trend", "h4_gap_atr", "h1_gap_atr",
    "h4_relative_slope_atr", "h1_relative_slope_atr",
    "impulse_start_time", "impulse_end_time",
    "impulse_high", "impulse_low", "impulse_pips", "impulse_atr",
    "pullback_depth_pips", "pullback_depth_atr", "impulse_retraced_pct",
    "ema8", "sma200", "atr14",
    "confirmation_type", "confirmation_candle_time",
    "confirmation_open", "confirmation_high", "confirmation_low", "confirmation_close",
    "entry", "structure_sl", "atr_buffer", "final_sl", "sl_pips",
    "risk_cad", "units", "lots",
    "tp_1_5r", "tp_1_75r", "tp_2r",
    "spread_pips", "state", "status",
    "result", "max_favorable_R", "max_adverse_R", "exit_R",
]


# ============================================================
# OANDA READ-ONLY API
# ============================================================

def api_get(path, params=None):
    """GET only. This service intentionally has no POST/order function."""
    r = SESSION.get(BASE_URL + path, params=params, timeout=20)
    if not r.ok:
        try:
            body = r.json()
        except Exception:
            body = r.text
        raise RuntimeError(f"OANDA GET {r.status_code}: {body}")
    return r.json()


def get_instrument(instrument):
    if instrument not in INSTRUMENT_CACHE:
        data = api_get(
            f"/v3/accounts/{ACCOUNT_ID}/instruments",
            {"instruments": instrument},
        )
        items = data.get("instruments", [])
        if not items:
            raise RuntimeError(f"Instrument not available: {instrument}")
        INSTRUMENT_CACHE[instrument] = items[0]
    return INSTRUMENT_CACHE[instrument]


def meta(instrument):
    x = get_instrument(instrument)
    return {
        "pip_size": 10 ** int(x["pipLocation"]),
        "display_precision": int(x["displayPrecision"]),
        "units_precision": int(x["tradeUnitsPrecision"]),
        "minimum_trade_size": float(x["minimumTradeSize"]),
    }


def get_candles(instrument, timeframe, count=CANDLE_COUNT):
    data = api_get(
        f"/v3/accounts/{ACCOUNT_ID}/instruments/{instrument}/candles",
        {
            "granularity": TIMEFRAME_MAP[timeframe],
            "count": count,
            "price": "M",
            "smooth": "false",
        },
    )
    rows = []
    for c in data.get("candles", []):
        mid = c.get("mid")
        if not mid:
            continue
        rows.append({
            "time": pd.to_datetime(c["time"], utc=True),
            "open": float(mid["o"]),
            "high": float(mid["h"]),
            "low": float(mid["l"]),
            "close": float(mid["c"]),
            "volume": int(c.get("volume", 0)),
            "complete": bool(c.get("complete", False)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def get_price(instrument):
    data = api_get(
        f"/v3/accounts/{ACCOUNT_ID}/pricing",
        {"instruments": instrument, "includeHomeConversions": "true"},
    )
    prices = data.get("prices", [])
    if not prices:
        raise RuntimeError(f"No price for {instrument}")
    p = prices[0]
    bid = float(p["bids"][0]["price"])
    ask = float(p["asks"][0]["price"])
    return bid, ask, data.get("homeConversions", [])


# ============================================================
# INDICATORS / STRUCTURE
# ============================================================

def wilder_rma(series, period):
    """
    Wilder RMA with SMA seed, then recursive:
        RMA_t = (RMA_(t-1)*(n-1) + x_t) / n
    """
    s = pd.Series(series, dtype="float64")
    out = pd.Series(index=s.index, dtype="float64")
    if len(s) < period:
        return out

    seed_pos = period - 1
    out.iloc[seed_pos] = s.iloc[:period].mean()
    for i in range(seed_pos + 1, len(s)):
        prev = out.iloc[i - 1]
        out.iloc[i] = ((prev * (period - 1)) + s.iloc[i]) / period
    return out


def add_indicators(df):
    df = df.copy()
    df["ema8"] = df["close"].ewm(
        span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD
    ).mean()
    df["sma200"] = df["close"].rolling(
        SMA_PERIOD, min_periods=SMA_PERIOD
    ).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = wilder_rma(tr, ATR_PERIOD)
    return df


def completed(df):
    return df[df["complete"]].copy().reset_index(drop=True)


def trend_context(df):
    if len(df) < SMA_PERIOD + TREND_SLOPE_LOOKBACK + 2:
        return {"direction": "NONE"}

    row = df.iloc[-1]
    old = df.iloc[-1 - TREND_SLOPE_LOOKBACK]
    atr = row["atr14"]

    if pd.isna(atr) or atr <= 0 or pd.isna(row["sma200"]):
        return {"direction": "NONE"}

    gap = row["ema8"] - row["sma200"]
    gap_atr = abs(gap) / atr
    relative_slope = (
        (row["ema8"] - old["ema8"]) -
        (row["sma200"] - old["sma200"])
    ) / atr

    bull = (
        row["ema8"] > row["sma200"]
        and row["close"] > row["sma200"]
        and gap_atr >= MIN_GAP_ATR
        and relative_slope >= MIN_REL_SLOPE_ATR
    )
    bear = (
        row["ema8"] < row["sma200"]
        and row["close"] < row["sma200"]
        and gap_atr >= MIN_GAP_ATR
        and relative_slope <= -MIN_REL_SLOPE_ATR
    )

    return {
        "direction": "BULL" if bull else "BEAR" if bear else "NONE",
        "gap_atr": gap_atr,
        "relative_slope_atr": relative_slope,
        "ema8": row["ema8"],
        "sma200": row["sma200"],
        "atr14": atr,
        "close": row["close"],
    }


def confirmed_pivots(df):
    """
    A pivot is only marked after PIVOT_RIGHT closed candles exist to its right.
    This avoids using a developing/unconfirmed swing.
    """
    highs, lows = [], []
    n = len(df)

    for i in range(PIVOT_LEFT, n - PIVOT_RIGHT):
        window = df.iloc[i - PIVOT_LEFT:i + PIVOT_RIGHT + 1]
        h = df.iloc[i]["high"]
        l = df.iloc[i]["low"]

        if h == window["high"].max() and (window["high"] == h).sum() == 1:
            highs.append(i)
        if l == window["low"].min() and (window["low"] == l).sum() == 1:
            lows.append(i)

    return highs, lows


def find_recent_impulse(df, direction):
    """
    BULL: most recent valid pivot low -> later pivot high.
    BEAR: most recent valid pivot high -> later pivot low.
    Requires impulse >= MIN_IMPULSE_ATR.
    """
    highs, lows = confirmed_pivots(df)
    candidates = []

    if direction == "BULL":
        for hi_idx in reversed(highs):
            prior_lows = [x for x in lows if x < hi_idx]
            if not prior_lows:
                continue
            lo_idx = prior_lows[-1]
            size = df.iloc[hi_idx]["high"] - df.iloc[lo_idx]["low"]
            atr = df.iloc[hi_idx]["atr14"]
            if pd.notna(atr) and atr > 0 and size / atr >= MIN_IMPULSE_ATR:
                candidates.append((lo_idx, hi_idx, size, size / atr))
                break
    else:
        for lo_idx in reversed(lows):
            prior_highs = [x for x in highs if x < lo_idx]
            if not prior_highs:
                continue
            hi_idx = prior_highs[-1]
            size = df.iloc[hi_idx]["high"] - df.iloc[lo_idx]["low"]
            atr = df.iloc[lo_idx]["atr14"]
            if pd.notna(atr) and atr > 0 and size / atr >= MIN_IMPULSE_ATR:
                candidates.append((hi_idx, lo_idx, size, size / atr))
                break

    if not candidates:
        return None

    start_idx, end_idx, size, impulse_atr = candidates[0]
    if (len(df) - 1 - end_idx) > MAX_IMPULSE_AGE_BARS:
        return None

    start = df.iloc[start_idx]
    end = df.iloc[end_idx]
    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_time": start["time"],
        "end_time": end["time"],
        "high": max(start["high"], end["high"]),
        "low": min(start["low"], end["low"]),
        "size": size,
        "impulse_atr": impulse_atr,
    }


def pullback_metrics(df, direction, impulse):
    row = df.iloc[-1]
    atr = row["atr14"]
    if pd.isna(atr) or atr <= 0 or impulse["size"] <= 0:
        return None

    if direction == "BULL":
        # Retracement down from impulse high; use current closed-candle low.
        distance = max(0.0, impulse["high"] - row["low"])
        structure_holds = row["low"] > impulse["low"]
        invalidated = row["close"] <= impulse["low"]
    else:
        # Retracement up from impulse low; use current closed-candle high.
        distance = max(0.0, row["high"] - impulse["low"])
        structure_holds = row["high"] < impulse["high"]
        invalidated = row["close"] >= impulse["high"]

    return {
        "distance": distance,
        "depth_atr": distance / atr,
        "retrace_pct": (distance / impulse["size"]) * 100.0,
        "structure_holds": structure_holds,
        "invalidated": invalidated,
    }


# ============================================================
# CLOSED-H1 CONFIRMATION
# ============================================================

def candle_confirmation(df, direction):
    if len(df) < 3:
        return None

    prev = df.iloc[-2]
    cur = df.iloc[-1]
    rng = cur["high"] - cur["low"]
    body = abs(cur["close"] - cur["open"])
    if rng <= 0:
        return None

    body_ratio = body / rng
    upper_wick = cur["high"] - max(cur["open"], cur["close"])
    lower_wick = min(cur["open"], cur["close"]) - cur["low"]

    if direction == "BULL":
        engulf = (
            cur["close"] > cur["open"]
            and prev["close"] < prev["open"]
            and cur["open"] <= prev["close"]
            and cur["close"] >= prev["open"]
        )
        rejection = (
            cur["close"] > cur["open"]
            and lower_wick >= body
            and cur["close"] >= cur["low"] + 0.65 * rng
        )
        strong_close = (
            cur["close"] > cur["open"]
            and body_ratio >= 0.60
            and cur["close"] > prev["high"]
        )
        structure_break = cur["close"] > prev["high"]

        if engulf:
            return "BULLISH_ENGULFING"
        if rejection:
            return "BULLISH_REJECTION"
        if strong_close:
            return "BULLISH_STRONG_CLOSE"
        if structure_break and body_ratio >= 0.45:
            return "BULLISH_MICRO_BREAK"

    else:
        engulf = (
            cur["close"] < cur["open"]
            and prev["close"] > prev["open"]
            and cur["open"] >= prev["close"]
            and cur["close"] <= prev["open"]
        )
        rejection = (
            cur["close"] < cur["open"]
            and upper_wick >= body
            and cur["close"] <= cur["low"] + 0.35 * rng
        )
        strong_close = (
            cur["close"] < cur["open"]
            and body_ratio >= 0.60
            and cur["close"] < prev["low"]
        )
        structure_break = cur["close"] < prev["low"]

        if engulf:
            return "BEARISH_ENGULFING"
        if rejection:
            return "BEARISH_REJECTION"
        if strong_close:
            return "BEARISH_STRONG_CLOSE"
        if structure_break and body_ratio >= 0.45:
            return "BEARISH_MICRO_BREAK"

    return None


def classify_state(df, direction, impulse, pb):
    if pb["invalidated"] or not pb["structure_holds"]:
        return "INVALIDATED"

    in_depth = PULLBACK_MIN_ATR <= pb["depth_atr"] <= PULLBACK_MAX_ATR
    in_retrace = (
        PULLBACK_MIN_RETRACE_PCT
        <= pb["retrace_pct"]
        <= PULLBACK_MAX_RETRACE_PCT
    )

    if in_depth and in_retrace:
        confirmation = candle_confirmation(df, direction)
        if confirmation:
            return "CONFIRMED"
        return "WAIT_CONFIRMATION"

    row = df.iloc[-1]
    atr = row["atr14"]
    ema = row["ema8"]
    if atr > 0:
        away = (
            (row["close"] - ema) / atr
            if direction == "BULL"
            else (ema - row["close"]) / atr
        )
        if away >= EXTENDED_ATR and pb["depth_atr"] < PULLBACK_MIN_ATR:
            return "EXTENDED"

    if pb["depth_atr"] > 0:
        return "PULLBACK"

    return "TREND"


# ============================================================
# RISK / TRADE PLAN
# ============================================================

def quote_to_cad_loss_factor(instrument, home_conversions):
    quote = instrument.split("_")[1]

    for item in home_conversions:
        if item.get("currency") == quote:
            v = float(item["accountLoss"])
            if v > 0:
                return v

    data = api_get(
        f"/v3/accounts/{ACCOUNT_ID}/pricing",
        {"instruments": instrument, "includeHomeConversions": "true"},
    )
    for item in data.get("homeConversions", []):
        if item.get("currency") == quote:
            v = float(item["accountLoss"])
            if v > 0:
                return v

    raise RuntimeError(f"Cannot convert {quote} loss to account home currency")


def calculate_units(instrument, risk_cad, risk_distance, conversions):
    conv = quote_to_cad_loss_factor(instrument, conversions)
    raw = risk_cad / (risk_distance * conv)
    m = meta(instrument)
    factor = 10 ** m["units_precision"]
    units = math.floor(raw * factor) / factor
    return units if units >= m["minimum_trade_size"] else 0.0


def make_trade_plan(instrument, direction, df, impulse):
    m = meta(instrument)
    row = df.iloc[-1]
    bid, ask, conversions = get_price(instrument)

    # Analysis-only entry reference uses executable side.
    entry = ask if direction == "BULL" else bid
    structure_sl = impulse["low"] if direction == "BULL" else impulse["high"]
    atr_buffer = row["atr14"] * SL_ATR_BUFFER

    final_sl = (
        structure_sl - atr_buffer
        if direction == "BULL"
        else structure_sl + atr_buffer
    )
    risk_distance = abs(entry - final_sl)
    if risk_distance <= 0:
        raise RuntimeError("Invalid risk distance")

    units = calculate_units(instrument, RISK_CAD, risk_distance, conversions)
    sl_pips = risk_distance / m["pip_size"]
    spread_pips = (ask - bid) / m["pip_size"]

    def target(rr):
        return (
            entry + risk_distance * rr
            if direction == "BULL"
            else entry - risk_distance * rr
        )

    p = m["display_precision"]
    return {
        "entry": round(entry, p),
        "structure_sl": round(structure_sl, p),
        "atr_buffer": round(atr_buffer, p),
        "final_sl": round(final_sl, p),
        "sl_pips": sl_pips,
        "risk_cad": RISK_CAD,
        "units": units,
        # OANDA units -> standard FX-lot analytical proxy.
        "lots": units / 100000.0,
        "tp_1_5r": round(target(1.50), p),
        "tp_1_75r": round(target(1.75), p),
        "tp_2r": round(target(2.00), p),
        "spread_pips": spread_pips,
    }


# ============================================================
# GOOGLE SHEETS — CONFIRMED SETUPS ONLY
# ============================================================

def init_google_sheet():
    global GSHEET_CLIENT, GSHEET_TAB, LOGGED_SETUP_IDS

    if not GSHEET_LOG_ENABLED:
        return None
    if GSHEET_TAB is not None:
        return GSHEET_TAB

    if not ONEFUNDED_SHEET_ID or not GSHEET_SERVICE_KEY:
        print("Google logging disabled: missing ONEFUNDED_SHEET_ID or GSHEET_SERVICE_KEY")
        return None
    if gspread is None or Credentials is None:
        print("Google logging disabled: pip install gspread google-auth")
        return None

    info = json.loads(GSHEET_SERVICE_KEY)
    credentials = Credentials.from_service_account_info(
        info,
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ],
    )
    GSHEET_CLIENT = gspread.authorize(credentials)
    spreadsheet = GSHEET_CLIENT.open_by_key(ONEFUNDED_SHEET_ID)

    try:
        GSHEET_TAB = spreadsheet.worksheet(GSHEET_WORKSHEET)
    except gspread.WorksheetNotFound:
        GSHEET_TAB = spreadsheet.add_worksheet(
            title=GSHEET_WORKSHEET,
            rows=2000,
            cols=len(GSHEET_COLUMNS),
        )

    existing_header = GSHEET_TAB.row_values(1)
    if existing_header != GSHEET_COLUMNS:
        GSHEET_TAB.update("A1", [GSHEET_COLUMNS])

    # Load prior IDs so restart/redeploy does not duplicate confirmed setups.
    try:
        values = GSHEET_TAB.col_values(1)
        LOGGED_SETUP_IDS = set(values[1:]) if len(values) > 1 else set()
    except Exception:
        LOGGED_SETUP_IDS = set()

    print(
        f"Google Sheets ready: {GSHEET_WORKSHEET} "
        f"({len(LOGGED_SETUP_IDS)} existing setup IDs)"
    )
    return GSHEET_TAB


def log_confirmed_setup(record):
    tab = init_google_sheet()
    if tab is None:
        return False

    setup_id = record["setup_id"]
    if setup_id in LOGGED_SETUP_IDS:
        return False

    row = [record.get(col, "") for col in GSHEET_COLUMNS]
    tab.append_row(row, value_input_option="USER_ENTERED")
    LOGGED_SETUP_IDS.add(setup_id)
    return True


# ============================================================
# PAIR ANALYSIS
# ============================================================

def fmt(v, digits=2):
    if v is None or pd.isna(v):
        return "--"
    return f"{v:.{digits}f}"


def iso_time(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def analyze_pair(instrument):
    h1 = completed(add_indicators(get_candles(instrument, "H1")))
    h4 = completed(add_indicators(get_candles(instrument, "H4")))

    if len(h1) < 220 or len(h4) < 220:
        return {"pair": instrument, "state": "NO_DATA", "action": "IGNORE"}

    t1 = trend_context(h1)
    t4 = trend_context(h4)

    if t1["direction"] == "NONE" or t4["direction"] == "NONE":
        return {
            "pair": instrument, "trend": "NONE",
            "state": "NO_TREND", "action": "IGNORE",
        }

    if t1["direction"] != t4["direction"]:
        return {
            "pair": instrument,
            "trend": f"H4 {t4['direction']} / H1 {t1['direction']}",
            "state": "TREND_DISAGREE", "action": "IGNORE",
        }

    direction = t1["direction"]
    impulse = find_recent_impulse(h1, direction)
    if not impulse:
        return {
            "pair": instrument, "trend": direction,
            "state": "NO_VALID_IMPULSE", "action": "WAIT",
        }

    pb = pullback_metrics(h1, direction, impulse)
    if not pb:
        return {
            "pair": instrument, "trend": direction,
            "state": "NO_PULLBACK_DATA", "action": "WAIT",
        }

    state = classify_state(h1, direction, impulse, pb)
    action = {
        "NO_TREND": "IGNORE",
        "TREND": "WATCH",
        "EXTENDED": "WAIT",
        "PULLBACK": "WATCH",
        "WAIT_CONFIRMATION": "WAIT",
        "CONFIRMED": "BUY SETUP" if direction == "BULL" else "SELL SETUP",
        "INVALIDATED": "IGNORE",
    }.get(state, "WAIT")

    result = {
        "pair": instrument,
        "trend": direction,
        "state": state,
        "action": action,
        "pullback_atr": pb["depth_atr"],
        "retrace_pct": pb["retrace_pct"],
    }

    if state != "CONFIRMED":
        return result

    confirmation = candle_confirmation(h1, direction)
    plan = make_trade_plan(instrument, direction, h1, impulse)
    cur = h1.iloc[-1]
    m = meta(instrument)
    pip = m["pip_size"]

    setup_id = f"{instrument}|{'BUY' if direction == 'BULL' else 'SELL'}|{iso_time(cur['time'])}"

    record = {
        "setup_id": setup_id,
        "confirmed_time": datetime.now(timezone.utc).isoformat(),
        "pair": instrument,
        "direction": "BUY" if direction == "BULL" else "SELL",
        "h4_trend": t4["direction"],
        "h1_trend": t1["direction"],
        "h4_gap_atr": t4.get("gap_atr", ""),
        "h1_gap_atr": t1.get("gap_atr", ""),
        "h4_relative_slope_atr": t4.get("relative_slope_atr", ""),
        "h1_relative_slope_atr": t1.get("relative_slope_atr", ""),
        "impulse_start_time": iso_time(impulse["start_time"]),
        "impulse_end_time": iso_time(impulse["end_time"]),
        "impulse_high": impulse["high"],
        "impulse_low": impulse["low"],
        "impulse_pips": impulse["size"] / pip,
        "impulse_atr": impulse["impulse_atr"],
        "pullback_depth_pips": pb["distance"] / pip,
        "pullback_depth_atr": pb["depth_atr"],
        "impulse_retraced_pct": pb["retrace_pct"],
        "ema8": cur["ema8"],
        "sma200": cur["sma200"],
        "atr14": cur["atr14"],
        "confirmation_type": confirmation,
        "confirmation_candle_time": iso_time(cur["time"]),
        "confirmation_open": cur["open"],
        "confirmation_high": cur["high"],
        "confirmation_low": cur["low"],
        "confirmation_close": cur["close"],
        **plan,
        "state": "CONFIRMED",
        "status": "NEW",
        "result": "",
        "max_favorable_R": "",
        "max_adverse_R": "",
        "exit_R": "",
    }

    result["record"] = record
    return result


def print_scan(results):
    print("\n" + "=" * 88)
    print(f"OneFunded Pullback Service | {datetime.now(timezone.utc).isoformat()}")
    print("=" * 88)
    print(f"{'PAIR':<10} {'TREND':<8} {'STATE':<22} {'PB ATR':>8} {'RETRACE':>9}  ACTION")
    print("-" * 88)

    for r in results:
        print(
            f"{r['pair']:<10} "
            f"{str(r.get('trend', '--')):<8} "
            f"{r.get('state', '--'):<22} "
            f"{fmt(r.get('pullback_atr')):>8} "
            f"{(fmt(r.get('retrace_pct')) + '%') if r.get('retrace_pct') is not None else '--':>9}  "
            f"{r.get('action', '--')}"
        )

        if r.get("state") == "CONFIRMED":
            x = r["record"]
            print(
                f"  -> {x['direction']} | {x['confirmation_type']} | "
                f"Entry {x['entry']} | SL {x['final_sl']} "
                f"({x['sl_pips']:.1f} pips) | Risk ${x['risk_cad']:.2f} | "
                f"Units {x['units']} | Lots {x['lots']:.3f} | "
                f"TP: 1.5R {x['tp_1_5r']} / 1.75R {x['tp_1_75r']} / 2R {x['tp_2r']}"
            )


def scan_once():
    results = []
    for pair in PAIRS:
        try:
            r = analyze_pair(pair)
            results.append(r)

            if r.get("state") == "CONFIRMED":
                logged = log_confirmed_setup(r["record"])
                if logged:
                    print(f"[LOGGED] {r['record']['setup_id']}")
        except Exception as e:
            results.append({
                "pair": pair,
                "state": "ERROR",
                "action": "CHECK",
                "error": str(e),
            })
            print(f"[ERROR] {pair}: {e}")

    print_scan(results)
    return results


def run():
    print("OneFunded Pullback Service V1")
    print("READ-ONLY: no POST/order function exists in this program.")
    print(f"Account data source: OANDA {ACCOUNT_MODE}")
    print(f"Pairs: {PAIR_GROUP} ({len(PAIRS)})")
    print(f"Risk plan: CAD ${RISK_CAD:.2f} max analytical risk")
    print(f"Google worksheet: {GSHEET_WORKSHEET}")

    if GSHEET_LOG_ENABLED:
        try:
            init_google_sheet()
        except Exception as e:
            print(f"Google Sheets warning: {e}")

    while True:
        try:
            scan_once()
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"[SCAN ERROR] {e}")

        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    run()
