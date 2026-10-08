"""Render web service: read-only scanner, refreshes on each /scan request."""
import os
from flask import Flask, jsonify
from oanda_client import OandaClient
from orb_strategy import analyze

app = Flask(__name__)

@app.get('/')
def health():
    return jsonify({'service': 'gold-orb-trading-bot', 'status': 'ready', 'execution': 'DISABLED'})

@app.get('/scan')
def scan():
    try:
        client = OandaClient()
        instrument = os.getenv('ORB_INSTRUMENT', 'XAU_USD')
        meta = client.verify_instrument(instrument)
        candles = client.candles(instrument, int(os.getenv('ORB_CANDLE_COUNT', '500')))
        sessions = analyze(candles, rr=float(os.getenv('ORB_RISK_REWARD', '1.0')),
                           sl_buffer=float(os.getenv('ORB_SL_BUFFER_PRICE', '0.20')))
        return jsonify({'instrument': instrument, 'display_precision': meta.get('displayPrecision'),
                        'sessions': sessions[-5:], 'execution': 'DISABLED'})
    except Exception as exc:
        app.logger.exception('Scan failed')
        return jsonify({'error': str(exc)}), 500
