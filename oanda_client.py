"""Read-only OANDA v20 client. Credentials come from Render environment variables."""
import os
import time
import requests

BASE = 'https://api-fxpractice.oanda.com'

class OandaClient:
    def __init__(self):
        if os.getenv('ACCOUNT_MODE', 'DEMO').upper() != 'DEMO':
            raise RuntimeError('This version is DEMO-only')
        self.account_id = os.environ['OANDA_ACCOUNT_ID']
        token = os.environ['OANDA_API_KEY']
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})

    def get(self, path, params=None):
        for attempt in range(3):
            try:
                response = self.session.get(BASE + path, params=params, timeout=20)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError('OANDA request failed')

    def verify_instrument(self, instrument='XAU_USD'):
        data = self.get(f'/v3/accounts/{self.account_id}/instruments', {'instruments': instrument})
        instruments = data.get('instruments', [])
        if not instruments:
            raise RuntimeError(f'{instrument} is unavailable in this demo account')
        return instruments[0]

    def candles(self, instrument='XAU_USD', count=500):
        data = self.get(f'/v3/instruments/{instrument}/candles', {
            'granularity': 'M5', 'count': count, 'price': 'M', 'smooth': 'false'
        })
        rows = []
        for candle in data.get('candles', []):
            if not candle.get('complete') or 'mid' not in candle:
                continue
            mid = candle['mid']
            rows.append({'time': candle['time'], 'open': float(mid['o']),
                         'high': float(mid['h']), 'low': float(mid['l']),
                         'close': float(mid['c']), 'volume': int(candle.get('volume', 0))})
        return rows
