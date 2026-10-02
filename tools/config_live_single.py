import os

ACCOUNT_MODE = os.environ.get("ACCOUNT_MODE", "DEMO").upper()

if ACCOUNT_MODE == "LIVE":
    API_KEY = os.environ["OANDA_API_KEY_LIVE"]
    ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID_LIVE"]
else:
    API_KEY = os.environ["OANDA_API_KEY_DEMO"]
    ACCOUNT_ID = os.environ["OANDA_ACCOUNT_ID_DEMO"]

