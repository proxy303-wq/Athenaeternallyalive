"""Safe Athena connection diagnostics. Never prints credential values."""

import os
import requests

from config import CLIENT_ID, ACCESS_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

print("=== ATHENA CONNECTION DIAGNOSTICS ===")
print(f"DHAN_CLIENT_ID present : {'YES' if CLIENT_ID else 'NO'}")
print(f"DHAN_ACCESS_TOKEN present: {'YES' if ACCESS_TOKEN else 'NO'}")
print(f"TELEGRAM_BOT_TOKEN present: {'YES' if TELEGRAM_BOT_TOKEN else 'NO'}")
print(f"TELEGRAM_CHAT_ID present: {'YES' if TELEGRAM_CHAT_ID else 'NO'}")

if CLIENT_ID and ACCESS_TOKEN:
    try:
        from dhanhq import DhanContext, dhanhq
        ctx = DhanContext(CLIENT_ID, ACCESS_TOKEN)
        dhan = dhanhq(ctx)
        profile = dhan.get_profile()
        print("DHAN API profile: OK")
        print("Dhan profile response received.")
        print("Do not paste credential values into chat.")
    except Exception as exc:
        print(f"DHAN API profile: FAILED ({type(exc).__name__}: {exc})")
else:
    print("DHAN API profile: SKIPPED (credentials missing)")

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe",
            timeout=10,
        )
        data = response.json()
        if response.ok and data.get("ok"):
            print("TELEGRAM BOT API: OK")
            print(f"Telegram bot username: @{data['result'].get('username', 'unknown')}")
        else:
            print(f"TELEGRAM BOT API: FAILED ({data})")
    except Exception as exc:
        print(f"TELEGRAM BOT API: FAILED ({type(exc).__name__}: {exc})")
else:
    print("TELEGRAM BOT API: SKIPPED (credentials missing)")

print("=== END DIAGNOSTICS ===")
