"""
ATHENA-X Telegram Notifier
---------------------------
Reliable, non-critical alerting for the trading engine.

Telegram is deliberately treated as an auxiliary service:
a Telegram failure must NEVER stop Athena, change a trade decision,
or interfere with broker execution.
"""

from __future__ import annotations

import html
import time

import requests

from logger import log


TELEGRAM_API = "https://api.telegram.org"

# Telegram message text limits are comfortably below 4096 characters.
MAX_MESSAGE_LENGTH = 3900
REQUEST_TIMEOUT = 5
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 0.5


class TelegramNotifier:
    def __init__(self):
        self.bot_token = None
        self.chat_id = None
        self.enabled = False
        self.session = requests.Session()

    def setup(self, bot_token, chat_id):
        """Configure Telegram. Invalid credentials disable notifications."""
        self.bot_token = str(bot_token).strip() if bot_token else ""
        self.chat_id = str(chat_id).strip() if chat_id else ""
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            log("Telegram notifier DISABLED - invalid credentials")
            return False

        log("Telegram notifier ENABLED")

        # Connection test is useful, but a failed test must not break startup.
        success = self.send(
            "✅ <b>Athena-X Telegram Connected</b>"
        )

        if not success:
            log(
                "Telegram connection test failed; "
                "notifications remain enabled for future retries."
            )

        return True

    def _build_url(self):
        return (
            f"{TELEGRAM_API}/bot"
            f"{self.bot_token}/sendMessage"
        )

    @staticmethod
    def _escape(value):
        """Safely escape dynamic values before sending HTML."""
        return html.escape(str(value))

    def _split_message(self, message):
        """Split long messages without silently dropping content."""
        text = str(message)

        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]

        return [
            text[index:index + MAX_MESSAGE_LENGTH]
            for index in range(
                0,
                len(text),
                MAX_MESSAGE_LENGTH,
            )
        ]

    def send(self, message):
        """
        Send a Telegram message.

        Returns True only when Telegram confirms success.
        All network/API errors are swallowed and logged.
        """
        if not self.enabled:
            return False

        if not message:
            return False

        chunks = self._split_message(message)
        url = self._build_url()

        for chunk in chunks:
            data = {
                "chat_id": self.chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }

            sent = False

            for attempt in range(MAX_RETRIES + 1):
                try:
                    response = self.session.post(
                        url,
                        json=data,
                        timeout=REQUEST_TIMEOUT,
                    )

                    if response.status_code == 200:
                        try:
                            payload = response.json()
                            if payload.get("ok") is True:
                                sent = True
                                break
                        except ValueError:
                            pass

                    # Retry temporary server/rate-limit failures.
                    if response.status_code in {
                        429,
                        500,
                        502,
                        503,
                        504,
                    } and attempt < MAX_RETRIES:
                        time.sleep(
                            RETRY_DELAY_SECONDS * (attempt + 1)
                        )
                        continue

                    log(
                        "Telegram error: "
                        f"HTTP {response.status_code}"
                    )
                    break

                except requests.RequestException as exc:
                    if attempt < MAX_RETRIES:
                        time.sleep(
                            RETRY_DELAY_SECONDS * (attempt + 1)
                        )
                        continue

                    log(
                        "Telegram network error: "
                        + str(exc)
                    )
                    break

                except Exception as exc:
                    log(
                        "Telegram unexpected error: "
                        + str(exc)
                    )
                    break

            if not sent:
                return False

        return True

    # ========================================================
    # ATHENA MESSAGE TYPES
    # ========================================================

    def send_trade_alert(self, trade_data):
        """Send a formatted trade-entry alert."""
        if not isinstance(trade_data, dict):
            return False

        option_type = self._escape(
            trade_data.get("option_type", "CE")
        )
        strike = self._escape(
            trade_data.get("strike", 0)
        )

        entry = float(trade_data.get("entry", 0) or 0)
        target = float(trade_data.get("target", 0) or 0)
        stop = float(trade_data.get("stop", 0) or 0)
        quantity = int(trade_data.get("quantity", 0) or 0)
        risk = float(trade_data.get("risk", 0) or 0)
        win_prob = float(
            trade_data.get("win_prob", 0.5) or 0.5
        )

        win_prob = max(0.0, min(1.0, win_prob))

        message = (
            "📊 <b>ATHENA-X TRADE ALERT</b>\n\n"
            f"🎯 <b>Option:</b> {option_type} {strike}\n"
            f"💰 <b>Entry:</b> ₹{entry:.2f}\n"
            f"🎯 <b>Target:</b> ₹{target:.2f}\n"
            f"🛑 <b>Stop:</b> ₹{stop:.2f}\n"
            f"📦 <b>Qty:</b> {quantity}\n"
            f"⚠️ <b>Risk:</b> ₹{risk:,.2f}\n"
            f"📈 <b>Win Prob:</b> {win_prob:.0%}"
        )

        if trade_data.get("trailing_stop"):
            message += "\n🔄 <b>Trailing Stop:</b> ON"

        return self.send(message)

    def send_daily_report(self, report_data):
        """Send daily performance summary."""
        if not isinstance(report_data, dict):
            return False

        date = self._escape(report_data.get("date", ""))

        today_pnl = float(
            report_data.get("today_pnl", 0) or 0
        )
        month_pnl = float(
            report_data.get("month_pnl", 0) or 0
        )
        wins = int(report_data.get("wins", 0) or 0)
        losses = int(report_data.get("losses", 0) or 0)
        win_rate = float(
            report_data.get("win_rate", 0) or 0
        )
        progress = float(
            report_data.get("progress", 0) or 0
        )

        message = (
            "📈 <b>ATHENA-X DAILY REPORT</b>\n\n"
            f"📅 <b>Date:</b> {date}\n"
            f"💰 <b>Today P&amp;L:</b> ₹{today_pnl:+,.2f}\n"
            f"📊 <b>Month P&amp;L:</b> ₹{month_pnl:+,.2f}\n"
            f"🏆 <b>Wins:</b> {wins}\n"
            f"📉 <b>Losses:</b> {losses}\n"
            f"📈 <b>Win Rate:</b> {win_rate:.1f}%\n"
            f"🎯 <b>Target Progress:</b> {progress:.1f}%"
        )

        return self.send(message)

    def send_error(self, error_msg):
        """Send a short error notification."""
        text = self._escape(error_msg)[:1000]

        return self.send(
            "❌ <b>ATHENA-X ERROR</b>\n\n"
            + text
        )

    def send_status(self, status_msg):
        """Send a status notification."""
        text = self._escape(status_msg)[:1000]

        return self.send(
            "ℹ️ <b>ATHENA-X STATUS</b>\n\n"
            + text
        )

    def send_ml_prediction(self, prediction):
        """Send the latest XGBoost prediction."""
        if not isinstance(prediction, dict):
            return False

        probability = float(
            prediction.get("win_probability", 0.5)
            or 0.5
        )
        confidence = float(
            prediction.get("confidence", 0)
            or 0
        )

        probability = max(0.0, min(1.0, probability))
        confidence = max(0.0, min(1.0, confidence))

        recommendation = self._escape(
            prediction.get("recommendation", "MAYBE")
        )

        message = (
            "🧠 <b>XGBOOST PREDICTION</b>\n\n"
            f"📊 <b>Win Probability:</b> {probability:.1%}\n"
            f"📈 <b>Confidence:</b> {confidence:.1%}\n"
            f"💡 <b>Recommendation:</b> {recommendation}"
        )

        return self.send(message)


# Singleton instance used by Athena.
telegram = TelegramNotifier()
