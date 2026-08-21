import math
"""
Order management for Athena-X - With Trailing Stop-Loss
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from config import *
from telegram import telegram

IST = ZoneInfo("Asia/Kolkata")

def safe_get(option, keys):
    if not isinstance(option, dict):
        return None
    for key in keys:
        value = option.get(key)
        if value is not None:
            return value
    return None

def get_option_security_id(option):
    val = safe_get(option, ['security_id', 'securityId', 'securityID'])
    return str(val) if val else None

def get_option_ltp(option):
    try:
        val = safe_get(option, ['ltp', 'last_price', 'lastPrice'])
        return float(val) if val else 0
    except Exception:
        return 0

def get_option_delta(option):
    try:
        if not isinstance(option, dict):
            return 0.0

        greeks = option.get("greeks", {})

        if isinstance(greeks, dict):
            val = greeks.get("delta")

            if val is not None:
                return float(val)

        # Compatibility with flat option-chain formats.
        val = safe_get(option, ["delta"])

        return float(val) if val is not None else 0.0

    except (TypeError, ValueError):
        return 0.0

def get_option_oi(option):
    try:
        val = safe_get(option, ['oi', 'open_interest', 'openInterest'])
        return float(val) if val else 0
    except Exception:
        return 0

def get_option_volume(option):
    try:
        val = safe_get(option, ['volume', 'traded_volume', 'tradedVolume'])
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


def get_option_bid_ask(option):
    try:
        bid = safe_get(
            option,
            ['bid', 'bid_price', 'bidPrice', 'top_bid_price', 'topBidPrice'],
        )
        ask = safe_get(
            option,
            ['ask', 'ask_price', 'askPrice', 'top_ask_price', 'topAskPrice'],
        )
        bid = float(bid) if bid is not None else 0.0
        ask = float(ask) if ask is not None else 0.0
        return bid, ask
    except Exception:
        return 0.0, 0.0


def option_liquidity_check(option):
    """Return (ok, diagnostics) without assuming optional Dhan fields exist."""
    ltp = get_option_ltp(option)
    oi = get_option_oi(option)
    volume = get_option_volume(option)
    bid, ask = get_option_bid_ask(option)

    diagnostics = {
        'ltp': ltp,
        'oi': oi,
        'volume': volume,
        'bid': bid,
        'ask': ask,
        'spread_pct': None,
    }

    if ltp <= 0:
        return False, diagnostics

    if oi < MIN_OPTION_OI:
        return False, diagnostics

    if volume < MIN_OPTION_VOLUME:
        return False, diagnostics

    # If Dhan supplies bid/ask, enforce the spread gate.
    # If either is absent, don't reject solely because the optional field
    # is unavailable; the LTP/OI/volume checks still apply.
    if bid > 0 and ask > 0 and ask >= bid:
        spread_pct = (ask - bid) / ltp
        diagnostics['spread_pct'] = spread_pct
        if spread_pct > MAX_OPTION_SPREAD_PCT:
            return False, diagnostics

    return True, diagnostics


def get_lot_size_for_instrument(instrument_name):
    if instrument_name == "NIFTY":
        return NIFTY_LOT_SIZE
    elif instrument_name == "BANKNIFTY":
        return BANKNIFTY_LOT_SIZE
    elif instrument_name == "FINNIFTY":
        return FINNIFTY_LOT_SIZE
    return 25

def calculate_trade_params(
    candidate,
    capital,
    instrument_name="NIFTY",
    probability=None,
    market=None,
    drawdown_pct=0.0,
):
    """Calculate one trade using per-trade risk; daily target is portfolio-level.

    Athena's +1% objective is NOT imposed on every trade's rupee P&L.
    Individual trades use the configured premium stop and a minimum 1:2 R:R.
    The daily portfolio objective is enforced by main.py.
    """
    try:
        entry = float(candidate["entry_price"])
        capital = float(capital)
    except (KeyError, TypeError, ValueError):
        return None

    if entry <= 0 or capital <= 0:
        return None

    # Hard maximum account risk per trade.
    risk_per_unit = entry * OPTION_STOP_PCT
    if risk_per_unit <= 0:
        return None

    max_loss = capital * MAX_RISK_PER_TRADE_PCT
    raw_quantity = max_loss / risk_per_unit
    lot_size = get_lot_size_for_instrument(instrument_name)
    lots = int(raw_quantity / lot_size)

    if lots < 1:
        return None

    quantity = lots * lot_size

    # Athena buys CE/PE options. Both make money when the option premium rises.
    stop = entry * (1 - OPTION_STOP_PCT)
    target_move = max(
        entry * TARGET_PCT,
        abs(entry - stop) * MIN_RISK_REWARD,
    )
    target = entry + target_move

    risk_money = (entry - stop) * quantity
    reward_money = (target - entry) * quantity
    risk_reward = (target - entry) / (entry - stop)

    return {
        "entry": entry,
        "target": target,
        "stop": stop,
        "quantity": quantity,
        "lots": lots,
        "risk": risk_money,
        "reward": reward_money,
        "target_money": reward_money,
        "risk_pct": risk_money / capital,
        "risk_reward": risk_reward,
        "instrument": instrument_name,
    }

def select_best_option(chain_df, market, instrument_name="NIFTY"):
    """
    Select the best option contract from the supplied option chain.

    Selection hierarchy:
        1. Direction determines CE/PE.
        2. ATM and expected-move target define the preferred strike region.
        3. Delta and liquidity are hard filters.
        4. Remaining candidates are ranked using target proximity,
           delta quality, ATM proximity, OI, volume and spread.

    The returned candidate structure is kept compatible with the
    downstream trade/execution pipeline.
    """
    from logger import log

    if chain_df is None or chain_df.empty:
        return None

    if not isinstance(market, dict):
        return None

    try:
        current_price = float(market.get("price", 0))
    except (TypeError, ValueError):
        return None

    if current_price <= 0:
        return None

    direction = str(
        market.get("direction", "")
    ).upper()

    if direction == "BULLISH":
        option_type = "CE"
        direction_sign = 1.0
    elif direction == "BEARISH":
        option_type = "PE"
        direction_sign = -1.0
    else:
        log(
            f"OPTION FILTER | {instrument_name} | "
            f"invalid market direction={direction}"
        )
        return None

    # ------------------------------------------------------------
    # Expected move
    # ------------------------------------------------------------

    try:
        expected_move = float(
            market.get(
                "expected_move",
                current_price * 0.01,
            )
        )
    except (TypeError, ValueError):
        expected_move = current_price * 0.01

    # Prevent a bad/zero ATR from collapsing the target.
    if expected_move <= 0:
        expected_move = current_price * 0.01

    # Do not allow an extreme model move to push the target
    # unrealistically far away from ATM.
    expected_move = min(
        expected_move,
        current_price * 0.02,
    )

    predicted_target = (
        current_price
        + direction_sign * expected_move
    )

    # ------------------------------------------------------------
    # ATM / target strikes
    # ------------------------------------------------------------

    try:
        strikes = sorted(
            float(x)
            for x in chain_df["strike"]
            if x is not None
        )
    except (TypeError, ValueError):
        return None

    if not strikes:
        return None

    atm_strike = min(
        strikes,
        key=lambda x: abs(x - current_price),
    )

    target_strike = min(
        strikes,
        key=lambda x: abs(x - predicted_target),
    )

    # Infer strike spacing from the actual chain instead of relying
    # exclusively on instrument-specific hard-coding.
    unique_strikes = sorted(set(strikes))

    strike_differences = [
        unique_strikes[i + 1] - unique_strikes[i]
        for i in range(len(unique_strikes) - 1)
        if unique_strikes[i + 1] > unique_strikes[i]
    ]

    if strike_differences:
        strike_step = min(strike_differences)
    else:
        strike_step = (
            100.0
            if instrument_name == "BANKNIFTY"
            else 50.0
        )

    if strike_step <= 0:
        strike_step = (
            100.0
            if instrument_name == "BANKNIFTY"
            else 50.0
        )

    # ------------------------------------------------------------
    # Adaptive candidate window
    # ------------------------------------------------------------
    #
    # Instead of rejecting everything outside ATM ± 2 strikes,
    # allow a region based on the expected move.
    #
    # Minimum window:
    #     4 strike steps from ATM
    #
    # Maximum window:
    #     8 strike steps from ATM
    #
    # This keeps the search local without making the selector
    # excessively restrictive.
    # ------------------------------------------------------------

    move_steps = expected_move / strike_step

    search_steps = max(
        4.0,
        min(8.0, move_steps * 1.5),
    )

    max_distance = strike_step * search_steps

    candidates = []

    rejected_liquidity = 0
    rejected_delta = 0
    rejected_distance = 0
    invalid_contracts = 0

    # ------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------

    for idx in range(len(chain_df)):
        row = chain_df.iloc[idx]

        try:
            strike = float(row["strike"])
        except (TypeError, ValueError, KeyError):
            invalid_contracts += 1
            continue

        distance_to_atm = abs(
            strike - atm_strike
        )

        distance_to_target = abs(
            strike - target_strike
        )

        # Adaptive distance gate.
        #
        # Target proximity is allowed to override ATM distance
        # when the expected move is large enough.
        if (
            distance_to_atm > max_distance
            and distance_to_target > max_distance
        ):
            rejected_distance += 1
            continue

        if option_type == "CE":
            option = (
                row.get("ce")
                if hasattr(row, "get")
                else row["ce"]
            )
        else:
            option = (
                row.get("pe")
                if hasattr(row, "get")
                else row["pe"]
            )

        if not isinstance(option, dict):
            invalid_contracts += 1
            continue

        ltp = get_option_ltp(option)
        delta = get_option_delta(option)
        security_id = get_option_security_id(option)
        oi = get_option_oi(option)

        if (
            ltp <= 0
            or delta == 0
            or not security_id
        ):
            invalid_contracts += 1
            continue

        abs_delta = abs(delta)

        # --------------------------------------------------------
        # Delta hard filter
        # --------------------------------------------------------

        if not (
            MIN_DELTA
            <= abs_delta
            <= MAX_DELTA
        ):
            rejected_delta += 1
            continue

        # --------------------------------------------------------
        # Liquidity hard filter
        # --------------------------------------------------------

        liquid, liquidity = option_liquidity_check(
            option
        )

        if not liquid:
            rejected_liquidity += 1
            continue

        # --------------------------------------------------------
        # Normalised scoring
        # --------------------------------------------------------

        distance_target_steps = (
            distance_to_target / strike_step
        )

        distance_atm_steps = (
            distance_to_atm / strike_step
        )

        # Target proximity:
        # 40 points maximum.
        target_score = max(
            0.0,
            40.0
            - distance_target_steps * 5.0,
        )

        # ATM proximity:
        # 15 points maximum.
        atm_score = max(
            0.0,
            15.0
            - distance_atm_steps * 2.5,
        )

        # Delta:
        # Prefer approximately 0.50 delta while allowing
        # the configured 0.35-0.70 range.
        delta_score = max(
            0.0,
            25.0
            - abs(abs_delta - 0.50) * 50.0,
        )

        # OI:
        # Logarithmic scaling prevents huge OI contracts from
        # completely dominating the ranking.
        oi_value = max(
            0.0,
            float(oi),
        )

        oi_score = min(
            10.0,
            math.log10(
                oi_value + 1.0
            ) * 2.0,
        )

        # Volume.
        volume = max(
            0.0,
            float(
                liquidity.get(
                    "volume",
                    get_option_volume(option),
                )
            ),
        )

        volume_score = min(
            5.0,
            math.log10(
                volume + 1.0
            ),
        )

        # Spread.
        spread_pct = liquidity.get(
            "spread_pct"
        )

        spread_penalty = 0.0

        if spread_pct is not None:
            try:
                spread_pct = float(
                    spread_pct
                )
            except (TypeError, ValueError):
                spread_pct = None

        if (
            spread_pct is not None
            and MAX_OPTION_SPREAD_PCT > 0
        ):
            spread_penalty = min(
                10.0,
                (
                    spread_pct
                    / MAX_OPTION_SPREAD_PCT
                ) * 10.0,
            )

        total_score = (
            target_score
            + atm_score
            + delta_score
            + oi_score
            + volume_score
            - spread_penalty
        )

        candidates.append({
            "strike": strike,
            "option_type": option_type,
            "security_id": security_id,
            "entry_price": ltp,
            "delta": abs_delta,
            "score": total_score,
            "oi": oi,
            "volume": volume,
            "bid": liquidity.get("bid"),
            "ask": liquidity.get("ask"),
            "spread_pct": spread_pct,
            "instrument": instrument_name,
            "atm_strike": atm_strike,
            "target_strike": target_strike,
            "expected_move": expected_move,
        })

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    log(
        f"OPTION FILTER | "
        f"{instrument_name} {option_type} | "
        f"rows={len(chain_df)} | "
        f"candidates={len(candidates)} | "
        f"distance_reject={rejected_distance} | "
        f"delta_reject={rejected_delta} | "
        f"liquidity_reject={rejected_liquidity} | "
        f"invalid={invalid_contracts} | "
        f"ATM={atm_strike} | "
        f"TARGET={target_strike} | "
        f"STEP={strike_step} | "
        f"WINDOW={search_steps:.1f}"
    )

    if not candidates:
        return None

    # Highest score wins.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    log(
        "Selected: "
        + instrument_name
        + " "
        + option_type
        + " "
        + str(best["strike"])
        + " | ATM: "
        + str(atm_strike)
        + " | Target: "
        + str(target_strike)
        + " | Expected move: "
        + f"{expected_move:.2f}"
        + " | Delta: "
        + f"{best['delta']:.3f}"
        + " | OI: "
        + str(best["oi"])
        + " | Volume: "
        + str(best["volume"])
        + " | Spread: "
        + (
            "NA"
            if best["spread_pct"] is None
            else f"{best['spread_pct']:.2%}"
        )
        + " | Score: "
        + f"{best['score']:.2f}"
    )

    return best

def execute_trade(dhan, market, candidate, trade, logger, state, ml_engine=None):
    from logger import log
    
    instrument = trade.get('instrument', 'NIFTY')
    
    log("="*60)
    log("ATHENA-X TRADE EXECUTION - " + instrument)
    log("="*60)
    log("Direction: " + market['direction'])
    log("Option: " + candidate['option_type'] + " " + str(candidate['strike']))
    log("Entry: " + str(trade['entry']))
    log("Target: " + str(trade['target']) + " (1%)")
    log("Stop: " + str(trade['stop']) + " (0.5%)")
    log("Trailing Stop: " + ("ENABLED" if TRAILING_STOP_ENABLED else "DISABLED"))
    log("Lots: " + str(trade['lots']) + " | Quantity: " + str(trade['quantity']))
    log("Risk: " + str(trade['risk']))
    log("="*60)
    
    alert_data = {
        'option_type': candidate['option_type'],
        'strike': candidate['strike'],
        'entry': trade['entry'],
        'target': trade['target'],
        'stop': trade['stop'],
        'quantity': trade['quantity'],
        'risk': trade['risk'],
        'win_prob': 0.7,
        'trailing_stop': TRAILING_STOP_ENABLED
    }
    
    if ml_engine and ml_engine.is_trained:
        ml_data = market.get('ml_features', {})
        prediction = ml_engine.predict(ml_data)
        alert_data['win_prob'] = prediction.get('win_probability', 0.7)
        log("ML Prediction: Win Prob " + str(prediction['win_probability']) + 
            " | Confidence " + str(prediction['confidence']) +
            " | " + prediction['recommendation'])
        telegram.send_ml_prediction(prediction)
    
    telegram.send_trade_alert(alert_data)
    
    logger.log_trade({
        'timestamp': datetime.now(IST).isoformat(),
        'symbol': instrument,
        'option_type': candidate['option_type'],
        'strike': candidate['strike'],
        'entry': trade['entry'],
        'exit': 0,
        'quantity': trade['quantity'],
        'pnl': 0,
        'exit_reason': 'OPEN'
    })
    
    state['trades_today'] += 1
    
    # Store trailing stop data if enabled
    trailing_data = {}
    if TRAILING_STOP_ENABLED:
        trailing_data = {
            'best_price': trade['entry'],
            'current_stop': trade['stop'],
            'activation_price': trade['entry'] * (1 + TRAILING_STOP_ACTIVATION) if candidate['option_type'] == 'CE' else trade['entry'] * (1 - TRAILING_STOP_ACTIVATION),
            'direction': market['direction'],
            'option_type': candidate['option_type']
        }
    
    state['active_trade'] = {
        'security_id': candidate['security_id'],
        'quantity': trade['quantity'],
        'entry': trade['entry'],
        'stop': trade['stop'],
        'target': trade['target'],
        'direction': market['direction'],
        'option_type': candidate['option_type'],
        'strike': candidate['strike'],
        'instrument': instrument,
        'rsi': market.get('rsi', 50),
        'ema_cross': 1 if market.get('ema20', 0) > market.get('ema50', 0) else -1,
        'delta': candidate.get('delta', 0.5),
        'score': market.get('score', 70),
        'confidence': market.get('confidence', 0.5),
        'entry_time': datetime.now(IST).isoformat(),
        'trailing_stop': trailing_data if TRAILING_STOP_ENABLED else None,
        'last_update_time': datetime.now(IST).isoformat()
    }
    
    if not LIVE_TRADING:
        log("PAPER MODE — NO REAL ORDER")
        telegram.send_status("PAPER TRADE: " + instrument + " " + candidate['option_type'] + " " + str(candidate['strike']))
        return {'paper': True}
    
    if dhan:
        try:
            response = dhan.place_super_order(
                security_id=candidate['security_id'],
                exchange_segment="NSE_FNO",
                transaction_type="BUY",
                quantity=trade['quantity'],
                order_type="LIMIT",
                price=round(trade['entry'], 2),
                target_price=round(trade['target'], 2),
                stop_loss_price=round(trade['stop'], 2),
                trailing_jump=0
            )
            log("Order placed: " + str(response))
            telegram.send_status("LIVE ORDER PLACED: " + instrument + " " + str(response))
            return response
        except Exception as e:
            log("Order error: " + str(e))
            telegram.send_error("Order failed: " + str(e))
    
    return None
# ============================================================
# TRAILING STOP / EXIT MANAGEMENT
# ============================================================

def update_trailing_stop(active, current_price):
    """
    Update trailing stop using the traded option LTP.

    The underlying/index price is never used here.
    """
    if not active or not TRAILING_STOP_ENABLED:
        return

    try:
        current_price = float(current_price)
        entry = float(active["entry"])
        stop = float(active["stop"])
    except (KeyError, TypeError, ValueError):
        return

    if current_price <= 0 or entry <= 0:
        return

    trailing_data = active.get("trailing_stop")

    if not isinstance(trailing_data, dict):
        return

    direction = str(
        active.get("direction", "BULLISH")
    ).upper()

    best_price = float(
        trailing_data.get("best_price", entry)
    )

    if direction == "BULLISH":

        profit_pct = (
            (current_price - entry) / entry
        )

        if (
            profit_pct >= TRAILING_STOP_ACTIVATION
            and current_price > best_price
        ):
            best_price = current_price

            new_stop = current_price * (
                1 - TRAILING_STOP_DISTANCE
            )

            if new_stop > stop:
                active["stop"] = new_stop

                log(
                    f"Trailing Stop Updated: "
                    f"{new_stop:.2f} "
                    f"(Profit: {profit_pct:.2%})"
                )

                try:
                    telegram.send_status(
                        f"Trailing Stop moved to "
                        f"{new_stop:.2f}"
                    )
                except Exception:
                    pass

    else:

        profit_pct = (
            (entry - current_price) / entry
        )

        if (
            profit_pct >= TRAILING_STOP_ACTIVATION
            and current_price < best_price
        ):
            best_price = current_price

            new_stop = current_price * (
                1 + TRAILING_STOP_DISTANCE
            )

            if new_stop < stop:
                active["stop"] = new_stop

                log(
                    f"Trailing Stop Updated: "
                    f"{new_stop:.2f} "
                    f"(Profit: {profit_pct:.2%})"
                )

                try:
                    telegram.send_status(
                        f"Trailing Stop moved to "
                        f"{new_stop:.2f}"
                    )
                except Exception:
                    pass

    trailing_data["best_price"] = best_price
    trailing_data["current_stop"] = active.get(
        "stop",
        stop,
    )

    active["last_update_time"] = (
        datetime.now(IST).isoformat()
    )


def check_exit_levels(active, current_price):
    """
    Check target and stop using the traded option LTP.

    Returns:
        TAKE_PROFIT
        STOP_LOSS
        TIME_EXIT
        None
    """

    if not active:
        return None

    try:
        price = float(current_price)
        target = float(active["target"])
        stop = float(active["stop"])
    except (KeyError, TypeError, ValueError):
        return None

    if price <= 0:
        return None

    # Athena buys option premium.
    # Therefore target is above entry and stop is below entry.
    if price >= target:
        return "TAKE_PROFIT"

    if price <= stop:
        return "STOP_LOSS"

    # Optional maximum holding time.
    entry_time = active.get("entry_time")

    max_hold_minutes = active.get(
        "max_hold_minutes",
        globals().get("MAX_HOLD_MINUTES"),
    )

    if max_hold_minutes and entry_time:

        try:

            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(
                    entry_time
                )

            held_minutes = (
                datetime.now(IST) - entry_time
            ).total_seconds() / 60.0

            if held_minutes >= float(
                max_hold_minutes
            ):
                return "TIME_EXIT"

        except (
            TypeError,
            ValueError,
        ):
            pass

    return None