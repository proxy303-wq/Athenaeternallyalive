"""
ATHENA-X Quantitative Risk Engine
---------------------------------
Mathematical layer between XGBoost and order execution.

Responsibilities:
- market-regime classification
- expected-value calculation
- fractional Kelly sizing
- drawdown risk throttling
- hard exposure/risk validation

This module never places orders.
"""

from __future__ import annotations

import math

from config import (
    DRAWDOWN_FULL_RISK_PCT,
    DRAWDOWN_STOP_PCT,
    EV_MIN_PER_RISK,
    HIGH_VOL_ATR_PCT,
    KELLY_FRACTION,
    LOW_VOL_ATR_PCT,
    MAX_DRAWDOWN_RISK_REDUCTION,
    MAX_KELLY_RISK_PCT,
    MAX_VOLATILITY,
    MIN_VOLATILITY,
    RANGE_ADX_THRESHOLD,
    SLIPPAGE_PCT,
    TRANSACTION_COST_PCT,
    TREND_ADX_THRESHOLD,
    TREND_EMA_SPREAD_PCT,
)


def _finite(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, _finite(value)))


def classify_regime(market):
    """
    Classify the underlying market into a small number of robust regimes.

    The classification deliberately uses measurements already produced by
    Athena instead of adding a large collection of new indicators.
    """
    if not isinstance(market, dict):
        return "UNKNOWN"

    adx = _finite(market.get("adx"))
    atr_percent = _finite(
        market.get(
            "atr_percent",
            (
                _finite(market.get("atr"))
                / _finite(market.get("price"), 1.0)
                * 100.0
            ),
        )
    )

    price = _finite(market.get("price"))
    ema20 = _finite(market.get("ema20"))
    ema50 = _finite(market.get("ema50"))

    ema_spread = (
        abs(ema20 - ema50) / price * 100.0
        if price > 0
        else 0.0
    )

    if atr_percent >= HIGH_VOL_ATR_PCT:
        return "HIGH_VOL"

    if atr_percent <= LOW_VOL_ATR_PCT and adx < TREND_ADX_THRESHOLD:
        return "LOW_VOL_RANGE"

    if (
        adx >= TREND_ADX_THRESHOLD
        and ema_spread >= TREND_EMA_SPREAD_PCT
    ):
        return "TRENDING"

    if adx <= RANGE_ADX_THRESHOLD:
        return "RANGING"

    return "NORMAL"


def drawdown_multiplier(drawdown_pct):
    """
    Convert current equity drawdown into a risk multiplier.

    0% to 2% drawdown: full risk.
    2% to 8%: linearly reduce risk.
    >=8%: no new risk.
    """
    drawdown = max(0.0, _finite(drawdown_pct))

    if drawdown >= DRAWDOWN_STOP_PCT:
        return 0.0

    if drawdown <= DRAWDOWN_FULL_RISK_PCT:
        return 1.0

    span = DRAWDOWN_STOP_PCT - DRAWDOWN_FULL_RISK_PCT
    progress = (
        (drawdown - DRAWDOWN_FULL_RISK_PCT) / span
        if span > 0
        else 1.0
    )

    reduction = min(
        MAX_DRAWDOWN_RISK_REDUCTION,
        progress * MAX_DRAWDOWN_RISK_REDUCTION,
    )

    return max(0.0, 1.0 - reduction)


def regime_risk_multiplier(regime):
    """
    Reduce risk in adverse regimes rather than rejecting every setup.

    The final hard risk ceiling still applies.
    """
    return {
        "TRENDING": 1.00,
        "NORMAL": 0.90,
        "RANGING": 0.65,
        "LOW_VOL_RANGE": 0.50,
        "HIGH_VOL": 0.50,
        "UNKNOWN": 0.00,
    }.get(regime, 0.50)


def expected_value(probability, reward, risk, costs=0.0):
    """
    Expected monetary value:

        EV = P(win)*Reward - P(loss)*Risk - Costs

    Returns a monetary value.
    """
    p = clamp(probability)
    reward = max(0.0, _finite(reward))
    risk = max(0.0, _finite(risk))
    costs = max(0.0, _finite(costs))

    return p * reward - (1.0 - p) * risk - costs


def expected_value_per_risk(probability, reward, risk, costs=0.0):
    """Normalize EV by risk so trades are comparable."""
    risk = max(0.0, _finite(risk))

    if risk <= 0:
        return float("-inf")

    return expected_value(probability, reward, risk, costs) / risk


def kelly_fraction(probability, reward, risk):
    """
    Calculate the theoretical Kelly fraction.

    This is never used raw. Athena applies KELLY_FRACTION and a hard cap.
    """
    p = clamp(probability)
    reward = max(0.0, _finite(reward))
    risk = max(0.0, _finite(risk))

    if reward <= 0 or risk <= 0:
        return 0.0

    b = reward / risk
    q = 1.0 - p

    if b <= 0:
        return 0.0

    raw = ((b * p) - q) / b
    return max(0.0, min(1.0, raw))


def effective_probability(
    ml_probability,
    ml_trained,
    market_confidence,
):
    """
    Produce the probability used by the quantitative layer.

    Trained XGBoost gets priority.
    Before ML is trained, market confidence is used conservatively and
    capped so the system cannot become overconfident without a model.
    """
    if ml_trained:
        return clamp(ml_probability, 0.01, 0.99)

    confidence = clamp(market_confidence, 0.50, 0.60)
    return confidence


def evaluate_trade(
    probability,
    reward,
    risk,
    market,
    drawdown_pct=0.0,
    estimated_costs=0.0,
):
    """
    Evaluate a candidate trade without executing it.

    Returns a complete mathematical decision record.
    """
    regime = classify_regime(market)

    ev = expected_value(
        probability,
        reward,
        risk,
        estimated_costs,
    )

    ev_per_risk = expected_value_per_risk(
        probability,
        reward,
        risk,
        estimated_costs,
    )

    kelly = kelly_fraction(
        probability,
        reward,
        risk,
    )

    drawdown_mult = drawdown_multiplier(drawdown_pct)
    regime_mult = regime_risk_multiplier(regime)

    total_risk_multiplier = drawdown_mult * regime_mult

    approved = True
    reason = "APPROVED"

    if regime == "UNKNOWN":
        approved = False
        reason = "UNKNOWN_REGIME"

    elif drawdown_mult <= 0:
        approved = False
        reason = "DRAWDOWN_STOP"

    elif ev_per_risk < EV_MIN_PER_RISK:
        approved = False
        reason = "NEGATIVE_OR_WEAK_EV"

    elif risk <= 0 or reward <= 0:
        approved = False
        reason = "INVALID_REWARD_RISK"

    return {
        "approved": approved,
        "reason": reason,
        "regime": regime,
        "probability": clamp(probability),
        "reward": max(0.0, _finite(reward)),
        "risk": max(0.0, _finite(risk)),
        "costs": max(0.0, _finite(estimated_costs)),
        "expected_value": ev,
        "expected_value_per_risk": ev_per_risk,
        "kelly_fraction": kelly,
        "fractional_kelly": kelly * KELLY_FRACTION,
        "drawdown_multiplier": drawdown_mult,
        "regime_multiplier": regime_mult,
        "risk_multiplier": total_risk_multiplier,
        "max_kelly_risk_pct": MAX_KELLY_RISK_PCT,
    }


def calculate_position_size(
    capital,
    entry,
    stop,
    reward,
    probability,
    lot_size,
    regime,
    drawdown_pct=0.0,
):
    """
    Calculate quantity using hard risk, regime throttling and fractional Kelly.

    Returns quantity/lot information plus the actual risk.
    """
    capital = max(0.0, _finite(capital))
    entry = max(0.0, _finite(entry))
    stop = max(0.0, _finite(stop))
    reward = max(0.0, _finite(reward))
    lot_size = max(1, int(_finite(lot_size, 1)))

    risk_per_unit = entry - stop

    if (
        capital <= 0
        or entry <= 0
        or risk_per_unit <= 0
        or reward <= 0
    ):
        return {
            "quantity": 0,
            "lots": 0,
            "risk_per_unit": risk_per_unit,
            "planned_risk": 0.0,
            "risk_pct": 0.0,
        }

    regime_mult = regime_risk_multiplier(regime)
    drawdown_mult = drawdown_multiplier(drawdown_pct)

    hard_risk_pct = (
        MAX_KELLY_RISK_PCT
        * regime_mult
        * drawdown_mult
    )

    theoretical_kelly = kelly_fraction(
        probability,
        reward,
        risk_per_unit,
    )

    kelly_risk_pct = min(
        MAX_KELLY_RISK_PCT,
        theoretical_kelly * KELLY_FRACTION,
    )

    # Never allow Kelly to increase risk above the hard ceiling.
    effective_risk_pct = min(
        hard_risk_pct,
        kelly_risk_pct if kelly_risk_pct > 0 else hard_risk_pct,
    )

    max_risk_money = capital * effective_risk_pct

    raw_quantity = max_risk_money / risk_per_unit
    lots = int(raw_quantity // lot_size)
    quantity = lots * lot_size

    planned_risk = quantity * risk_per_unit

    return {
        "quantity": quantity,
        "lots": lots,
        "risk_per_unit": risk_per_unit,
        "planned_risk": planned_risk,
        "risk_pct": (
            planned_risk / capital
            if capital > 0
            else 0.0
        ),
        "kelly_theoretical": theoretical_kelly,
        "kelly_fraction_used": KELLY_FRACTION,
        "regime_multiplier": regime_mult,
        "drawdown_multiplier": drawdown_mult,
        "effective_risk_pct": effective_risk_pct,
    }
