"""
ATHENA-X Quantitative Risk Engine
---------------------------------

Isolated quantitative decision and risk layer between:

    Market Analysis
          ↓
       XGBoost
          ↓
    Risk Engine
          ↓
    Order Sizing

This module NEVER places orders.

Design goals:
- Preserve Athena's existing public interfaces.
- Keep regime classification independent from order execution.
- Avoid treating indicator direction as certainty.
- Use ATR-normalized trend strength.
- Separate trade approval from position sizing.
- Apply hard drawdown and risk limits.
- Never allow zero-edge trades to receive full risk.
- Keep probability handling conservative when ML is unavailable.
- Keep all calculations deterministic and lightweight.
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


# ============================================================
# NUMERIC HELPERS
# ============================================================

def _finite(value, default=0.0):
    """
    Convert a value to a finite float.

    Invalid, missing, NaN and infinite values become default.
    """
    try:
        value = float(value)

        if math.isfinite(value):
            return value

        return default

    except (TypeError, ValueError):
        return default


def clamp(value, low=0.0, high=1.0):
    """
    Clamp a numeric value into [low, high].
    """
    value = _finite(value)

    low = _finite(low)
    high = _finite(high)

    if low > high:
        low, high = high, low

    return max(low, min(high, value))


def _safe_ratio(numerator, denominator, default=0.0):
    """
    Safe division helper.
    """
    numerator = _finite(numerator)
    denominator = _finite(denominator)

    if denominator == 0:
        return default

    return numerator / denominator


# ============================================================
# MARKET REGIME
# ============================================================

def classify_regime(market):
    """
    Classify the current underlying market regime.

    The classification uses measurements Athena already calculates:

        ADX
        ATR%
        EMA20
        EMA50
        price

    The important improvement is that EMA separation is considered
    relative to ATR rather than relying only on an absolute percentage.

    Possible regimes:

        HIGH_VOL
        LOW_VOL_RANGE
        TRENDING
        RANGING
        NORMAL
        UNKNOWN
    """

    if not isinstance(market, dict):
        return "UNKNOWN"

    price = _finite(market.get("price"))

    if price <= 0:
        return "UNKNOWN"

    adx = _finite(market.get("adx"))

    atr = _finite(market.get("atr"))

    atr_percent = market.get("atr_percent")

    if atr_percent is None:
        atr_percent = _safe_ratio(
            atr,
            price,
            0.0,
        ) * 100.0

    atr_percent = max(0.0, _finite(atr_percent))

    ema20 = _finite(market.get("ema20"))
    ema50 = _finite(market.get("ema50"))
    ema200 = _finite(market.get("ema200"))

    # --------------------------------------------------------
    # ATR-normalized EMA separation
    # --------------------------------------------------------

    ema_distance = abs(ema20 - ema50)

    if atr > 0:
        ema_spread_atr = ema_distance / atr
    else:
        ema_spread_atr = 0.0

    # Existing configuration threshold is retained for
    # compatibility, but we also require meaningful ATR-relative
    # separation before calling something a strong trend.
    configured_spread = max(
        0.0,
        _finite(TREND_EMA_SPREAD_PCT),
    )

    configured_spread_fraction = (
        configured_spread / 100.0
    )

    percentage_spread = (
        ema_distance / price
        if price > 0
        else 0.0
    )

    percentage_trend_ok = (
        percentage_spread >= configured_spread_fraction
    )

    atr_trend_ok = ema_spread_atr >= 0.50

    # Require either:
    #
    #   - configured percentage separation
    #   OR
    #   - meaningful separation relative to ATR.
    #
    # This prevents tiny EMA differences from being treated
    # as a strong directional trend.

    trend_structure_ok = (
        percentage_trend_ok
        or atr_trend_ok
    )

    # --------------------------------------------------------
    # High volatility
    # --------------------------------------------------------

    if atr_percent >= HIGH_VOL_ATR_PCT:
        return "HIGH_VOL"

    # --------------------------------------------------------
    # Low-volatility range
    # --------------------------------------------------------

    if (
        atr_percent <= LOW_VOL_ATR_PCT
        and adx < TREND_ADX_THRESHOLD
    ):
        return "LOW_VOL_RANGE"

    # --------------------------------------------------------
    # Strong directional trend
    # --------------------------------------------------------

    if (
        adx >= TREND_ADX_THRESHOLD
        and trend_structure_ok
    ):
        return "TRENDING"

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    if adx <= RANGE_ADX_THRESHOLD:
        return "RANGING"

    # --------------------------------------------------------
    # Everything between range and trend
    # --------------------------------------------------------

    return "NORMAL"


# ============================================================
# DRAWDOWN CONTROL
# ============================================================

def drawdown_multiplier(drawdown_pct):
    """
    Convert current equity drawdown into a risk multiplier.

    Behaviour:

        0% → full risk

        <= DRAWDOWN_FULL_RISK_PCT
             → 1.0

        Between FULL_RISK and STOP
             → progressively reduce risk

        >= DRAWDOWN_STOP_PCT
             → 0.0
    """

    drawdown = max(
        0.0,
        _finite(drawdown_pct),
    )

    full_risk = max(
        0.0,
        _finite(DRAWDOWN_FULL_RISK_PCT),
    )

    stop = max(
        full_risk,
        _finite(DRAWDOWN_STOP_PCT),
    )

    if drawdown >= stop:
        return 0.0

    if drawdown <= full_risk:
        return 1.0

    span = stop - full_risk

    if span <= 0:
        return 0.0

    progress = (
        (drawdown - full_risk)
        / span
    )

    reduction_limit = clamp(
        MAX_DRAWDOWN_RISK_REDUCTION,
        0.0,
        1.0,
    )

    reduction = min(
        reduction_limit,
        progress * reduction_limit,
    )

    return max(
        0.0,
        1.0 - reduction,
    )


# ============================================================
# REGIME RISK
# ============================================================

def regime_risk_multiplier(regime):
    """
    Convert market regime into a risk multiplier.

    This does NOT determine whether a trade is correct.

    It only determines how aggressively capital may be exposed
    if the trade otherwise passes the quantitative checks.
    """

    regime = str(
        regime or "UNKNOWN"
    ).upper()

    multipliers = {
        # Strong directional conditions are the most favourable
        # for directional option buying.
        "TRENDING": 1.00,

        # Transitional conditions receive slightly less exposure.
        "NORMAL": 0.75,

        # Range conditions are less attractive for directional
        # option trades.
        "RANGING": 0.50,

        "LOW_VOL_RANGE": 0.35,

        # High volatility can create both opportunity and
        # execution risk, so exposure is reduced.
        "HIGH_VOL": 0.35,

        # Unknown conditions should never receive risk.
        "UNKNOWN": 0.00,
    }

    return multipliers.get(
        regime,
        0.00,
    )


# ============================================================
# EXPECTED VALUE
# ============================================================

def expected_value(
    probability,
    reward,
    risk,
    costs=0.0,
):
    """
    Expected monetary value:

        EV =
            P(win) * Reward
            -
            P(loss) * Risk
            -
            Costs
    """

    p = clamp(
        probability,
        0.0,
        1.0,
    )

    reward = max(
        0.0,
        _finite(reward),
    )

    risk = max(
        0.0,
        _finite(risk),
    )

    costs = max(
        0.0,
        _finite(costs),
    )

    return (
        p * reward
        -
        (1.0 - p) * risk
        -
        costs
    )


def expected_value_per_risk(
    probability,
    reward,
    risk,
    costs=0.0,
):
    """
    Normalize expected value by the amount being risked.
    """

    risk = max(
        0.0,
        _finite(risk),
    )

    if risk <= 0:
        return float("-inf")

    return (
        expected_value(
            probability,
            reward,
            risk,
            costs,
        )
        / risk
    )


# ============================================================
# KELLY
# ============================================================

def kelly_fraction(
    probability,
    reward,
    risk,
):
    """
    Calculate theoretical Kelly fraction.

    This is NOT used raw.

    Athena applies:
        KELLY_FRACTION

    and:
        MAX_KELLY_RISK_PCT
    """

    p = clamp(
        probability,
        0.0,
        1.0,
    )

    reward = max(
        0.0,
        _finite(reward),
    )

    risk = max(
        0.0,
        _finite(risk),
    )

    if reward <= 0 or risk <= 0:
        return 0.0

    b = reward / risk

    if b <= 0:
        return 0.0

    q = 1.0 - p

    raw = (
        (b * p) - q
    ) / b

    return clamp(
        raw,
        0.0,
        1.0,
    )


# ============================================================
# PROBABILITY
# ============================================================

def effective_probability(
    ml_probability,
    ml_trained,
    market_confidence,
):
    """
    Produce the probability used by the quantitative layer.

    IMPORTANT:

    If XGBoost is not trained, Athena must NOT turn technical
    confidence into an unrealistic probability.

    Therefore pre-ML confidence is capped at 60%.

    When ML is trained, its probability is used, but constrained
    to avoid mathematically extreme 0/1 probabilities.
    """

    if ml_trained:
        probability = clamp(
            ml_probability,
            0.01,
            0.99,
        )

        return probability

    # No trained model:
    #
    # Market confidence is useful as a signal-strength measure,
    # but it is not a statistically validated win probability.
    #
    # Keep it conservative.
    confidence = clamp(
        market_confidence,
        0.50,
        0.60,
    )

    return confidence


# ============================================================
# COST MODEL
# ============================================================

def estimate_trade_costs(
    reward,
    risk,
    extra_costs=0.0,
):
    """
    Estimate normalized transaction/friction costs.

    The function intentionally remains lightweight because the
    actual option execution price is handled elsewhere.
    """

    reward = max(
        0.0,
        _finite(reward),
    )

    risk = max(
        0.0,
        _finite(risk),
    )

    extra_costs = max(
        0.0,
        _finite(extra_costs),
    )

    notional_proxy = max(
        reward,
        risk,
    )

    slippage = (
        notional_proxy
        * max(
            0.0,
            _finite(SLIPPAGE_PCT),
        )
    )

    transaction = (
        notional_proxy
        * max(
            0.0,
            _finite(TRANSACTION_COST_PCT),
        )
    )

    return (
        slippage
        + transaction
        + extra_costs
    )


# ============================================================
# TRADE EVALUATION
# ============================================================

def evaluate_trade(
    probability,
    reward,
    risk,
    market,
    drawdown_pct=0.0,
    estimated_costs=0.0,
):
    """
    Evaluate a candidate trade.

    This function does NOT:

        - place an order
        - modify account state
        - modify database state
        - modify global state

    It only returns a decision record.
    """

    # --------------------------------------------------------
    # Normalize inputs
    # --------------------------------------------------------

    probability = clamp(
        probability,
        0.0,
        1.0,
    )

    reward = max(
        0.0,
        _finite(reward),
    )

    risk = max(
        0.0,
        _finite(risk),
    )

    estimated_costs = max(
        0.0,
        _finite(estimated_costs),
    )

    # --------------------------------------------------------
    # Regime
    # --------------------------------------------------------

    regime = classify_regime(
        market
    )

    regime_mult = regime_risk_multiplier(
        regime
    )

    drawdown_mult = drawdown_multiplier(
        drawdown_pct
    )

    total_risk_multiplier = (
        regime_mult
        * drawdown_mult
    )

    # --------------------------------------------------------
    # Hard input validation FIRST
    # --------------------------------------------------------

    if risk <= 0:
        return {
            "approved": False,
            "reason": "INVALID_RISK",
            "regime": regime,
            "probability": probability,
            "reward": reward,
            "risk": risk,
            "costs": estimated_costs,
            "expected_value": float("-inf"),
            "expected_value_per_risk": float("-inf"),
            "kelly_fraction": 0.0,
            "fractional_kelly": 0.0,
            "drawdown_multiplier": drawdown_mult,
            "regime_multiplier": regime_mult,
            "risk_multiplier": total_risk_multiplier,
            "max_kelly_risk_pct": MAX_KELLY_RISK_PCT,
        }

    if reward <= 0:
        return {
            "approved": False,
            "reason": "INVALID_REWARD",
            "regime": regime,
            "probability": probability,
            "reward": reward,
            "risk": risk,
            "costs": estimated_costs,
            "expected_value": float("-inf"),
            "expected_value_per_risk": float("-inf"),
            "kelly_fraction": 0.0,
            "fractional_kelly": 0.0,
            "drawdown_multiplier": drawdown_mult,
            "regime_multiplier": regime_mult,
            "risk_multiplier": total_risk_multiplier,
            "max_kelly_risk_pct": MAX_KELLY_RISK_PCT,
        }

    # --------------------------------------------------------
    # EV
    # --------------------------------------------------------

    ev = expected_value(
        probability,
        reward,
        risk,
        estimated_costs,
    )

    ev_per_risk = (
        ev / risk
        if risk > 0
        else float("-inf")
    )

    # --------------------------------------------------------
    # Kelly
    # --------------------------------------------------------

    kelly = kelly_fraction(
        probability,
        reward,
        risk,
    )

    fractional_kelly = (
        kelly
        * clamp(
            KELLY_FRACTION,
            0.0,
            1.0,
        )
    )
def timeframe_alignment(
    entry_direction,
    regime_direction,
    regime=None,
):
    """
    Compare the short-term entry direction with the higher-timeframe
    regime direction.

    Returns a decision label only. It does not execute or size trades.
    """

    entry = str(entry_direction or "").upper()
    regime_direction = str(regime_direction or "").upper()
    regime = str(regime or "").upper()

    if entry not in {"BULLISH", "BEARISH"}:
        return {
            "aligned": False,
            "action": "WAIT",
            "reason": "INVALID_ENTRY_DIRECTION",
        }

    if regime_direction not in {"BULLISH", "BEARISH"}:
        return {
            "aligned": False,
            "action": "WAIT",
            "reason": "INVALID_REGIME_DIRECTION",
        }

    if regime in {"UNKNOWN", ""}:
        return {
            "aligned": False,
            "action": "WAIT",
            "reason": "UNKNOWN_REGIME",
        }

    if entry == regime_direction:
        return {
            "aligned": True,
            "action": "PROCEED",
            "reason": "TIMEFRAME_ALIGNED",
        }

    return {
        "aligned": False,
        "action": "WAIT",
        "reason": "TIMEFRAME_CONFLICT",
    }
    # --------------------------------------------------------
    # Approval gates
    # --------------------------------------------------------

    approved = True
    reason = "APPROVED"

    if regime == "UNKNOWN":
        approved = False
        reason = "UNKNOWN_REGIME"

    elif drawdown_mult <= 0:
        approved = False
        reason = "DRAWDOWN_STOP"

    elif regime_mult <= 0:
        approved = False
        reason = "REGIME_REJECTED"

    elif ev_per_risk < EV_MIN_PER_RISK:
        approved = False
        reason = "NEGATIVE_OR_WEAK_EV"

    elif kelly <= 0:
        # Critical safety improvement:
        #
        # A zero Kelly fraction means the probability/reward/risk
        # combination does not justify risking capital.
        #
        # The old implementation could still allow hard risk when
        # Kelly was zero. That is undesirable.
        approved = False
        reason = "NO_POSITIVE_KELLY_EDGE"

    return {
        "approved": approved,
        "reason": reason,

        "regime": regime,

        "probability": probability,

        "reward": reward,
        "risk": risk,
        "costs": estimated_costs,

        "expected_value": ev,
        "expected_value_per_risk": ev_per_risk,

        "kelly_fraction": kelly,
        "fractional_kelly": fractional_kelly,

        "drawdown_multiplier": drawdown_mult,
        "regime_multiplier": regime_mult,

        "risk_multiplier": total_risk_multiplier,

        "max_kelly_risk_pct": MAX_KELLY_RISK_PCT,
    }


# ============================================================
# POSITION SIZING
# ============================================================

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
    Calculate option quantity using:

        account capital
        stop distance
        Kelly edge
        regime risk
        drawdown risk

    The function NEVER exceeds MAX_KELLY_RISK_PCT.

    Quantity is always rounded DOWN to complete lots.
    """

    capital = max(
        0.0,
        _finite(capital),
    )

    entry = max(
        0.0,
        _finite(entry),
    )

    stop = max(
        0.0,
        _finite(stop),
    )

    reward = max(
        0.0,
        _finite(reward),
    )

    probability = clamp(
        probability,
        0.0,
        1.0,
    )

    try:
        lot_size = int(
            _finite(
                lot_size,
                1,
            )
        )
    except (TypeError, ValueError):
        lot_size = 1

    lot_size = max(
        1,
        lot_size,
    )

    # --------------------------------------------------------
    # Stop distance
    # --------------------------------------------------------

    risk_per_unit = (
        entry - stop
    )

    # --------------------------------------------------------
    # Invalid trade
    # --------------------------------------------------------

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
            "kelly_theoretical": 0.0,
            "kelly_fraction_used": KELLY_FRACTION,
            "regime_multiplier": regime_risk_multiplier(
                regime
            ),
            "drawdown_multiplier": drawdown_multiplier(
                drawdown_pct
            ),
            "effective_risk_pct": 0.0,
        }

    # --------------------------------------------------------
    # Regime and drawdown
    # --------------------------------------------------------

    regime_mult = regime_risk_multiplier(
        regime
    )

    drawdown_mult = drawdown_multiplier(
        drawdown_pct
    )

    # No capital risk in a blocked regime/drawdown.
    if (
        regime_mult <= 0
        or drawdown_mult <= 0
    ):
        return {
            "quantity": 0,
            "lots": 0,
            "risk_per_unit": risk_per_unit,
            "planned_risk": 0.0,
            "risk_pct": 0.0,
            "kelly_theoretical": 0.0,
            "kelly_fraction_used": KELLY_FRACTION,
            "regime_multiplier": regime_mult,
            "drawdown_multiplier": drawdown_mult,
            "effective_risk_pct": 0.0,
        }

    # --------------------------------------------------------
    # Kelly
    # --------------------------------------------------------

    theoretical_kelly = kelly_fraction(
        probability,
        reward,
        risk_per_unit,
    )

    # No positive mathematical edge.
    #
    # Do NOT fall back to the hard risk ceiling.
    if theoretical_kelly <= 0:
        return {
            "quantity": 0,
            "lots": 0,
            "risk_per_unit": risk_per_unit,
            "planned_risk": 0.0,
            "risk_pct": 0.0,
            "kelly_theoretical": theoretical_kelly,
            "kelly_fraction_used": KELLY_FRACTION,
            "regime_multiplier": regime_mult,
            "drawdown_multiplier": drawdown_mult,
            "effective_risk_pct": 0.0,
        }

    # --------------------------------------------------------
    # Hard account risk ceiling
    # --------------------------------------------------------

    max_risk_pct = max(
        0.0,
        _finite(MAX_KELLY_RISK_PCT),
    )

    hard_risk_pct = (
        max_risk_pct
        * regime_mult
        * drawdown_mult
    )

    # --------------------------------------------------------
    # Fractional Kelly risk
    # --------------------------------------------------------

    fractional_kelly = (
        theoretical_kelly
        * clamp(
            KELLY_FRACTION,
            0.0,
            1.0,
        )
    )

    kelly_risk_pct = min(
        max_risk_pct,
        fractional_kelly,
    )

    # --------------------------------------------------------
    # Final risk
    # --------------------------------------------------------

    effective_risk_pct = min(
        hard_risk_pct,
        kelly_risk_pct,
    )

    effective_risk_pct = max(
        0.0,
        effective_risk_pct,
    )

    # --------------------------------------------------------
    # Maximum monetary risk
    # --------------------------------------------------------

    max_risk_money = (
        capital
        * effective_risk_pct
    )

    if max_risk_money <= 0:
        return {
            "quantity": 0,
            "lots": 0,
            "risk_per_unit": risk_per_unit,
            "planned_risk": 0.0,
            "risk_pct": 0.0,
            "kelly_theoretical": theoretical_kelly,
            "kelly_fraction_used": KELLY_FRACTION,
            "regime_multiplier": regime_mult,
            "drawdown_multiplier": drawdown_mult,
            "effective_risk_pct": 0.0,
        }

    # --------------------------------------------------------
    # Quantity
    # --------------------------------------------------------

    raw_quantity = (
        max_risk_money
        / risk_per_unit
    )

    # Always round DOWN.
    lots = int(
        raw_quantity // lot_size
    )

    quantity = (
        lots
        * lot_size
    )

    # --------------------------------------------------------
    # Actual planned risk
    # --------------------------------------------------------

    planned_risk = (
        quantity
        * risk_per_unit
    )

    actual_risk_pct = (
        planned_risk / capital
        if capital > 0
        else 0.0
    )

    return {
        "quantity": quantity,
        "lots": lots,

        "risk_per_unit": risk_per_unit,

        "planned_risk": planned_risk,

        "risk_pct": actual_risk_pct,

        "kelly_theoretical": theoretical_kelly,

        "kelly_fraction_used": KELLY_FRACTION,

        "regime_multiplier": regime_mult,

        "drawdown_multiplier": drawdown_mult,

        "effective_risk_pct": effective_risk_pct,
    }