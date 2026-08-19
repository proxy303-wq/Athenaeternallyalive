"""
ATHENA-X Technical Indicators
Robust, dependency-light indicator calculations for the underlying index data.

The functions in this module intentionally do not make trading decisions.
They only calculate and return indicator values.
"""

import numpy as np
import pandas as pd


# ============================================================
# DATA HELPERS
# ============================================================

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with lowercase, stripped column names."""
    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]
    return result


def _require_ohlc(df: pd.DataFrame) -> None:
    """Fail early if the required OHLC columns are absent."""
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required OHLC columns: {sorted(missing)}"
        )


def _numeric(series: pd.Series, name: str) -> pd.Series:
    """Convert a series to numeric and reject an entirely invalid series."""
    result = pd.to_numeric(series, errors="coerce")
    if result.notna().sum() == 0:
        raise ValueError(f"Column '{name}' contains no valid numeric values.")
    return result


def get_column(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Backward-compatible column accessor."""
    normalised = _normalise_columns(df)
    key = col_name.strip().lower()

    if key not in normalised.columns:
        raise KeyError(f"Column '{col_name}' not found")

    return normalised[key]


# ============================================================
# BASIC INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    if period <= 0:
        raise ValueError("EMA period must be greater than 0.")
    return series.ewm(span=period, adjust=False, min_periods=1).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("SMA period must be greater than 0.")
    return series.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder-style exponential smoothing.

    Values remain NaN until enough observations exist. This avoids
    silently converting an uninitialized RSI into a neutral 50.
    """
    if period <= 0:
        raise ValueError("RSI period must be greater than 0.")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))

    # A flat series after initialization is conventionally neutral.
    flat = (avg_gain == 0) & (avg_loss == 0)
    result = result.mask(flat, 50.0)

    # If there are gains but no losses, RSI is 100.
    result = result.mask((avg_gain > 0) & (avg_loss == 0), 100.0)

    return result.clip(0, 100)


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
):
    """Return MACD line, signal line and histogram."""
    if min(fast, slow, signal) <= 0:
        raise ValueError("MACD periods must be greater than 0.")
    if fast >= slow:
        raise ValueError("MACD fast period must be less than slow period.")

    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
):
    """Return upper band, middle SMA and lower band."""
    if period <= 0 or std_dev < 0:
        raise ValueError("Invalid Bollinger Band parameters.")

    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()

    upper = middle + std * std_dev
    lower = middle - std * std_dev

    return upper, middle, lower


# ============================================================
# ATR / ADX
# ============================================================

def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder-style smoothing."""
    if period <= 0:
        raise ValueError("ATR period must be greater than 0.")

    tr = true_range(df)

    return tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index.

    Uses directional movement and Wilder-style smoothing. The result
    remains NaN until sufficient history exists rather than filling
    unavailable values with an artificial 25.
    """
    if period <= 0:
        raise ValueError("ADX period must be greater than 0.")

    high = df["high"]
    low = df["low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        ),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        ),
        index=df.index,
    )

    tr = true_range(df)

    atr_value = tr.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_dm_smoothed = plus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()
    minus_dm_smoothed = minus_dm.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    plus_di = 100 * plus_dm_smoothed / atr_value.replace(0, np.nan)
    minus_di = 100 * minus_dm_smoothed / atr_value.replace(0, np.nan)

    denominator = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denominator

    return dx.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean().clip(0, 100)


# ============================================================
# VWAP
# ============================================================

def vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate cumulative VWAP over the supplied dataframe.

    The caller should supply a session-scoped dataframe when using
    VWAP as an intraday signal.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
    else:
        # Preserve compatibility with index feeds that do not expose volume.
        volume = pd.Series(1.0, index=df.index)

    volume = volume.fillna(0.0).clip(lower=0.0)

    cumulative_volume = volume.cumsum()
    cumulative_value = (typical_price * volume).cumsum()

    return cumulative_value.div(
        cumulative_volume.replace(0, np.nan)
    )


# ============================================================
# ICHIMOKU
# ============================================================

def ichimoku(
    df: pd.DataFrame,
    conversion: int = 9,
    base: int = 26,
    lagging: int = 52,
):
    """
    Calculate Ichimoku components.

    Returns:
        tenkan, kijun, senkou_a, senkou_b, chikou

    The existing Athena strategy uses the current/latest values of
    these columns for trend identification.
    """
    if min(conversion, base, lagging) <= 0:
        raise ValueError("Ichimoku periods must be greater than 0.")

    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (
        high.rolling(conversion, min_periods=conversion).max()
        + low.rolling(conversion, min_periods=conversion).min()
    ) / 2.0

    kijun = (
        high.rolling(base, min_periods=base).max()
        + low.rolling(base, min_periods=base).min()
    ) / 2.0

    # Preserve the original Athena alignment.
    senkou_a = ((tenkan + kijun) / 2.0).shift(base)

    senkou_b = (
        (
            high.rolling(lagging, min_periods=lagging).max()
            + low.rolling(lagging, min_periods=lagging).min()
        )
        / 2.0
    ).shift(base)

    chikou = close.shift(-base)

    return tenkan, kijun, senkou_a, senkou_b, chikou


# ============================================================
# COMPLETE INDICATOR PIPELINE
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all indicators required by Athena-X.

    Returns a new dataframe. The input dataframe is never modified.
    """
    if df is None:
        return pd.DataFrame()

    if not isinstance(df, pd.DataFrame):
        raise TypeError("calculate_indicators expects a pandas DataFrame.")

    if df.empty:
        return df.copy()

    result = _normalise_columns(df)
    _require_ohlc(result)

    # Convert core market columns to numeric.
    for column in ("open", "high", "low", "close", "volume"):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    # Reject invalid OHLC rows instead of producing misleading indicators.
    result = result.dropna(subset=["high", "low", "close"]).copy()

    if result.empty:
        return result

    # Ensure chronological ordering when a datetime index is available.
    if isinstance(result.index, pd.DatetimeIndex):
        result = result.sort_index()

    close = result["close"]

    # 1. Moving averages
    result["ema20"] = ema(close, 20)
    result["ema50"] = ema(close, 50)
    result["ema200"] = ema(close, 200)

    # 2. SMA
    result["sma20"] = sma(close, 20)
    result["sma50"] = sma(close, 50)

    # 3. RSI
    result["rsi"] = rsi(close, 14)

    # 4. MACD
    (
        result["macd"],
        result["macd_signal"],
        result["macd_hist"],
    ) = macd(close, 12, 26, 9)

    # 5. Bollinger Bands
    (
        result["bb_upper"],
        result["bb_middle"],
        result["bb_lower"],
    ) = bollinger_bands(close, 20, 2)

    # 6. ADX
    result["adx"] = adx(result, 14)

    # 7. ATR
    result["atr"] = atr(result, 14)

    # 8. VWAP
    result["vwap"] = vwap(result)

    # 9. Ichimoku
    (
        result["tenkan"],
        result["kijun"],
        result["senkou_a"],
        result["senkou_b"],
        result["chikou"],
    ) = ichimoku(result, 9, 26, 52)

    # 10. Rolling support / resistance
    result["resistance"] = result["high"].rolling(
        20, min_periods=20
    ).max()
    result["support"] = result["low"].rolling(
        20, min_periods=20
    ).min()

    return result


def latest_valid_row(df: pd.DataFrame) -> pd.Series | None:
    """
    Return the latest row with the core indicators required for trading.

    This prevents Athena from making a decision using partially initialized
    indicator values.
    """
    if df is None or df.empty:
        return None

    required = [
        "ema20",
        "ema50",
        "ema200",
        "sma20",
        "sma50",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "adx",
        "atr",
        "vwap",
        "tenkan",
        "kijun",
        "resistance",
        "support",
    ]

    available = [column for column in required if column in df.columns]
    if not available:
        return None

    valid = df.dropna(subset=available)
    if valid.empty:
        return None

    return valid.iloc[-1]
