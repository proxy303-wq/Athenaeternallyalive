"""
ATHENA-X Wealth Dashboard
Minimal Streamlit dashboard shell.

This file is intentionally independent from the trading engine.
It does NOT start main.py and does NOT place trades.
"""

import os
import sqlite3
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

from live_market import get_market_snapshot, start_live_feed


st.set_page_config(
    page_title="ATHENA-X Wealth Manager",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Athena runtime snapshot (read-only)
# ------------------------------------------------------------

def get_athena_runtime_state():
    """
    Read Athena's persisted active-trade state.

    The dashboard does not import or execute main.py.
    It only reads the active_trade table.
    """
    try:
        active = read_athena_table("active_trade")

        if active.empty:
            return {
                "active": False,
                "trade": {},
            }

        row = active.iloc[0].to_dict()
        payload = row.get("payload")

        if not payload:
            return {
                "active": False,
                "trade": {},
            }

        import json

        trade = json.loads(payload)

        if not isinstance(trade, dict):
            return {
                "active": False,
                "trade": {},
            }

        return {
            "active": True,
            "trade": trade,
        }

    except Exception as exc:
        return {
            "active": False,
            "trade": {},
            "error": str(exc),
        }


# ------------------------------------------------------------
# Read-only Dhan portfolio connection
# ------------------------------------------------------------

def get_dhan_client():
    """Create a Dhan client for read-only dashboard queries."""
    try:
        from config import CLIENT_ID, ACCESS_TOKEN
        from dhanhq import DhanContext, dhanhq

        if not CLIENT_ID or not ACCESS_TOKEN:
            return None

        return dhanhq(
            DhanContext(
                CLIENT_ID,
                ACCESS_TOKEN,
            )
        )
    except Exception:
        return None


def get_dhan_portfolio():
    """
    Read funds, holdings and positions from Dhan.

    This function deliberately exposes only read operations.
    It never places, modifies or cancels an order.
    """
    client = get_dhan_client()

    result = {
        "connected": bool(client),
        "funds": {},
        "holdings": pd.DataFrame(),
        "positions": pd.DataFrame(),
        "errors": [],
    }

    if client is None:
        result["errors"].append("Dhan credentials unavailable.")
        return result

    # Funds
    try:
        response = client.get_fund_limits()

        if (
            isinstance(response, dict)
            and response.get("status") == "success"
        ):
            result["funds"] = response.get("data", {}) or {}
        else:
            result["errors"].append(
                "Unable to retrieve Dhan funds."
            )
    except Exception as exc:
        result["errors"].append(
            f"Funds lookup failed: {exc}"
        )

    # Holdings
    try:
        response = client.get_holdings()

        if (
            isinstance(response, dict)
            and response.get("status") == "success"
        ):
            data = response.get("data", []) or []
            result["holdings"] = (
                pd.DataFrame(data)
                if isinstance(data, list)
                else pd.DataFrame()
            )
        else:
            # Dhan returns HOLDING_ERROR when the account has no holdings.
            remarks = (
                response.get("remarks", {})
                if isinstance(response, dict)
                else {}
            )
            error_code = (
                remarks.get("error_code")
                if isinstance(remarks, dict)
                else None
            )

            if error_code != "DH-1111":
                result["errors"].append(
                    "Unable to retrieve Dhan holdings."
                )
    except Exception as exc:
        result["errors"].append(
            f"Holdings lookup failed: {exc}"
        )

    # Positions
    try:
        response = client.get_positions()

        if (
            isinstance(response, dict)
            and response.get("status") == "success"
        ):
            data = response.get("data", []) or []
            result["positions"] = (
                pd.DataFrame(data)
                if isinstance(data, list)
                else pd.DataFrame()
            )
        else:
            result["errors"].append(
                "Unable to retrieve Dhan positions."
            )
    except Exception as exc:
        result["errors"].append(
            f"Positions lookup failed: {exc}"
        )

    return result


# ------------------------------------------------------------
# Read-only Athena database
# ------------------------------------------------------------

DATABASE_PATH = os.getenv(
    "ATHENA_DATABASE_PATH",
    "data/athena.db",
)


def read_athena_table(table_name):
    """Read an Athena table without writing to the database."""
    allowed = {
        "trades",
        "ml_history",
        "app_state",
        "active_trade",
        "wealth_monthly",
        "wealth_goals",
    }

    if table_name not in allowed:
        raise ValueError("Unsupported Athena table")

    db_path = Path(DATABASE_PATH)

    if not db_path.exists():
        return pd.DataFrame()

    connection = sqlite3.connect(
        f"file:{db_path.resolve()}?mode=ro",
        uri=True,
        timeout=5,
    )

    try:
        return pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            connection,
        )
    finally:
        connection.close()


def load_trades():
    try:
        return read_athena_table("trades")
    except Exception as exc:
        st.error(f"Unable to read Athena trades: {exc}")
        return pd.DataFrame()


def calculate_trade_metrics(trades):
    if trades.empty:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    pnl = pd.to_numeric(
        trades.get("pnl"),
        errors="coerce",
    ).fillna(0.0)

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum())
    gross_loss = abs(float(losses.sum()))

    equity = pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = equity - running_peak

    return {
        "total_trades": int(len(trades)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate": float((pnl > 0).mean() * 100),
        "total_pnl": float(pnl.sum()),
        "average_win": (
            float(wins.mean()) if not wins.empty else 0.0
        ),
        "average_loss": (
            float(losses.mean()) if not losses.empty else 0.0
        ),
        "profit_factor": (
            gross_profit / gross_loss
            if gross_loss > 0
            else 0.0
        ),
        "max_drawdown": (
            float(drawdown.min()) if not drawdown.empty else 0.0
        ),
    }


# Start the read-only Dhan WebSocket worker once per Streamlit process.
start_live_feed()


# ------------------------------------------------------------
# Wealth manager helpers
# ------------------------------------------------------------

def get_wealth_summary():
    """
    Build a read-only wealth snapshot from:
    - Dhan available cash
    - Dhan holdings
    - Dhan positions
    - Athena completed-trade P&L

    Manual assets/liabilities are kept in session state for now.
    No broker or trading mutations are performed.
    """
    portfolio = get_dhan_portfolio()
    funds = portfolio.get("funds", {})

    available_cash = float(
        funds.get("availabelBalance", 0.0) or 0.0
    )

    holdings = portfolio.get("holdings", pd.DataFrame())
    positions = portfolio.get("positions", pd.DataFrame())

    holdings_value = 0.0
    holdings_cost = 0.0

    if not holdings.empty:
        value_columns = [
            "totalQty",
            "totalQtyAvailable",
            "currentValue",
            "marketValue",
            "ltp",
        ]

        # Dhan payloads vary by endpoint/version, so calculate
        # conservatively from whatever fields are available.
        for col in ("currentValue", "marketValue"):
            if col in holdings.columns:
                holdings_value = pd.to_numeric(
                    holdings[col],
                    errors="coerce",
                ).fillna(0).sum()
                break

        if holdings_value == 0 and {"totalQty", "ltp"}.issubset(
            holdings.columns
        ):
            holdings_value = (
                pd.to_numeric(
                    holdings["totalQty"],
                    errors="coerce",
                ).fillna(0)
                * pd.to_numeric(
                    holdings["ltp"],
                    errors="coerce",
                ).fillna(0)
            ).sum()

        for col in ("totalCostValue", "costValue", "investedValue"):
            if col in holdings.columns:
                holdings_cost = pd.to_numeric(
                    holdings[col],
                    errors="coerce",
                ).fillna(0).sum()
                break

    positions_value = 0.0
    if not positions.empty:
        for col in (
            "daySellValue",
            "buyAvg",
            "sellAvg",
            "realizedProfit",
            "unrealizedProfit",
        ):
            if col == "unrealizedProfit" and col in positions.columns:
                # Keep P&L separately; don't treat it as gross portfolio value.
                continue

        if "unrealizedProfit" in positions.columns:
            positions_pnl = pd.to_numeric(
                positions["unrealizedProfit"],
                errors="coerce",
            ).fillna(0).sum()
        else:
            positions_pnl = 0.0
    else:
        positions_pnl = 0.0

    trades = load_trades()
    trade_metrics = calculate_trade_metrics(trades)

    if "manual_assets" not in st.session_state:
        st.session_state.manual_assets = 0.0

    if "manual_liabilities" not in st.session_state:
        st.session_state.manual_liabilities = 0.0

    gross_assets = (
        available_cash
        + holdings_value
        + float(st.session_state.manual_assets)
    )

    net_worth = gross_assets - float(
        st.session_state.manual_liabilities
    )

    return {
        "cash": available_cash,
        "holdings_value": float(holdings_value),
        "holdings_cost": float(holdings_cost),
        "positions_pnl": float(positions_pnl),
        "manual_assets": float(st.session_state.manual_assets),
        "manual_liabilities": float(
            st.session_state.manual_liabilities
        ),
        "gross_assets": float(gross_assets),
        "net_worth": float(net_worth),
        "trading_pnl": float(trade_metrics["total_pnl"]),
    }



# ------------------------------------------------------------
# Persistent Wealth Manager
# ------------------------------------------------------------

def ensure_wealth_tables():
    """Create small wealth-manager tables if they do not exist."""
    db_path = Path(DATABASE_PATH)
    if not db_path.exists():
        return

    connection = sqlite3.connect(str(db_path), timeout=5)

    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS wealth_monthly (
                month TEXT PRIMARY KEY,
                income REAL NOT NULL DEFAULT 0,
                expenses REAL NOT NULL DEFAULT 0,
                investments REAL NOT NULL DEFAULT 0,
                other_assets REAL NOT NULL DEFAULT 0,
                liabilities REAL NOT NULL DEFAULT 0,
                notes TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS wealth_goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                target_amount REAL NOT NULL,
                current_amount REAL NOT NULL DEFAULT 0,
                target_date TEXT,
                notes TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        connection.commit()
    finally:
        connection.close()


def read_wealth_months():
    ensure_wealth_tables()

    try:
        return read_athena_table("wealth_monthly")
    except Exception:
        return pd.DataFrame()


def read_wealth_goals():
    ensure_wealth_tables()

    try:
        return read_athena_table("wealth_goals")
    except Exception:
        return pd.DataFrame()


def save_wealth_month(
    month,
    income,
    expenses,
    investments,
    other_assets,
    liabilities,
    notes,
):
    ensure_wealth_tables()

    connection = sqlite3.connect(str(Path(DATABASE_PATH)), timeout=5)

    try:
        connection.execute(
            """
            INSERT INTO wealth_monthly (
                month, income, expenses, investments,
                other_assets, liabilities, notes, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(month) DO UPDATE SET
                income=excluded.income,
                expenses=excluded.expenses,
                investments=excluded.investments,
                other_assets=excluded.other_assets,
                liabilities=excluded.liabilities,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (
                month,
                float(income),
                float(expenses),
                float(investments),
                float(other_assets),
                float(liabilities),
                notes,
                datetime.now().astimezone().isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def save_wealth_goal(
    name,
    target_amount,
    current_amount,
    target_date,
    notes,
):
    ensure_wealth_tables()

    connection = sqlite3.connect(str(Path(DATABASE_PATH)), timeout=5)

    try:
        connection.execute(
            """
            INSERT INTO wealth_goals (
                name, target_amount, current_amount,
                target_date, notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                float(target_amount),
                float(current_amount),
                target_date,
                notes,
                datetime.now().astimezone().isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def calculate_wealth_metrics(months):
    if months.empty:
        return {
            "income": 0.0,
            "expenses": 0.0,
            "investments": 0.0,
            "savings": 0.0,
            "savings_rate": 0.0,
        }

    income = pd.to_numeric(
        months.get("income"),
        errors="coerce",
    ).fillna(0).sum()

    expenses = pd.to_numeric(
        months.get("expenses"),
        errors="coerce",
    ).fillna(0).sum()

    investments = pd.to_numeric(
        months.get("investments"),
        errors="coerce",
    ).fillna(0).sum()

    savings = income - expenses

    return {
        "income": float(income),
        "expenses": float(expenses),
        "investments": float(investments),
        "savings": float(savings),
        "savings_rate": (
            float(savings / income * 100)
            if income > 0
            else 0.0
        ),
    }


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "dashboard_mode" not in st.session_state:
    st.session_state.dashboard_mode = "PAPER"

if "transactions" not in st.session_state:
    st.session_state.transactions = pd.DataFrame(
        columns=[
            "Date",
            "Type",
            "Asset",
            "Quantity",
            "Price",
            "Amount",
            "Broker",
        ]
    )


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.title("ATHENA-X")
st.caption("Personal Wealth & Portfolio Manager")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    st.write("Command Center")

with col2:
    mode = st.selectbox(
        "Trading Mode",
        ["PAPER", "LIVE"],
        index=0,
        key="dashboard_mode",
    )

with col3:
    st.metric(
        "System",
        "ONLINE",
        help="Dashboard shell status only.",
    )

if mode == "LIVE":
    st.warning(
        "LIVE mode selected in the dashboard. "
        "This does not enable live trading in Athena. "
        "The trading engine remains controlled by LIVE_TRADING in config.py."
    )
else:
    st.info("PAPER mode — dashboard only; no orders are placed.")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("ATHENA-X")

    page = st.radio(
        "Navigate",
        [
            "Dashboard",
            "Portfolio",
            "Trading",
            "Wealth",
            "Risk",
            "Analytics",
            "ML",
            "System",
            "Goals",
            "Transactions",
            "Settings",
        ],
    )

    st.divider()

    st.caption(
        "Minimal architecture\n\n"
        "Dashboard: app.py\n"
        "Trading engine: main.py\n"
        "Live trading: disabled"
    )


# ------------------------------------------------------------
# Command Center data
#
# Athena-specific dashboard metrics. These are read-only and are
# built from the existing Dhan connection, Athena database and
# configured paper capital. No trading functions are called here.
# ------------------------------------------------------------

def get_command_center_data():
    try:
        from config import FALLBACK_CAPITAL, LIVE_TRADING
        starting_capital = float(FALLBACK_CAPITAL)
    except Exception:
        LIVE_TRADING = False
        starting_capital = 500000.0

    trades = load_trades()
    metrics = calculate_trade_metrics(trades)

    today_pnl = 0.0
    if not trades.empty and "pnl" in trades.columns and "timestamp" in trades.columns:
        dated = trades.copy()
        dated["_time"] = pd.to_datetime(
            dated["timestamp"], errors="coerce", utc=True
        )
        dated["_pnl"] = pd.to_numeric(
            dated["pnl"], errors="coerce"
        ).fillna(0.0)
        today = datetime.now().astimezone().date()
        try:
            local_dates = dated["_time"].dt.tz_convert(
                datetime.now().astimezone().tzinfo
            ).dt.date
        except Exception:
            local_dates = dated["_time"].dt.date
        today_pnl = float(
            dated.loc[local_dates == today, "_pnl"].sum()
        )

    realized_pnl = float(metrics.get("total_pnl", 0.0))
    athena_equity = starting_capital + realized_pnl

    # Athena's current dashboard objective is +1% of its capital anchor.
    daily_target = starting_capital * 0.01
    daily_loss_limit = starting_capital * 0.01

    peak_equity = starting_capital
    current_drawdown = 0.0
    if not trades.empty and "pnl" in trades.columns:
        pnl = pd.to_numeric(
            trades["pnl"], errors="coerce"
        ).fillna(0.0)
        equity_curve = starting_capital + pnl.cumsum()
        peaks = equity_curve.cummax()
        peak_equity = float(max(starting_capital, equity_curve.max()))
        current_drawdown = float(equity_curve.iloc[-1] - peaks.iloc[-1])

    portfolio = get_dhan_portfolio()
    funds = portfolio.get("funds", {}) or {}
    dhan_cash = float(
        funds.get("availabelBalance", 0.0) or 0.0
    )

    runtime = get_athena_runtime_state()

    return {
        "starting_capital": starting_capital,
        "athena_equity": athena_equity,
        "dhan_cash": dhan_cash,
        "today_pnl": today_pnl,
        "total_pnl": realized_pnl,
        "daily_target": daily_target,
        "daily_loss_limit": daily_loss_limit,
        "target_progress": (
            max(0.0, min(today_pnl / daily_target * 100.0, 100.0))
            if daily_target > 0 else 0.0
        ),
        "remaining_loss": max(0.0, daily_loss_limit + today_pnl),
        "current_drawdown": current_drawdown,
        "peak_equity": peak_equity,
        "trades": metrics.get("total_trades", 0),
        "active_trade": bool(runtime.get("active")),
        "live_trading": bool(LIVE_TRADING),
        "dhan_connected": bool(portfolio.get("connected")),
    }


# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------

if page == "Dashboard":
    st.subheader("Command Center")

    @st.fragment(run_every="5s")
    def live_market_panel():
        snapshot = get_market_snapshot()

        st.subheader("Live Market")

        left, right = st.columns(2)

        for column, symbol in zip(
            (left, right),
            ("NIFTY", "BANKNIFTY"),
        ):
            values = snapshot["data"][symbol]

            with column:
                ltp = values.get("ltp")
                previous = values.get("previous_close")
                updated = values.get("timestamp")

                if ltp is not None:
                    change = (
                        ltp - previous
                        if previous not in (None, 0)
                        else None
                    )
                    change_pct = (
                        change / previous * 100
                        if change is not None and previous
                        else None
                    )

                    st.metric(
                        symbol,
                        f"₹{ltp:,.2f}",
                        (
                            f"{change:+,.2f} ({change_pct:+.2f}%)"
                            if change is not None and change_pct is not None
                            else None
                        ),
                    )

                    if updated:
                        age = (
                            datetime.now(updated.tzinfo) - updated
                        ).total_seconds()
                        status = "🟢 LIVE" if age <= 10 else "🟠 STALE"
                        st.caption(
                            f"{status} · Updated {age:.1f}s ago"
                        )
                    else:
                        st.caption("🟠 Waiting for tick")
                else:
                    st.metric(symbol, "—")
                    st.caption("🟠 Waiting for Dhan WebSocket")

        if snapshot.get("error"):
            st.warning(
                "Dhan WebSocket: " + str(snapshot["error"])
            )

    live_market_panel()

    @st.fragment(run_every="5s")
    def current_movement_panel():
        snapshot = get_market_snapshot()

        st.subheader("Current Movement")

        cards = st.columns(2)

        for column, symbol in zip(
            cards,
            ("NIFTY", "BANKNIFTY"),
        ):
            values = snapshot["data"][symbol]

            with column:
                ltp = values.get("ltp")
                previous = values.get("previous_close")

                if ltp is None:
                    st.info(f"{symbol}: waiting for live ticks")
                    continue

                change = (
                    ltp - previous
                    if previous not in (None, 0)
                    else None
                )
                change_pct = (
                    change / previous * 100
                    if change is not None and previous
                    else None
                )

                if change is None:
                    direction = "FLAT"
                elif change > 0:
                    direction = "UP"
                elif change < 0:
                    direction = "DOWN"
                else:
                    direction = "FLAT"

                st.metric(
                    f"{symbol} Movement",
                    direction,
                    (
                        f"{change:+,.2f} pts "
                        f"({change_pct:+.2f}%)"
                        if change is not None and change_pct is not None
                        else None
                    ),
                )

    current_movement_panel()
    st.divider()

    @st.fragment(run_every="5s")
    def command_center_metrics():
        data = get_command_center_data()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Athena Equity",
            f"₹{data['athena_equity']:,.2f}",
        )
        c2.metric(
            "Dhan Cash",
            f"₹{data['dhan_cash']:,.2f}",
            "CONNECTED" if data["dhan_connected"] else "OFFLINE",
        )
        c3.metric(
            "Today's P&L",
            f"₹{data['today_pnl']:,.2f}",
            f"Target ₹{data['daily_target']:,.0f}",
        )
        c4.metric(
            "Total P&L",
            f"₹{data['total_pnl']:,.2f}",
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.metric(
            "Daily Target",
            f"₹{data['daily_target']:,.0f}",
            f"{data['target_progress']:.1f}%",
        )
        p2.metric(
            "Daily Loss Room",
            f"₹{data['remaining_loss']:,.2f}",
        )
        p3.metric(
            "Current Drawdown",
            f"₹{abs(min(data['current_drawdown'], 0.0)):,.2f}",
        )
        p4.metric(
            "Open Athena Trade",
            "YES" if data["active_trade"] else "NONE",
        )

        if data["today_pnl"] >= data["daily_target"]:
            st.success(
                "🎯 Daily objective reached. Athena should not open new trades today."
            )
        elif data["today_pnl"] <= -data["daily_loss_limit"]:
            st.error(
                "🛑 Daily loss limit reached. Athena should not open new trades today."
            )
        else:
            st.caption(
                f"Daily objective progress: {data['today_pnl']:,.2f} / "
                f"₹{data['daily_target']:,.0f} · "
                f"Risk room remaining: ₹{data['remaining_loss']:,.2f}"
            )

    command_center_metrics()
    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Portfolio Allocation")

        data = get_command_center_data()
        allocation = pd.DataFrame(
            {
                "Asset": ["Athena Equity", "Dhan Cash"],
                "Value": [
                    data["athena_equity"],
                    data["dhan_cash"],
                ],
            }
        )

        st.bar_chart(
            allocation.set_index("Asset"),
            y="Value",
        )

    with right:
        st.subheader("Athena Trading")

        data = get_command_center_data()
        t1, t2 = st.columns(2)
        t1.metric(
            "Mode",
            "LIVE" if data["live_trading"] else "PAPER",
        )
        t2.metric(
            "Open Trades",
            "1" if data["active_trade"] else "0",
        )

        st.write("Market status")
        if data["dhan_connected"]:
            st.success("Dhan connected · read-only dashboard")
        else:
            st.warning("Dhan connection unavailable")

    st.divider()

    st.subheader("System Health")

    db_exists = Path(DATABASE_PATH).exists()

    health = pd.DataFrame(
        [
            ["Dashboard", "ONLINE"],
            ["Trading Engine", "AVAILABLE"],
            ["Live Trading", "DISABLED"],
            [
                "Database",
                "CONNECTED (READ ONLY)"
                if db_exists
                else "NOT FOUND",
            ],
            ["Dhan", "ENGINE CONTROLLED"],
            ["Telegram", "ENGINE CONTROLLED"],
        ],
        columns=["Component", "Status"],
    )

    st.dataframe(
        health,
        width="stretch",
        hide_index=True,
    )


# ------------------------------------------------------------
# Portfolio
# ------------------------------------------------------------

elif page == "Portfolio":
    st.subheader("Dhan Portfolio")

    portfolio = get_dhan_portfolio()
    funds = portfolio["funds"]

    available = float(
        funds.get("availabelBalance", 0.0) or 0.0
    )
    withdrawable = float(
        funds.get("withdrawableBalance", 0.0) or 0.0
    )
    utilized = float(
        funds.get("utilizedAmount", 0.0) or 0.0
    )
    collateral = float(
        funds.get("collateralAmount", 0.0) or 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available Balance", f"₹{available:,.2f}")
    c2.metric("Withdrawable", f"₹{withdrawable:,.2f}")
    c3.metric("Utilized", f"₹{utilized:,.2f}")
    c4.metric("Collateral", f"₹{collateral:,.2f}")

    if portfolio["connected"]:
        st.success("Dhan connected — read-only dashboard access")
    else:
        st.error("Dhan connection unavailable")

    for error in portfolio["errors"]:
        st.warning(error)

    st.divider()

    holdings = portfolio["holdings"]
    positions = portfolio["positions"]

    h1, h2 = st.columns(2)

    with h1:
        st.subheader("Holdings")
        if holdings.empty:
            st.info("No Dhan holdings available.")
        else:
            st.dataframe(
                holdings,
                width="stretch",
                hide_index=True,
            )

    with h2:
        st.subheader("Open Positions")
        if positions.empty:
            st.info("No open Dhan positions.")
        else:
            st.dataframe(
                positions,
                width="stretch",
                hide_index=True,
            )

    st.caption(
        "Portfolio data is read directly from Dhan. "
        "This dashboard does not place, modify, or cancel orders."
    )


# ------------------------------------------------------------
# Trading
# ------------------------------------------------------------

elif page == "Trading":
    st.subheader("Athena Trading")

    @st.fragment(run_every="5s")
    def trading_live_market():
        snapshot = get_market_snapshot()
        c1, c2 = st.columns(2)

        for column, symbol in zip(
            (c1, c2),
            ("NIFTY", "BANKNIFTY"),
        ):
            values = snapshot["data"][symbol]
            ltp = values.get("ltp")
            previous = values.get("previous_close")

            with column:
                if ltp is None:
                    st.metric(symbol, "—")
                else:
                    change = (
                        ltp - previous
                        if previous not in (None, 0)
                        else None
                    )
                    st.metric(
                        symbol,
                        f"₹{ltp:,.2f}",
                        f"{change:+,.2f}"
                        if change is not None
                        else None,
                    )

    trading_live_market()

    trades = load_trades()
    metrics = calculate_trade_metrics(trades)
    runtime = get_athena_runtime_state()
    active_trade = runtime.get("trade", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", "PAPER")
    c2.metric("Completed Trades", metrics["total_trades"])
    c3.metric("Total P&L", f"₹{metrics['total_pnl']:,.2f}")
    c4.metric(
        "Win Rate",
        f"{metrics['win_rate']:.1f}%",
    )

    st.divider()

    st.subheader("Current Athena State")

    if runtime.get("active") and active_trade:
        st.success("Active Athena trade found in database.")

        instrument = active_trade.get(
            "instrument",
            active_trade.get("name", "—"),
        )
        option_type = active_trade.get("option_type", "—")
        strike = active_trade.get("strike", "—")
        entry = active_trade.get("entry", active_trade.get("entry_price", "—"))
        target = active_trade.get("target", "—")
        stop = active_trade.get("stop", active_trade.get("stop_loss", "—"))
        trailing = active_trade.get("trailing_stop", "—")
        quantity = active_trade.get("quantity", "—")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Instrument", str(instrument))
        r2.metric(
            "Option",
            f"{option_type} {strike}",
        )
        r3.metric("Quantity", str(quantity))
        r4.metric(
            "Entry",
            f"₹{float(entry):,.2f}"
            if isinstance(entry, (int, float))
            else str(entry),
        )

        r5, r6, r7 = st.columns(3)
        r5.metric(
            "Target",
            f"₹{float(target):,.2f}"
            if isinstance(target, (int, float))
            else str(target),
        )
        r6.metric(
            "Stop",
            f"₹{float(stop):,.2f}"
            if isinstance(stop, (int, float))
            else str(stop),
        )
        r7.metric(
            "Trailing Stop",
            f"₹{float(trailing):,.2f}"
            if isinstance(trailing, (int, float))
            else str(trailing),
        )
    else:
        st.info(
            "No active Athena trade is currently stored. "
            "This is expected outside market hours or before a paper trade."
        )

    if runtime.get("error"):
        st.warning(
            "Athena runtime state could not be read: "
            + str(runtime["error"])
        )

    st.divider()

    st.subheader("Trade Journal")

    if trades.empty:
        st.info(
            "No completed trades yet. "
            "This is expected until Athena records its first paper trade."
        )
    else:
        columns = [
            "timestamp",
            "instrument",
            "symbol",
            "option_type",
            "strike",
            "entry",
            "exit",
            "quantity",
            "pnl",
            "win",
            "exit_reason",
            "regime",
        ]
        visible = [
            column
            for column in columns
            if column in trades.columns
        ]

        st.dataframe(
            trades[visible].sort_values(
                "timestamp",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )


# ------------------------------------------------------------
# Wealth
# ------------------------------------------------------------

elif page == "Wealth":
    st.subheader("Athena Wealth")

    # Athena-specific wealth/performance view.
    # Portfolio remains the raw Dhan account view;
    # Trading remains the execution/signal view.
    # Wealth answers: "How is Athena performing financially?"

    trades = load_trades()
    metrics = calculate_trade_metrics(trades)
    portfolio = get_dhan_portfolio()
    funds = portfolio.get("funds", {})

    available_balance = float(
        funds.get("availabelBalance", 0.0) or 0.0
    )

    # Starting capital is the configured Athena capital anchor.
    starting_capital = float(
        os.getenv(
            "ATHENA_STARTING_CAPITAL",
            os.getenv("FALLBACK_CAPITAL", "500000"),
        )
        or 500000
    )

    realized_pnl = metrics["total_pnl"]
    current_equity = starting_capital + realized_pnl

    return_pct = (
        realized_pnl / starting_capital * 100
        if starting_capital > 0
        else 0.0
    )

    # Trade-level equity curve.
    peak_equity = starting_capital
    max_drawdown = 0.0

    if not trades.empty and "pnl" in trades.columns:
        pnl_series = pd.to_numeric(
            trades["pnl"],
            errors="coerce",
        ).fillna(0.0)

        equity_curve = starting_capital + pnl_series.cumsum()
        peak_curve = equity_curve.cummax()
        drawdowns = equity_curve - peak_curve

        peak_equity = float(
            max(starting_capital, equity_curve.max())
        )
        max_drawdown = float(drawdowns.min())

    # Today / month / YTD P&L.
    today_pnl = 0.0
    month_pnl = 0.0
    ytd_pnl = 0.0

    if not trades.empty and "timestamp" in trades.columns:
        dated = trades.copy()
        dated["_timestamp"] = pd.to_datetime(
            dated["timestamp"],
            errors="coerce",
        )
        dated["_pnl"] = pd.to_numeric(
            dated["pnl"],
            errors="coerce",
        ).fillna(0.0)

        now = datetime.now().astimezone()

        today_mask = dated["_timestamp"].dt.date == now.date()
        month_mask = (
            dated["_timestamp"].dt.year == now.year
        ) & (
            dated["_timestamp"].dt.month == now.month
        )
        ytd_mask = dated["_timestamp"].dt.year == now.year

        today_pnl = float(dated.loc[today_mask, "_pnl"].sum())
        month_pnl = float(dated.loc[month_mask, "_pnl"].sum())
        ytd_pnl = float(dated.loc[ytd_mask, "_pnl"].sum())

    # Capital deployment from current Dhan positions.
    positions = portfolio.get("positions", pd.DataFrame())

    deployed = 0.0
    unrealized_pnl = 0.0

    if not positions.empty:
        for column in (
            "buyValue",
            "buy_value",
            "costValue",
        ):
            if column in positions.columns:
                deployed = float(
                    pd.to_numeric(
                        positions[column],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                break

        for column in (
            "unrealizedProfit",
            "unrealized_pnl",
        ):
            if column in positions.columns:
                unrealized_pnl = float(
                    pd.to_numeric(
                        positions[column],
                        errors="coerce",
                    ).fillna(0).sum()
                )
                break

    utilization = (
        deployed / starting_capital * 100
        if starting_capital > 0
        else 0.0
    )

    # ────────────────────────────────────────────────────────
    # CAPITAL
    # ────────────────────────────────────────────────────────
    st.markdown("### Capital")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Starting Capital",
        f"₹{starting_capital:,.2f}",
    )
    c2.metric(
        "Current Equity",
        f"₹{current_equity:,.2f}",
        f"{realized_pnl:+,.2f}",
    )
    c3.metric(
        "Available Balance",
        f"₹{available_balance:,.2f}",
    )
    c4.metric(
        "Capital Deployed",
        f"₹{deployed:,.2f}",
        f"{utilization:.1f}% utilized",
    )

    st.divider()

    # ────────────────────────────────────────────────────────
    # PERFORMANCE
    # ────────────────────────────────────────────────────────
    st.markdown("### Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Realized P&L",
        f"₹{realized_pnl:,.2f}",
    )
    c2.metric(
        "Return",
        f"{return_pct:+.2f}%",
    )
    c3.metric(
        "Today's P&L",
        f"₹{today_pnl:,.2f}",
    )
    c4.metric(
        "Monthly P&L",
        f"₹{month_pnl:,.2f}",
    )

    # Daily portfolio objective — this is a portfolio-level target,
    # not the target of an individual option trade.
    daily_target = starting_capital * 0.01
    daily_loss_limit = starting_capital * 0.01
    daily_target_progress = (
        max(0.0, min(today_pnl / daily_target * 100.0, 100.0))
        if daily_target > 0 else 0.0
    )
    daily_loss_used = max(0.0, -today_pnl)
    daily_loss_remaining = max(0.0, daily_loss_limit - daily_loss_used)

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "YTD P&L",
        f"₹{ytd_pnl:,.2f}",
    )
    c6.metric(
        "Peak Equity",
        f"₹{peak_equity:,.2f}",
    )
    c7.metric(
        "Max Drawdown",
        f"₹{max_drawdown:,.2f}",
    )
    c8.metric(
        "Unrealized P&L",
        f"₹{unrealized_pnl:,.2f}",
    )

    st.divider()

    # ────────────────────────────────────────────────────────
    # TRADING QUALITY
    # ────────────────────────────────────────────────────────
    st.markdown("### Trading Quality")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Trades",
        metrics["total_trades"],
    )
    c2.metric(
        "Win Rate",
        f"{metrics['win_rate']:.1f}%",
    )
    c3.metric(
        "Profit Factor",
        (
            f"{metrics['profit_factor']:.2f}"
            if metrics["profit_factor"] > 0
            else "—"
        ),
    )

    expectancy = (
        metrics["win_rate"] / 100
        * metrics["average_win"]
        + (1 - metrics["win_rate"] / 100)
        * metrics["average_loss"]
    )

    c4.metric(
        "Expectancy / Trade",
        f"₹{expectancy:,.2f}",
    )

    c5, c6, c7, c8 = st.columns(4)

    largest_win = 0.0
    largest_loss = 0.0

    if not trades.empty and "pnl" in trades.columns:
        pnl_values = pd.to_numeric(
            trades["pnl"],
            errors="coerce",
        ).dropna()

        if not pnl_values.empty:
            largest_win = float(pnl_values.max())
            largest_loss = float(pnl_values.min())

    c5.metric(
        "Average Win",
        f"₹{metrics['average_win']:,.2f}",
    )
    c6.metric(
        "Average Loss",
        f"₹{metrics['average_loss']:,.2f}",
    )
    c7.metric(
        "Largest Win",
        f"₹{largest_win:,.2f}",
    )
    c8.metric(
        "Largest Loss",
        f"₹{largest_loss:,.2f}",
    )

    st.divider()

    # ────────────────────────────────────────────────────────
    # DAILY OBJECTIVE & RISK
    # ────────────────────────────────────────────────────────
    st.markdown("### Daily Objective")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Daily Objective", f"₹{daily_target:,.0f}")
    d2.metric("Today's Progress", f"{today_pnl:+,.2f}")
    d3.metric("Objective Progress", f"{daily_target_progress:.1f}%")
    d4.metric("Daily Loss Remaining", f"₹{daily_loss_remaining:,.2f}")

    st.progress(int(daily_target_progress))

    if today_pnl >= daily_target:
        st.success("🎯 Daily portfolio objective reached — Athena should stop opening new trades.")
    elif today_pnl <= -daily_loss_limit:
        st.error("🛑 Daily loss limit reached — Athena should stop opening new trades.")
    else:
        st.caption(
            f"Risk budget: ₹{daily_loss_limit:,.0f} max daily loss · "
            f"Current drawdown: ₹{max_drawdown:,.2f}"
        )

    st.divider()

    # ────────────────────────────────────────────────────────
    # EQUITY CURVE
    # ────────────────────────────────────────────────────────
    st.markdown("### Athena Equity Curve")

    if trades.empty:
        st.info(
            "The equity curve will appear after Athena records "
            "its first completed trade."
        )
    else:
        curve = trades.copy()
        curve["_pnl"] = pd.to_numeric(
            curve["pnl"],
            errors="coerce",
        ).fillna(0.0)

        curve["Athena Equity"] = (
            starting_capital + curve["_pnl"].cumsum()
        )

        if "timestamp" in curve.columns:
            curve["_time"] = pd.to_datetime(
                curve["timestamp"],
                errors="coerce",
            )
            curve = curve.sort_values("_time")

        chart = curve[["Athena Equity"]].reset_index(drop=True)
        chart.index = chart.index + 1
        chart.index.name = "Completed Trade"

        st.line_chart(chart)

        # Daily P&L view — more useful for evaluating Athena's
        # portfolio objective than trade-level equity alone.
        daily = curve.copy()
        if "_time" in daily.columns:
            daily["Date"] = daily["_time"].dt.date
            daily_pnl = daily.groupby("Date", as_index=True)["_pnl"].sum()
            daily_pnl.index = pd.to_datetime(daily_pnl.index)
            st.markdown("### Daily P&L")
            st.bar_chart(daily_pnl.rename("Daily P&L"))

            wins = int((daily_pnl > 0).sum())
            losses = int((daily_pnl < 0).sum())
            flats = int((daily_pnl == 0).sum())
            q1, q2, q3 = st.columns(3)
            q1.metric("Winning Days", wins)
            q2.metric("Losing Days", losses)
            q3.metric("Flat Days", flats)

    st.divider()

    st.caption(
        "Athena Wealth is intentionally trading-specific. "
        "It combines Athena's realized trade history with the "
        "current Dhan account state. It does not track salary, "
        "household expenses or unrelated personal finances."
    )


elif page == "Risk":
    st.subheader("Athena Risk Center")

    trades = load_trades()

    try:
        from config import (
            MAX_DAILY_LOSS_PCT,
            MAX_DRAWDOWN_PCT,
            RISK_PER_TRADE_PCT,
        )
    except Exception:
        MAX_DAILY_LOSS_PCT = 2.0
        MAX_DRAWDOWN_PCT = 10.0
        RISK_PER_TRADE_PCT = 1.0

    try:
        from config import LIVE_TRADING, FALLBACK_CAPITAL
    except Exception:
        LIVE_TRADING = False
        FALLBACK_CAPITAL = 500000

    if LIVE_TRADING:
        portfolio = get_dhan_portfolio()
        funds = portfolio.get("funds", {})
        risk_capital = float(
            funds.get("availabelBalance", 0.0) or 0.0
        )
        capital_source = "Dhan live balance"
    else:
        risk_capital = float(FALLBACK_CAPITAL)
        capital_source = "Athena paper capital"

    pnl = pd.Series(dtype=float)
    total_pnl = 0.0
    today_pnl = 0.0
    peak_equity = risk_capital
    current_drawdown = 0.0
    current_drawdown_pct = 0.0

    if not trades.empty and "pnl" in trades.columns:
        pnl = pd.to_numeric(
            trades["pnl"],
            errors="coerce",
        ).fillna(0.0)

        total_pnl = float(pnl.sum())

        if "timestamp" in trades.columns:
            dated = trades.copy()
            dated["_time"] = pd.to_datetime(
                dated["timestamp"],
                errors="coerce",
            )
            dated["_pnl"] = pnl

            now = datetime.now().astimezone()
            today_pnl = float(
                dated.loc[
                    dated["_time"].dt.date == now.date(),
                    "_pnl",
                ].sum()
            )

        equity = risk_capital + pnl.cumsum()
        peaks = equity.cummax()
        peak_equity = float(max(risk_capital, equity.max()))
        current_drawdown = float(equity.iloc[-1] - peaks.iloc[-1])

        if peak_equity > 0:
            current_drawdown_pct = (
                abs(current_drawdown) / peak_equity * 100
            )

    daily_loss_limit = risk_capital * float(
        MAX_DAILY_LOSS_PCT
    ) / 100

    drawdown_limit = risk_capital * float(
        MAX_DRAWDOWN_PCT
    ) / 100

    risk_per_trade = risk_capital * float(
        RISK_PER_TRADE_PCT
    ) / 100

    if today_pnl <= -daily_loss_limit:
        daily_status = "🔴 LIMIT BREACHED"
    elif today_pnl < 0:
        daily_status = "🟠 LOSS DAY"
    else:
        daily_status = "🟢 NORMAL"

    if current_drawdown_pct >= float(MAX_DRAWDOWN_PCT):
        dd_status = "🔴 LIMIT BREACHED"
    elif current_drawdown_pct >= float(MAX_DRAWDOWN_PCT) * 0.75:
        dd_status = "🟠 CAUTION"
    else:
        dd_status = "🟢 NORMAL"

    st.caption(f"Risk capital: {capital_source}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Capital", f"₹{risk_capital:,.2f}")
    c2.metric("Today's P&L", f"₹{today_pnl:,.2f}")
    c3.metric(
        "Drawdown",
        f"{current_drawdown_pct:.2f}%",
    )
    c4.metric(
        "Risk / Trade",
        f"₹{risk_per_trade:,.2f}",
    )

    st.divider()
    st.markdown("### Risk Limits")

    limits = pd.DataFrame(
        [
            [
                "Daily Loss",
                f"₹{daily_loss_limit:,.2f}",
                f"₹{abs(min(today_pnl, 0)):,.2f}",
                daily_status,
            ],
            [
                "Maximum Drawdown",
                f"₹{drawdown_limit:,.2f}",
                f"₹{abs(min(current_drawdown, 0)):,.2f}",
                dd_status,
            ],
            [
                "Risk Per Trade",
                f"₹{risk_per_trade:,.2f}",
                f"{float(RISK_PER_TRADE_PCT):.2f}%",
                "🟢 CONFIGURED",
            ],
        ],
        columns=["Limit", "Maximum", "Current", "Status"],
    )

    st.dataframe(
        limits,
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.markdown("### Trade Risk Quality")

    if pnl.empty:
        st.info(
            "Risk statistics will populate after Athena records "
            "completed trades."
        )
    else:
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_probability = float((pnl > 0).mean())
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0

        expectancy = (
            win_probability * avg_win
            + (1 - win_probability) * avg_loss
        )

        profit_factor = (
            float(wins.sum() / abs(losses.sum()))
            if not losses.empty and losses.sum() != 0
            else 0.0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Win Probability",
            f"{win_probability * 100:.1f}%",
        )
        c2.metric("Average Win", f"₹{avg_win:,.2f}")
        c3.metric("Average Loss", f"₹{avg_loss:,.2f}")
        c4.metric("Expectancy", f"₹{expectancy:,.2f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric(
            "Profit Factor",
            f"{profit_factor:.2f}"
            if profit_factor > 0
            else "—",
        )
        c6.metric(
            "Largest Loss",
            f"₹{float(pnl.min()):,.2f}",
        )
        c7.metric("Completed Trades", len(pnl))
        c8.metric(
            "Peak Equity",
            f"₹{peak_equity:,.2f}",
        )

    st.divider()
    st.markdown("### Athena Risk Engine History")

    risk_columns = [
        column
        for column in [
            "timestamp",
            "instrument",
            "regime",
            "risk_pct_at_entry",
            "risk_reward",
            "win_probability_at_entry",
            "expected_value_at_entry",
            "expected_value_per_risk",
            "barrier_target",
            "barrier_stop",
            "barrier_volatility",
        ]
        if column in trades.columns
    ]

    if risk_columns:
        st.dataframe(
            trades[risk_columns].sort_values(
                "timestamp",
                ascending=False,
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No entry-risk snapshots are available yet."
        )

    st.divider()

    if (
        today_pnl <= -daily_loss_limit
        or current_drawdown_pct >= float(MAX_DRAWDOWN_PCT)
    ):
        st.error(
            "ATHENA RISK STATE: DEFENSIVE — a configured risk "
            "limit has been reached."
        )
    elif (
        today_pnl <= -daily_loss_limit * 0.75
        or current_drawdown_pct >= float(MAX_DRAWDOWN_PCT) * 0.75
    ):
        st.warning(
            "ATHENA RISK STATE: CAUTION — risk is approaching "
            "a configured limit."
        )
    else:
        st.success(
            "ATHENA RISK STATE: NORMAL — no configured risk "
            "limit is currently breached."
        )


    st.divider()
    st.markdown("### Portfolio Risk Controls")

    # Athena's portfolio objective is +1% of the active risk capital.
    daily_target = risk_capital * 0.01
    monthly_loss_limit = risk_capital * 0.05
    daily_target_progress = (
        max(0.0, min(today_pnl / daily_target * 100.0, 100.0))
        if daily_target > 0 else 0.0
    )
    daily_loss_used = abs(min(today_pnl, 0.0))
    daily_loss_remaining = max(0.0, daily_loss_limit - daily_loss_used)
    risk_utilization = (
        daily_loss_used / daily_loss_limit * 100.0
        if daily_loss_limit > 0 else 0.0
    )

    if pnl.empty:
        monthly_pnl = 0.0
    else:
        if "timestamp" in trades.columns:
            times = pd.to_datetime(trades["timestamp"], errors="coerce")
            now = datetime.now().astimezone()
            monthly_pnl = float(
                pnl.loc[
                    (times.dt.year == now.year)
                    & (times.dt.month == now.month)
                ].sum()
            )
        else:
            monthly_pnl = 0.0

    monthly_loss_used = abs(min(monthly_pnl, 0.0))
    monthly_loss_remaining = max(
        0.0,
        monthly_loss_limit - monthly_loss_used,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Daily Objective",
        f"₹{daily_target:,.0f}",
        f"{daily_target_progress:.1f}% complete",
    )
    c2.metric(
        "Daily Loss Used",
        f"₹{daily_loss_used:,.2f}",
        f"{risk_utilization:.1f}% of limit",
    )
    c3.metric(
        "Daily Risk Remaining",
        f"₹{daily_loss_limit - daily_loss_used:,.2f}",
    )
    c4.metric(
        "Monthly Loss Remaining",
        f"₹{monthly_loss_remaining:,.2f}",
    )

    st.markdown("### Risk Utilization")
    st.progress(min(max(risk_utilization / 100.0, 0.0), 1.0))
    st.caption(
        f"Daily loss limit: ₹{daily_loss_limit:,.2f} · "
        f"Monthly loss limit: ₹{monthly_loss_limit:,.2f}"
    )

    st.markdown("### Risk-Adjusted Performance")

    if pnl.empty or len(pnl) < 2:
        st.info(
            "Sharpe/Sortino and streak statistics require at least "
            "two completed trades."
        )
    else:
        returns = pnl / risk_capital
        mean_return = float(returns.mean())
        std_return = float(returns.std(ddof=1))
        downside = returns[returns < 0]
        downside_std = (
            float(downside.std(ddof=1))
            if len(downside) > 1 else 0.0
        )

        # Trade-level ratios are descriptive only; they are not used to
        # approve or reject trades.
        sharpe = (
            mean_return / std_return * (len(returns) ** 0.5)
            if std_return > 0 else 0.0
        )
        sortino = (
            mean_return / downside_std * (len(returns) ** 0.5)
            if downside_std > 0 else 0.0
        )

        streak = 0
        max_loss_streak = 0
        for value in pnl.tolist():
            if value < 0:
                streak += 1
                max_loss_streak = max(max_loss_streak, streak)
            else:
                streak = 0

        avg_r = (
            float(pnl.mean() / risk_per_trade)
            if risk_per_trade > 0 else 0.0
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sharpe (Trade-Level)", f"{sharpe:.2f}")
        c2.metric("Sortino (Trade-Level)", f"{sortino:.2f}")
        c3.metric("Average R", f"{avg_r:+.2f}R")
        c4.metric("Max Loss Streak", max_loss_streak)

    st.markdown("### Circuit Breakers")
    breaker_rows = [
        [
            "Daily Profit Objective",
            f"+₹{daily_target:,.2f}",
            f"₹{today_pnl:,.2f}",
            "🟢 REACHED" if today_pnl >= daily_target else "🟡 ACTIVE",
        ],
        [
            "Daily Loss Limit",
            f"-₹{daily_loss_limit:,.2f}",
            f"₹{today_pnl:,.2f}",
            "🔴 STOP" if today_pnl <= -daily_loss_limit else "🟢 ACTIVE",
        ],
        [
            "Maximum Drawdown",
            f"-₹{drawdown_limit:,.2f}",
            f"₹{abs(min(current_drawdown, 0.0)):,.2f}",
            "🔴 STOP" if current_drawdown_pct >= float(MAX_DRAWDOWN_PCT) else "🟢 ACTIVE",
        ],
        [
            "Monthly Loss Limit",
            f"-₹{monthly_loss_limit:,.2f}",
            f"₹{monthly_loss_used:,.2f}",
            "🔴 STOP" if monthly_loss_used >= monthly_loss_limit else "🟢 ACTIVE",
        ],
    ]

    st.dataframe(
        pd.DataFrame(
            breaker_rows,
            columns=["Control", "Limit", "Current", "Status"],
        ),
        width="stretch",
        hide_index=True,
    )


elif page == "Analytics":
    st.subheader("Athena Analytics")
    st.caption("Performance intelligence from Athena's completed-trade history.")

    trades = load_trades().copy()

    if trades.empty:
        st.info(
            "Analytics will populate automatically after Athena records "
            "completed trades."
        )
    else:
        # --------------------------------------------------------
        # Normalize the persisted trade data once for all analytics.
        # --------------------------------------------------------
        trades["_pnl"] = pd.to_numeric(
            trades.get("pnl"), errors="coerce"
        ).fillna(0.0)

        if "timestamp" in trades.columns:
            trades["_time"] = pd.to_datetime(
                trades["timestamp"], errors="coerce"
            )
            trades = trades.sort_values("_time", na_position="last").reset_index(drop=True)
        else:
            trades["_time"] = pd.NaT

        pnl = trades["_pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        total = len(pnl)

        win_rate = float((pnl > 0).mean() * 100) if total else 0.0
        gross_profit = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        avg_win = float(wins.mean()) if not wins.empty else 0.0
        avg_loss = float(losses.mean()) if not losses.empty else 0.0
        expectancy = (
            (win_rate / 100) * avg_win
            + (1 - win_rate / 100) * avg_loss
        )

        # --------------------------------------------------------
        # Core performance
        # --------------------------------------------------------
        st.markdown("### Performance Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Completed Trades", total)
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor > 0 else "—")
        c4.metric("Expectancy / Trade", f"₹{expectancy:,.2f}")
        c5.metric("Net P&L", f"₹{float(pnl.sum()):,.2f}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Average Win", f"₹{avg_win:,.2f}")
        c2.metric("Average Loss", f"₹{avg_loss:,.2f}")
        c3.metric("Largest Win", f"₹{float(pnl.max()):,.2f}")
        c4.metric("Largest Loss", f"₹{float(pnl.min()):,.2f}")
        c5.metric("Win / Loss", f"{len(wins)} / {len(losses)}")

        st.divider()

        # --------------------------------------------------------
        # Equity curve + drawdown
        # --------------------------------------------------------
        st.markdown("### Equity & Drawdown")
        try:
            from config import FALLBACK_CAPITAL
            starting = float(FALLBACK_CAPITAL)
        except Exception:
            starting = 500000.0

        curve = starting + pnl.cumsum()
        peak = curve.cummax()
        drawdown = curve - peak

        left, right = st.columns(2)
        with left:
            equity_chart = pd.DataFrame({"Athena Equity": curve.values})
            equity_chart.index = range(1, len(equity_chart) + 1)
            equity_chart.index.name = "Completed Trade"
            st.line_chart(equity_chart)

        with right:
            dd_chart = pd.DataFrame({"Drawdown": drawdown.values})
            dd_chart.index = range(1, len(dd_chart) + 1)
            dd_chart.index.name = "Completed Trade"
            st.line_chart(dd_chart)

        st.divider()

        # --------------------------------------------------------
        # Daily / monthly performance
        # --------------------------------------------------------
        st.markdown("### Time Performance")
        if trades["_time"].notna().any():
            dated = trades.dropna(subset=["_time"]).copy()
            dated["Date"] = dated["_time"].dt.date
            dated["Month"] = dated["_time"].dt.to_period("M").astype(str)

            daily = dated.groupby("Date")["_pnl"].agg(
                PnL="sum", Trades="count"
            )
            monthly = dated.groupby("Month")["_pnl"].agg(
                PnL="sum", Trades="count"
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Daily P&L")
                st.line_chart(daily[["PnL"]])
            with c2:
                st.markdown("#### Monthly P&L")
                st.bar_chart(monthly[["PnL"]])

            winning_days = int((daily["PnL"] > 0).sum())
            losing_days = int((daily["PnL"] < 0).sum())
            flat_days = int((daily["PnL"] == 0).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Winning Days", winning_days)
            c2.metric("Losing Days", losing_days)
            c3.metric("Flat Days", flat_days)
        else:
            st.info("Time-based analytics will appear when trades contain timestamps.")

        st.divider()

        # --------------------------------------------------------
        # Instrument / option-side analysis
        # --------------------------------------------------------
        st.markdown("### Trade Breakdown")
        c1, c2 = st.columns(2)

        with c1:
            if "instrument" in trades.columns:
                by_instrument = (
                    trades.groupby(trades["instrument"].fillna("UNKNOWN"))["_pnl"]
                    .agg(Trades="count", PnL="sum", Avg="mean")
                    .sort_values("PnL", ascending=False)
                )
                st.markdown("#### By Instrument")
                st.dataframe(by_instrument, width="stretch")
            else:
                st.info("Instrument data unavailable.")

        with c2:
            if "option_type" in trades.columns:
                by_side = (
                    trades.groupby(trades["option_type"].fillna("UNKNOWN"))["_pnl"]
                    .agg(Trades="count", PnL="sum", Avg="mean")
                    .sort_values("PnL", ascending=False)
                )
                st.markdown("#### CE vs PE")
                st.dataframe(by_side, width="stretch")
            else:
                st.info("Option-side data unavailable.")

        if "regime" in trades.columns:
            st.markdown("#### By Market Regime")
            by_regime = (
                trades.groupby(trades["regime"].fillna("UNKNOWN"))["_pnl"]
                .agg(Trades="count", PnL="sum", Avg="mean")
                .sort_values("PnL", ascending=False)
            )
            st.dataframe(by_regime, width="stretch")

        st.divider()

        # --------------------------------------------------------
        # Holding time
        # --------------------------------------------------------
        if "holding_minutes" in trades.columns:
            holding = pd.to_numeric(
                trades["holding_minutes"], errors="coerce"
            ).dropna()
            if not holding.empty:
                st.markdown("### Holding Time")
                c1, c2, c3 = st.columns(3)
                c1.metric("Average Hold", f"{holding.mean():.1f} min")
                c2.metric("Median Hold", f"{holding.median():.1f} min")
                c3.metric("Longest Hold", f"{holding.max():.1f} min")

        # --------------------------------------------------------
        # Streak analysis
        # --------------------------------------------------------
        results = pnl.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).tolist()
        max_win_streak = max_loss_streak = 0
        current_win = current_loss = 0
        for result in results:
            if result > 0:
                current_win += 1
                current_loss = 0
            elif result < 0:
                current_loss += 1
                current_win = 0
            else:
                current_win = current_loss = 0
            max_win_streak = max(max_win_streak, current_win)
            max_loss_streak = max(max_loss_streak, current_loss)

        st.markdown("### Streaks")
        c1, c2 = st.columns(2)
        c1.metric("Max Winning Streak", max_win_streak)
        c2.metric("Max Losing Streak", max_loss_streak)

        st.divider()

        # --------------------------------------------------------
        # P&L distribution
        # --------------------------------------------------------
        st.markdown("### P&L Distribution")
        distribution = pd.DataFrame({"P&L": pnl.values})
        st.bar_chart(distribution)

        st.caption(
            "Analytics are descriptive only. They do not modify Athena's "
            "signals, risk limits or order execution."
        )


elif page == "ML":
    st.subheader("Athena ML Center")
    st.caption("Read-only model diagnostics. This dashboard does not train or modify the ML model.")

    trades = load_trades()
    try:
        from config import MODEL_PATH
        model_path = Path(MODEL_PATH)
    except Exception:
        model_path = Path("athena_xgb_model.json")

    model_exists = model_path.exists()
    model_size = model_path.stat().st_size if model_exists else 0

    ml_history = read_athena_table("ml_history")
    sample_count = len(ml_history)
    completed_count = len(trades)

    # Model metadata / availability
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("XGBoost Model", "AVAILABLE" if model_exists else "NOT FOUND")
    c2.metric("ML Samples", sample_count)
    c3.metric("Completed Trades", completed_count)
    c4.metric("Model File", f"{model_size / 1024:.1f} KB" if model_exists else "—")

    if not model_exists:
        st.warning(
            f"XGBoost model file not found at '{model_path}'. "
            "Athena can still run, but model diagnostics are unavailable."
        )
    else:
        st.success("XGBoost model file detected. Dashboard is read-only.")

    st.divider()
    st.markdown("### Model Readiness")

    readiness = []
    readiness.append(["Model artifact", "READY" if model_exists else "MISSING"])
    readiness.append(["ML history", "AVAILABLE" if sample_count > 0 else "NO DATA"])
    readiness.append(["Completed trade outcomes", "AVAILABLE" if completed_count > 0 else "NO DATA"])
    readiness.append([
        "Performance evaluation",
        "READY" if completed_count >= 20 else "INSUFFICIENT DATA",
    ])

    st.dataframe(
        pd.DataFrame(readiness, columns=["Component", "Status"]),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    st.markdown("### Prediction / Outcome Diagnostics")

    # The database stores entry-time probabilities. Evaluate them only when
    # actual completed outcomes are present; never manufacture accuracy.
    probability_col = "win_probability_at_entry"
    if probability_col in trades.columns and "win" in trades.columns:
        diagnostic = trades[[probability_col, "win"]].copy()
        diagnostic[probability_col] = pd.to_numeric(
            diagnostic[probability_col], errors="coerce"
        )
        diagnostic["win"] = pd.to_numeric(
            diagnostic["win"], errors="coerce"
        )
        diagnostic = diagnostic.dropna()

        if not diagnostic.empty:
            predicted = diagnostic[probability_col] >= 0.5
            actual = diagnostic["win"] >= 1
            accuracy = float((predicted == actual).mean() * 100)
            brier = float(((diagnostic[probability_col] - actual.astype(float)) ** 2).mean())

            c1, c2, c3 = st.columns(3)
            c1.metric("Directional Accuracy", f"{accuracy:.1f}%")
            c2.metric("Mean Brier Score", f"{brier:.4f}")
            c3.metric("Evaluated Predictions", len(diagnostic))

            plot = diagnostic.rename(
                columns={probability_col: "Predicted Probability"}
            )
            plot["Actual Win"] = actual.astype(int).values
            st.line_chart(plot[["Predicted Probability", "Actual Win"]].reset_index(drop=True))
        else:
            st.info("Prediction history exists only after Athena records entry probabilities with completed outcomes.")
    else:
        st.info("No completed prediction/outcome dataset is available yet.")

    st.divider()
    st.markdown("### Feature Importance")

    if model_exists:
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            importance = model.feature_importances_
            names = list(getattr(model, "feature_names_in_", []))
            if not names or len(names) != len(importance):
                names = [f"Feature {i + 1}" for i in range(len(importance))]

            feature_df = pd.DataFrame({
                "Feature": names,
                "Importance": importance,
            }).sort_values("Importance", ascending=False)

            st.dataframe(feature_df, width="stretch", hide_index=True)
            st.bar_chart(feature_df.set_index("Feature").head(15))
        except Exception as exc:
            st.info(f"Model detected, but feature importance could not be read: {exc}")
    else:
        st.info("Feature importance will appear when the Athena XGBoost model artifact is available.")

    st.divider()
    st.markdown("### ML History")

    if ml_history.empty:
        st.info("ML history will populate automatically as Athena records completed trades.")
    else:
        columns = [
            c for c in [
                "created_at",
                "trade_id",
                "payload",
            ] if c in ml_history.columns
        ]
        st.dataframe(
            ml_history[columns].sort_values(
                "created_at", ascending=False
            ) if "created_at" in columns else ml_history[columns],
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "ML analytics are descriptive only. The dashboard never trains, replaces, "
        "or modifies Athena's production model."
    )


elif page == "Goals":
    st.subheader("Financial Goals")

    goal_name = st.text_input(
        "Goal",
        placeholder="e.g. ₹1 Crore Net Worth",
    )

    c1, c2, c3 = st.columns(3)
    current = c1.number_input(
        "Current Value",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )
    target = c2.number_input(
        "Target Value",
        min_value=0.0,
        value=10000000.0,
        step=100000.0,
    )
    monthly = c3.number_input(
        "Monthly Investment",
        min_value=0.0,
        value=0.0,
        step=1000.0,
    )

    if target > 0:
        progress = min(current / target, 1.0)
        st.progress(progress)
        st.write(f"Progress: **{progress:.1%}**")

    st.caption(
        "Goal projections and required CAGR will be added "
        "after the wealth database is connected."
    )


# ------------------------------------------------------------
# Transactions
# ------------------------------------------------------------

elif page == "Transactions":
    st.subheader("Transactions")

    with st.form("transaction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            date = st.date_input("Date")
            tx_type = st.selectbox(
                "Type",
                [
                    "BUY",
                    "SELL",
                    "DIVIDEND",
                    "DEPOSIT",
                    "WITHDRAWAL",
                    "EXPENSE",
                ],
            )

        with c2:
            asset = st.text_input("Asset")
            broker = st.text_input("Broker")

        with c3:
            quantity = st.number_input(
                "Quantity",
                min_value=0.0,
                value=0.0,
            )
            price = st.number_input(
                "Price / Amount",
                min_value=0.0,
                value=0.0,
            )

        submitted = st.form_submit_button("Add Transaction")

        if submitted:
            amount = quantity * price

            new_row = pd.DataFrame(
                [
                    {
                        "Date": date.isoformat(),
                        "Type": tx_type,
                        "Asset": asset,
                        "Quantity": quantity,
                        "Price": price,
                        "Amount": amount,
                        "Broker": broker,
                    }
                ]
            )

            st.session_state.transactions = pd.concat(
                [st.session_state.transactions, new_row],
                ignore_index=True,
            )

            st.success("Transaction added to the current dashboard session.")

    st.dataframe(
        st.session_state.transactions,
        width="stretch",
        hide_index=True,
    )


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

elif page == "System":
    st.subheader("Athena System Health")
    st.caption("Read-only diagnostics. This page never places, modifies or cancels orders.")

    now = datetime.now().astimezone()

    # Configuration / mode
    try:
        from config import (
            LIVE_TRADING,
            DATABASE_PATH,
            MODEL_PATH,
            CLIENT_ID,
            ACCESS_TOKEN,
        )
    except Exception:
        LIVE_TRADING = False
        DATABASE_PATH = "data/athena.db"
        MODEL_PATH = "athena_xgb_model.json"
        CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
        ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

    db_path = Path(DATABASE_PATH)
    model_path = Path(MODEL_PATH)

    # Dhan API health
    dhan_configured = bool(CLIENT_ID and ACCESS_TOKEN)
    dhan_connected = False
    dhan_error = ""
    funds = {}
    if dhan_configured:
        try:
            portfolio = get_dhan_portfolio()
            dhan_connected = bool(portfolio.get("connected")) and not portfolio.get("errors")
            funds = portfolio.get("funds", {}) or {}
            if portfolio.get("errors"):
                dhan_error = "; ".join(portfolio["errors"])
        except Exception as exc:
            dhan_error = str(exc)

    # WebSocket / market-feed health
    try:
        snapshot = get_market_snapshot() or {}
    except Exception as exc:
        snapshot = {"error": str(exc)}

    feed_error = snapshot.get("error")
    feed_time = snapshot.get("timestamp") or snapshot.get("updated_at")
    nifty = snapshot.get("NIFTY", {}) if isinstance(snapshot, dict) else {}
    banknifty = snapshot.get("BANKNIFTY", {}) if isinstance(snapshot, dict) else {}

    def _feed_ok(item):
        if not isinstance(item, dict):
            return False
        ltp = item.get("ltp")
        return ltp not in (None, 0, "") and not item.get("error")

    nifty_ok = _feed_ok(nifty)
    banknifty_ok = _feed_ok(banknifty)
    websocket_ok = nifty_ok or banknifty_ok

    # Database health
    db_ok = db_path.exists()
    db_size = db_path.stat().st_size if db_ok else 0
    trade_count = 0
    try:
        if db_ok:
            with sqlite3.connect(str(db_path), timeout=5) as conn:
                trade_count = int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
                conn.execute("SELECT 1")
    except Exception as exc:
        db_ok = False
        db_error = str(exc)
    else:
        db_error = ""

    # ML health
    model_ok = model_path.exists() and model_path.stat().st_size > 0
    ml_history = read_athena_table("ml_history")

    # Telegram is configuration-only here; never expose the token.
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_ok = bool(telegram_token and telegram_chat)

    def status(ok, good="🟢 HEALTHY", bad="🔴 UNAVAILABLE"):
        return good if ok else bad

    st.markdown("### System Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading Mode", "🔴 LIVE" if LIVE_TRADING else "🟡 PAPER")
    c2.metric("Dhan API", status(dhan_connected if dhan_configured else False))
    c3.metric("WebSocket", status(websocket_ok, "🟢 LIVE", "🟠 WAITING"))
    c4.metric("Database", status(db_ok))

    st.divider()
    st.markdown("### Data Feeds")
    c1, c2, c3 = st.columns(3)
    c1.metric("NIFTY Feed", status(nifty_ok, "🟢 LIVE", "🔴 STALE / OFFLINE"))
    c2.metric("BANKNIFTY Feed", status(banknifty_ok, "🟢 LIVE", "🔴 STALE / OFFLINE"))
    c3.metric("Last Feed Update", str(feed_time or "Not available"))

    if feed_error:
        st.warning(f"WebSocket: {feed_error}")

    st.divider()
    st.markdown("### Athena Services")
    services = pd.DataFrame([
        ["Dhan API", status(dhan_connected if dhan_configured else False), "Credentials configured" if dhan_configured else "Credentials missing"],
        ["Dhan WebSocket", status(websocket_ok, "LIVE", "WAITING / OFFLINE"), "Live market snapshot" if websocket_ok else "No valid live snapshot"],
        ["SQLite", status(db_ok), f"{trade_count} completed trades" if db_ok else db_error],
        ["XGBoost", status(model_ok, "AVAILABLE", "NOT FOUND"), f"{model_path} · {model_path.stat().st_size:,} bytes" if model_ok else str(model_path)],
        ["ML History", status(len(ml_history) > 0, "POPULATED", "EMPTY"), f"{len(ml_history)} samples"],
        ["Telegram", status(telegram_ok, "CONFIGURED", "DISABLED / NOT CONFIGURED"), "Credentials present" if telegram_ok else "Alerts unavailable"],
    ], columns=["Service", "Status", "Details"])
    st.dataframe(services, width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Runtime")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completed Trades", trade_count)
    c2.metric("ML Samples", len(ml_history))
    c3.metric("Dhan Available", f"₹{float(funds.get('availabelBalance', 0.0) or 0.0):,.2f}" if funds else "—")
    c4.metric("Dashboard Time", now.strftime("%H:%M:%S"))

    st.caption(f"Database: {db_path.resolve()}")
    st.caption(f"Model: {model_path.resolve()}")
    st.caption("System diagnostics are read-only. app.py does not execute trading orders.")


elif page == "Settings":
    st.subheader("Settings")

    st.write("Trading")

    st.checkbox(
        "Live trading",
        value=False,
        disabled=True,
        help="Live trading remains controlled by config.py.",
    )

    st.checkbox(
        "Telegram alerts",
        value=True,
        disabled=True,
    )

    st.write("Dashboard")

    st.selectbox(
        "Currency",
        ["INR"],
        disabled=True,
    )

    st.caption(
        f"Dashboard started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    st.divider()

    st.warning(
        "This dashboard is currently a UI shell. "
        "No live orders, broker mutations, or trading decisions "
        "are performed by app.py."
    )