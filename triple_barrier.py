"""Athena path-aware triple-barrier trade engine."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class BarrierSet:
    entry: float
    take_profit: float
    stop_loss: float
    expiry: datetime

@dataclass
class BarrierResult:
    event: str
    exit_price: float
    timestamp: datetime
    pnl_per_unit: float

def build_barriers(entry, volatility, reward_multiple=1.5, risk_multiple=1.0,
                   max_hold_minutes=30, direction="LONG"):
    entry=float(entry); volatility=float(volatility)
    if entry<=0 or volatility<=0: raise ValueError("entry/volatility must be positive")
    reward_distance=volatility*float(reward_multiple)
    risk_distance=volatility*float(risk_multiple)
    direction=str(direction).upper()
    if direction=="LONG":
        tp,sl=entry+reward_distance,entry-risk_distance
    elif direction=="SHORT":
        tp,sl=entry-reward_distance,entry+risk_distance
    else: raise ValueError("direction must be LONG or SHORT")
    if sl<=0: raise ValueError("stop must be positive")
    return BarrierSet(entry,tp,sl,datetime.now()+timedelta(minutes=int(max_hold_minutes)))

def check_barriers(price, barriers: BarrierSet, timestamp: Optional[datetime]=None):
    price=float(price); timestamp=timestamp or datetime.now()
    if price<=0: return None
    if price>=barriers.take_profit:
        return BarrierResult("TAKE_PROFIT",price,timestamp,price-barriers.entry)
    if price<=barriers.stop_loss:
        return BarrierResult("STOP_LOSS",price,timestamp,price-barriers.entry)
    if timestamp>=barriers.expiry:
        return BarrierResult("TIME_EXIT",price,timestamp,price-barriers.entry)
    return None

def label_path(prices, entry, volatility, reward_multiple=1.5,
               risk_multiple=1.0, max_bars=30, direction="LONG"):
    """Path-aware ML label: PT first=1, otherwise SL/TIME=0."""
    entry=float(entry); volatility=float(volatility)
    if entry<=0 or volatility<=0: raise ValueError("entry/volatility must be positive")
    direction=str(direction).upper()
    pt=entry+volatility*reward_multiple if direction=="LONG" else entry-volatility*reward_multiple
    sl=entry-volatility*risk_multiple if direction=="LONG" else entry+volatility*risk_multiple
    observed=list(prices)[:max_bars]
    for i,price in enumerate(observed):
        price=float(price)
        if direction=="LONG":
            if price>=pt: return {"label":1,"event":"TAKE_PROFIT","bar_index":i,"exit_price":price}
            if price<=sl: return {"label":0,"event":"STOP_LOSS","bar_index":i,"exit_price":price}
        else:
            if price<=pt: return {"label":1,"event":"TAKE_PROFIT","bar_index":i,"exit_price":price}
            if price>=sl: return {"label":0,"event":"STOP_LOSS","bar_index":i,"exit_price":price}
    final=float(observed[-1]) if observed else entry
    favorable=final>entry if direction=="LONG" else final<entry
    return {"label":1 if favorable else 0,"event":"TIME_EXIT","bar_index":max(0,len(observed)-1),"exit_price":final}
