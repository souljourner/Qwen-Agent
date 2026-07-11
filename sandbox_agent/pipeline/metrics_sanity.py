"""Consistency-based sanity validation for trading-strategy metrics.

Philosophy: reject INTERNAL CONTRADICTIONS, not surprising numbers. Short and
leveraged strategies legitimately produce drawdowns beyond -100% (account
equity can go negative in a squeeze), so naive bounds would false-reject real
results. What is always invalid:

- NaN / inf anywhere
- declared stats that contradict the provided equity curve
- drawdown < -100% while the equity curve never goes <= 0
- trading activity AFTER the equity curve hit <= 0 (real accounts are
  liquidated — post-blowup trades are the ledger bug behind the observed
  -235%-drawdown runs)
- for claims WITHOUT an equity curve to verify against: crude configurable
  ceilings (the only line of defense against fabricated numbers)

Units convention: keys ending in `_pct` are percentages (dd_pct=-35.0 means
-35%); `total_return` / `max_drawdown` are fractions (-0.35). Both appear in
the wild: pilot_history rows use _pct, backtest/full/metrics.json is mixed.
"""

import hashlib
import json
import math
import os
from typing import List, Optional

# Crude ceilings — applied ONLY when no equity curve is present to verify
# against. Env-overridable for exotic strategies.
RETURN_PCT_CEILING = float(os.environ.get("METRICS_RETURN_PCT_CEILING", 10_000))   # 100x
DD_PCT_FLOOR = float(os.environ.get("METRICS_DD_PCT_FLOOR", -1_000))               # -10x

_CONSISTENCY_TOL = 0.01  # declared vs recomputed, relative


def _walk_numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        yield path, obj


def _parse_curve(curve) -> Optional[List[tuple]]:
    """Normalize equity_curve into [(ts_or_index, equity), ...] or None."""
    if not isinstance(curve, list) or len(curve) < 2:
        return None
    out = []
    for i, item in enumerate(curve):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            out.append((item[0], float(item[1])))
        elif isinstance(item, dict) and "equity" in item:
            out.append((item.get("ts", i), float(item["equity"])))
        elif isinstance(item, (int, float)):
            out.append((i, float(item)))
        else:
            return None
    return out


def _max_drawdown(equities: List[float]) -> float:
    """Peak-to-trough drawdown as a fraction of the peak. Can be < -1 when
    equity goes negative (short blowup) — that's a real number, not an error."""
    peak = equities[0]
    worst = 0.0
    for e in equities:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (e - peak) / peak
            if dd < worst:
                worst = dd
    return worst


def validate_metrics(m: dict, strategy_type: str = "long_short") -> List[str]:
    """Return a list of violation strings (empty = sane). May stamp
    `blown_up: True` onto `m` when the equity curve crosses <= 0."""
    errs: List[str] = []

    # 1. NaN/inf anywhere — always invalid.
    for path, v in _walk_numbers(m):
        if math.isnan(v) or math.isinf(v):
            errs.append(f"non-finite value at {path}")
    if errs:
        return errs

    curve = _parse_curve(m.get("equity_curve"))

    # Collect declared values in FRACTION units.
    def _declared(frac_key: str, pct_key: str) -> Optional[float]:
        if m.get(frac_key) is not None:
            try:
                return float(m[frac_key])
            except (TypeError, ValueError):
                return None
        if m.get(pct_key) is not None:
            try:
                return float(m[pct_key]) / 100.0
            except (TypeError, ValueError):
                return None
        return None

    declared_return = _declared("total_return", "return_pct")
    declared_dd = _declared("max_drawdown", "dd_pct")

    if declared_dd is not None and declared_dd > 0:
        errs.append(f"max drawdown must be <= 0, got {declared_dd:.2%}")

    if curve is not None:
        # --- strong path: internal consistency against the curve ---
        timestamps = [ts for ts, _ in curve]
        if any(str(timestamps[i]) > str(timestamps[i + 1]) for i in range(len(timestamps) - 1)):
            errs.append("equity_curve timestamps are not monotonically increasing")
            return errs

        equities = [e for _, e in curve]
        blowup_idx = next((i for i, e in enumerate(equities) if e <= 0), None)
        if blowup_idx is not None:
            m["blown_up"] = True
            # Trades after the blowup point = ledger kept trading past
            # liquidation — the actual accounting bug.
            blowup_ts = str(curve[blowup_idx][0])
            for t in m.get("trades_log") or []:
                ts = str(t.get("ts", ""))
                if ts and ts > blowup_ts:
                    errs.append(
                        f"trade at {ts} occurs AFTER equity hit <= 0 at {blowup_ts} — "
                        f"real accounts are liquidated at blowup; post-blowup trading "
                        f"invalidates the ledger")
                    break

        first, last = equities[0], equities[-1]
        if first > 0 and declared_return is not None:
            recomputed = (last - first) / first
            denom = max(abs(recomputed), 1e-9)
            if abs(recomputed - declared_return) / denom > _CONSISTENCY_TOL and \
               abs(recomputed - declared_return) > 0.005:
                errs.append(
                    f"declared return {declared_return:.2%} does not match the equity "
                    f"curve ({recomputed:.2%}) — metrics/curve mismatch")

        recomputed_dd = _max_drawdown(equities)
        if declared_dd is not None:
            denom = max(abs(recomputed_dd), 1e-9)
            if abs(recomputed_dd - declared_dd) / denom > 0.05 and \
               abs(recomputed_dd - declared_dd) > 0.01:
                errs.append(
                    f"declared drawdown {declared_dd:.2%} does not match the equity "
                    f"curve ({recomputed_dd:.2%})")
        if declared_dd is not None and declared_dd < -1.0 and blowup_idx is None:
            errs.append(
                f"drawdown {declared_dd:.2%} is beyond -100% but the equity curve "
                f"never reaches <= 0 — contradicts the curve")
    else:
        # --- weak path: unverifiable claims get crude ceilings ---
        if declared_return is not None and abs(declared_return) * 100 > RETURN_PCT_CEILING:
            errs.append(
                f"declared return {declared_return:.0%} exceeds the sanity ceiling "
                f"({RETURN_PCT_CEILING:.0f}%) and no equity_curve is provided to verify it")
        if declared_dd is not None and declared_dd * 100 < DD_PCT_FLOOR:
            errs.append(
                f"declared drawdown {declared_dd:.0%} is beyond the sanity floor "
                f"({DD_PCT_FLOOR:.0f}%) and no equity_curve is provided to verify it")

    # Long-only strategies cannot lose more than 100% (no negative equity).
    if strategy_type == "long_only" and declared_dd is not None and declared_dd < -1.0:
        errs.append(
            f"drawdown {declared_dd:.2%} beyond -100% is impossible for a long_only strategy")

    return errs


def canonical_hash(m: dict) -> str:
    """Stable content hash over the metrics dict, excluding the hash field
    itself (so stamping is idempotent)."""
    body = {k: v for k, v in m.items() if k != "content_hash"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def stamp_metrics_file(path: str, strategy_version: str) -> dict:
    """Evaluator-owned stamping: inject schema/version/hash fields and rewrite
    the file. The agent never computes these — the evaluator does, after gates
    pass, which is what makes the stage-4 hash pin trustworthy."""
    from datetime import datetime
    with open(path) as f:
        m = json.load(f)
    m["schema_version"] = 1
    m["strategy_version"] = strategy_version
    m["stamped_at"] = datetime.now().isoformat()
    m["content_hash"] = canonical_hash(m)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=2, default=str)
    os.replace(tmp, path)
    return m
