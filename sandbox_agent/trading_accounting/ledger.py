"""Position ledger with a proper long/short margin model and halt-at-blowup.

Equity = cash + Σ position_qty × mark_price. Short positions (negative qty)
add sale proceeds to cash and subtract the (possibly rising) buyback cost at
mark time — so equity CAN go negative in a squeeze. That is a legitimate
result to REPORT (drawdowns beyond -100% are real for shorts), but a real
account is liquidated at that point: once equity marks <= 0 the ledger sets
`blown_up` and refuses further fills. Post-blowup trading is the accounting
bug behind historical -235%-drawdown pipeline runs; this makes it impossible.

Pure stdlib — no third-party deps, so it imports in any project venv.
"""

from typing import Dict, List, Optional, Tuple


class LedgerBlownUp(RuntimeError):
    """Raised on fill() after equity has marked <= 0 (account liquidated)."""


class Ledger:
    def __init__(self, cash: float = 100_000.0):
        if cash <= 0:
            raise ValueError("starting cash must be positive")
        self.starting_cash = float(cash)
        self.cash = float(cash)
        self.positions: Dict[str, float] = {}
        self.blown_up = False
        self._curve: List[Tuple[str, float]] = []
        self._trades: List[dict] = []

    def fill(self, ts, symbol: str, qty: float, price: float,
             commission: float = 0.0) -> None:
        """Execute a fill. qty > 0 buys/covers, qty < 0 sells/shorts."""
        if self.blown_up:
            raise LedgerBlownUp(
                f"account blew up (equity <= 0) before {ts} — a real account is "
                f"liquidated; no further trading is possible")
        if price <= 0:
            raise ValueError(f"fill price must be positive, got {price}")
        self.cash -= qty * price + commission
        self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
        if abs(self.positions[symbol]) < 1e-9:
            del self.positions[symbol]
        self._trades.append({"ts": str(ts), "symbol": symbol, "qty": qty,
                             "price": price, "commission": commission})

    def mark(self, ts, prices: Dict[str, float]) -> float:
        """Mark-to-market all open positions and record an equity point.
        Call once per bar with that bar's closing prices."""
        equity = self.cash
        for symbol, qty in self.positions.items():
            px = prices.get(symbol)
            if px is None:
                raise ValueError(
                    f"mark({ts}): no price provided for open position {symbol!r}")
            equity += qty * px
        self._curve.append((str(ts), equity))
        if equity <= 0:
            self.blown_up = True
        return equity

    def equity_curve(self) -> List[Tuple[str, float]]:
        return list(self._curve)

    @property
    def trades(self) -> List[dict]:
        return list(self._trades)
