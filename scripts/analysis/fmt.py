"""Number formatting helpers: fmt_pct rounds half up and writes a middle dot as the decimal mark and a true minus sign, and fmt_span formats a range from two such numbers. It reads and writes nothing."""
import math
from decimal import ROUND_HALF_UP, Decimal

MID, MINUS, NDASH = "·", "−", "–"


def _q(x, dp):
    if not math.isfinite(x):
        raise ValueError("not a finite number: %r" % (x,))
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP)


def fmt_pct(x, dp=1):
    q = _q(x, dp)
    return (MINUS if q < 0 else "") + format(abs(q), "f").replace(".", MID)


def fmt_span(lo, hi, dp=1):
    a, b = fmt_pct(lo, dp), fmt_pct(hi, dp)
    return a + (" to " if MINUS in a + b else NDASH) + b
