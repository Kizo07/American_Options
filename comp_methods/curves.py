from dataclasses import dataclass

import numpy as np

from ._validation import validate_choice


@dataclass(frozen=True)
class FlatCurve:
    rate: float

    def zero_rate(self, t):
        return np.full_like(np.asarray(t, dtype=float), self.rate, dtype=float)

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        return np.exp(-self.rate * t)

    def forward_rate(self, t1, t2):
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        return self.rate


@dataclass(frozen=True)
class ZeroCurve:
    times: object
    rates: object
    compounding: str = "continuous"

    def __post_init__(self):
        times = np.asarray(self.times, dtype=float)
        rates = np.asarray(self.rates, dtype=float)
        if times.ndim != 1 or rates.ndim != 1 or len(times) != len(rates):
            raise ValueError("times and rates must be one-dimensional arrays of equal length")
        if len(times) == 0 or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("times must be non-empty, non-negative, and sorted")
        validate_choice("compounding", self.compounding, {"continuous", "annual"})
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "rates", rates)

    def zero_rate(self, t):
        return np.interp(t, self.times, self.rates, left=self.rates[0], right=self.rates[-1])

    def discount(self, t):
        t = np.asarray(t, dtype=float)
        rates = self.zero_rate(t)
        if self.compounding == "continuous":
            return np.exp(-rates * t)
        return (1 + rates) ** (-t)

    def forward_rate(self, t1, t2):
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        d1 = float(self.discount(t1))
        d2 = float(self.discount(t2))
        return -np.log(d2 / d1) / (t2 - t1)


def as_flat_rate(rate_or_curve, t):
    if hasattr(rate_or_curve, "zero_rate"):
        return float(np.asarray(rate_or_curve.zero_rate(t)))
    return float(rate_or_curve)


def discount_factor(rate_or_curve, t):
    if hasattr(rate_or_curve, "discount"):
        return rate_or_curve.discount(t)
    return np.exp(-float(rate_or_curve) * np.asarray(t, dtype=float))
