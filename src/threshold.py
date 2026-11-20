"""Alert-budget threshold selection.

Why not maximise F1? In an ICU the scarce resource is *attention*, not labels.
A model that alerts on 40% of patient-days will be muted, and a muted alarm has
zero sensitivity regardless of its AUROC. So the operating point is chosen on
**validation** by:

    maximise sensitivity  subject to  alerts_per_patient_day <= budget

with hysteresis so a single noisy hour cannot fire an alert, and a refractory
period so the same deterioration is not paged every hour.

If no threshold meets the budget, this module raises — it does not silently
return a threshold that violates the clinical constraint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    ALERT_BUDGET_PER_PATIENT_DAY,
    EARLY_WINDOW_HOURS,
    ON_TIME_WINDOW_HOURS,
    PERSISTENCE_HOURS,
    REFRACTORY_HOURS,
    THRESHOLD_GRID,
)


class ThresholdError(ValueError):
    """Base class for threshold-selection failures."""


class InfeasibleAlertBudget(ThresholdError):
    """No threshold in the grid respects the alert budget."""


@dataclass(frozen=True)
class AlertPolicy:
    threshold: float
    refractory_hours: int = REFRACTORY_HOURS
    persistence_hours: int = PERSISTENCE_HOURS


@dataclass(frozen=True)
class PolicyReport:
    threshold: float
    sensitivity: float
    alerts_per_patient_day: float
    median_lead_hours: float | None
    n_patients: int
    n_eligible: int
    n_alerts: int
    n_early_detected: int
    n_on_time: int
    n_late: int
    n_missed: int
    patient_days: float

    def as_dict(self) -> dict:
        return {
            "threshold": round(self.threshold, 4),
            "sensitivity": round(self.sensitivity, 4),
            "alerts_per_patient_day": round(self.alerts_per_patient_day, 4),
            "median_lead_hours": (
                None if self.median_lead_hours is None else round(self.median_lead_hours, 2)
            ),
            "n_patients": self.n_patients,
            "n_eligible": self.n_eligible,
            "n_alerts": self.n_alerts,
            "n_early_detected": self.n_early_detected,
            "n_on_time": self.n_on_time,
            "n_late": self.n_late,
            "n_missed": self.n_missed,
            "patient_days": round(self.patient_days, 2),
        }


@dataclass(frozen=True)
class _Stream:
    hours: np.ndarray
    probs: np.ndarray
    t_sepsis: int | None
    eligible: bool


def _build_streams(frame: pd.DataFrame) -> tuple[list[_Stream], int]:
    required = {"patient_id", "icu_hour", "p", "t_sepsis_est", "is_sepsis_patient"}
    missing = required - set(frame.columns)
    if missing:
        raise ThresholdError(f"missing columns: {sorted(missing)}")
    if frame["p"].isna().any():
        raise ThresholdError("calibrated probabilities contain NaN")
    if ((frame["p"] < 0) | (frame["p"] > 1)).any():
        raise ThresholdError("probabilities must be in [0, 1]")

    has_censored = "onset_censored" in frame.columns
    streams: list[_Stream] = []
    for _, block in frame.sort_values(["patient_id", "icu_hour"]).groupby(
        "patient_id", sort=True
    ):
        t_raw = block["t_sepsis_est"].dropna().unique()
        t_sepsis = int(t_raw[0]) if len(t_raw) else None
        censored = bool(block["onset_censored"].iloc[0]) if has_censored else False
        eligible = bool(block["is_sepsis_patient"].iloc[0]) and t_sepsis is not None and not censored
        streams.append(
            _Stream(
                hours=block["icu_hour"].to_numpy(dtype=np.int64),
                probs=block["p"].to_numpy(dtype=np.float64),
                t_sepsis=t_sepsis,
                eligible=eligible,
            )
        )
    return streams, len(frame)


def alert_hours(stream: _Stream, policy: AlertPolicy) -> list[int]:
    """Fire at most one alert per refractory window, after persistence holds."""
    alerts: list[int] = []
    streak = 0
    last_alert: int | None = None
    for hour, prob in zip(stream.hours, stream.probs):
        if prob >= policy.threshold:
            streak += 1
            if streak >= policy.persistence_hours and (
                last_alert is None or hour - last_alert >= policy.refractory_hours
            ):
                alerts.append(int(hour))
                last_alert = int(hour)
                streak = 0
        else:
            streak = 0
    return alerts


def _evaluate_streams(
    streams: list[_Stream],
    policy: AlertPolicy,
    *,
    total_rows: int,
    early_window: int,
    on_time_window: int,
) -> PolicyReport:
    n_alerts = 0
    n_early = n_on_time = n_late = n_missed = n_eligible = 0
    leads: list[float] = []

    for stream in streams:
        alerts = alert_hours(stream, policy)
        n_alerts += len(alerts)
        if not stream.eligible or stream.t_sepsis is None:
            continue
        n_eligible += 1
        t = stream.t_sepsis
        early = [a for a in alerts if t - early_window <= a < t]
        on_time = [a for a in alerts if t <= a <= t + on_time_window]
        if early:
            n_early += 1
            leads.append(float(t - min(early)))
        elif on_time:
            n_on_time += 1
        elif any(a > t + on_time_window for a in alerts):
            n_late += 1
        else:
            n_missed += 1

    patient_days = total_rows / 24.0
    return PolicyReport(
        threshold=policy.threshold,
        sensitivity=(n_early / n_eligible) if n_eligible else 0.0,
        alerts_per_patient_day=(n_alerts / patient_days) if patient_days else 0.0,
        median_lead_hours=float(np.median(leads)) if leads else None,
        n_patients=len(streams),
        n_eligible=n_eligible,
        n_alerts=n_alerts,
        n_early_detected=n_early,
        n_on_time=n_on_time,
        n_late=n_late,
        n_missed=n_missed,
        patient_days=patient_days,
    )


def evaluate_policy(
    frame: pd.DataFrame,
    policy: AlertPolicy,
    *,
    early_window: int = EARLY_WINDOW_HOURS,
    on_time_window: int = ON_TIME_WINDOW_HOURS,
) -> PolicyReport:
    """Measure one policy on one split. Never call this on test to tune."""
    streams, total_rows = _build_streams(frame)
    return _evaluate_streams(
        streams,
        policy,
        total_rows=total_rows,
        early_window=early_window,
        on_time_window=on_time_window,
    )


def select_threshold(
    frame: pd.DataFrame,
    *,
    budget: float = ALERT_BUDGET_PER_PATIENT_DAY,
    refractory_hours: int = REFRACTORY_HOURS,
    persistence_hours: int = PERSISTENCE_HOURS,
    grid: np.ndarray | None = None,
    early_window: int = EARLY_WINDOW_HOURS,
    on_time_window: int = ON_TIME_WINDOW_HOURS,
) -> tuple[AlertPolicy, PolicyReport]:
    """Choose a policy on validation data under an alert budget.

    Ties are broken by (sensitivity, median lead, higher threshold). The last
    key prefers fewer alerts at equal performance, which is the clinically
    conservative choice.
    """
    if budget < 0:
        raise ThresholdError("budget must be non-negative")
    streams, total_rows = _build_streams(frame)
    if grid is None:
        # Empirical quantiles avoid a grid-boundary artefact: a calibrated model
        # can put almost all mass below a fixed 0.01 floor, and the selector
        # would then look like it "chose" 0.01 when it really wanted lower.
        probabilities = frame["p"].to_numpy(dtype=float)
        quantiles = np.quantile(probabilities, np.linspace(0.50, 0.9999, 250))
        start, stop, step = THRESHOLD_GRID
        fixed = np.arange(start, stop, step)
        grid = np.unique(np.clip(np.concatenate([quantiles, fixed]), 0.0, 1.0))

    reports: list[PolicyReport] = []
    for tau in grid:
        policy = AlertPolicy(float(tau), refractory_hours, persistence_hours)
        reports.append(
            _evaluate_streams(
                streams,
                policy,
                total_rows=total_rows,
                early_window=early_window,
                on_time_window=on_time_window,
            )
        )

    feasible = [r for r in reports if r.alerts_per_patient_day <= budget + 1e-9]
    if not feasible:
        best = min(reports, key=lambda r: r.alerts_per_patient_day)
        raise InfeasibleAlertBudget(
            f"no threshold meets the {budget:.2f} alerts/patient-day budget; "
            f"best observed is {best.alerts_per_patient_day:.2f} at "
            f"threshold {best.threshold:.2f}. Raise the budget or improve "
            f"specificity — do not lower the gate silently."
        )

    chosen = max(
        feasible,
        key=lambda r: (r.sensitivity, r.median_lead_hours or -1.0, r.threshold),
    )
    policy = AlertPolicy(chosen.threshold, refractory_hours, persistence_hours)
    return policy, chosen
