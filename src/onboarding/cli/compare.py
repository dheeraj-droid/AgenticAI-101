"""Run every framework over the same fixtures and collect the results."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from onboarding.adapters.base import FRAMEWORKS, get_adapter, load_record
from onboarding.core.audit import new_run_id
from onboarding.core.config import paths
from onboarding.core.schemas import CustomerRecord, DeterministicOutcome, OnboardingResult


@dataclass
class RunOutcome:
    """One (fixture, framework) cell of the comparison matrix."""

    fixture: str
    framework: str
    result: OnboardingResult | None = None
    error: str | None = None
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.result is not None

    def deterministic(self) -> DeterministicOutcome | None:
        return self.result.deterministic() if self.result else None


@dataclass
class ComparisonReport:
    """The full matrix plus the divergence analysis."""

    outcomes: list[RunOutcome] = field(default_factory=list)
    frameworks: tuple[str, ...] = FRAMEWORKS
    fixtures: list[str] = field(default_factory=list)
    llm_configured: bool = False
    pii_engine: str = "unknown"

    def for_fixture(self, fixture: str) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.fixture == fixture]

    def divergences(self, fixture: str) -> dict[str, dict[str, Any]]:
        """Fields where the frameworks' deterministic outcomes disagree."""
        cells = {o.framework: o.deterministic() for o in self.for_fixture(fixture) if o.ok}
        if len(cells) < 2:
            return {}
        fields = next(iter(cells.values())).model_dump().keys()
        out: dict[str, dict[str, Any]] = {}
        for name in fields:
            values = {fw: getattr(d, name) for fw, d in cells.items()}
            if len({_key(v) for v in values.values()}) > 1:
                out[name] = values
        return out

    def comparable(self, fixture: str) -> bool:
        """True when at least two frameworks actually produced a result.

        Without this, a fixture where every framework failed would report
        "no divergences" and read as agreement.
        """
        return sum(1 for o in self.for_fixture(fixture) if o.ok) >= 2

    @property
    def identical(self) -> bool:
        comparable = [f for f in self.fixtures if self.comparable(f)]
        return bool(comparable) and all(not self.divergences(f) for f in comparable)

    @property
    def skipped(self) -> list[str]:
        return [f for f in self.fixtures if not self.comparable(f)]


def _key(value: Any) -> str:
    return repr(value)


async def run_comparison(
    fixture_dir: Path | None = None,
    frameworks: tuple[str, ...] = FRAMEWORKS,
    only: list[str] | None = None,
) -> ComparisonReport:
    """Run each framework over each fixture. One failure never aborts the matrix."""
    from onboarding.core.llm import llm_spec
    from onboarding.core.pii import get_pii_engine

    directory = fixture_dir or paths().fixtures
    files = sorted(p for p in directory.glob("*.json") if not only or p.stem in only)

    report = ComparisonReport(
        fixtures=[p.stem for p in files],
        frameworks=frameworks,
        llm_configured=llm_spec().configured,
        pii_engine=get_pii_engine().name,
    )

    for path in files:
        record: CustomerRecord = load_record(path)
        for framework in frameworks:
            started = time.perf_counter()
            outcome = RunOutcome(fixture=path.stem, framework=framework)
            try:
                adapter = get_adapter(framework)
                outcome.result = await adapter.run(
                    record,
                    run_id=new_run_id(record.record_id, framework),
                    record_path=str(path),
                )
            except Exception as exc:
                outcome.error = f"{type(exc).__name__}: {exc}"
                outcome.result = None
                if _verbose():
                    traceback.print_exc()
            outcome.duration_ms = int((time.perf_counter() - started) * 1000)
            report.outcomes.append(outcome)
    return report


def _verbose() -> bool:
    import os

    return os.environ.get("ONBOARDING_VERBOSE", "").strip() not in ("", "0", "false")
