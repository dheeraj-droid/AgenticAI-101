"""Third-party telemetry opt-outs.

CrewAI exports OpenTelemetry spans to its own endpoint unless told not to.
Nothing in this project should be talking to a third party about customer
records, so every place that imports ``crewai`` calls ``opt_out()`` first.

``setdefault`` rather than assignment: an explicit value already in the
environment wins, so telemetry can still be turned back on deliberately.
"""

from __future__ import annotations

import os


def opt_out() -> None:
    """Silence vendor telemetry. Must run before ``crewai`` is imported."""
    os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
