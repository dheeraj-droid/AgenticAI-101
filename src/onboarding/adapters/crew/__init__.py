"""The CrewAI adapter.

The telemetry opt-out runs here, before anything in the package imports
``crewai``, so it cannot be forgotten at a call site.
"""

from __future__ import annotations

from onboarding.core.telemetry import opt_out

opt_out()
