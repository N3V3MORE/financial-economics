"""Compatibility entry point; implementation: financial_economics.reporting.baseline."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.reporting.baseline", "report.py", globals())
