"""Compatibility entry point; implementation: financial_economics.reporting.comparison."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.reporting.comparison", "comparison_report.py", globals())
