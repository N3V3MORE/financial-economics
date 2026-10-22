"""Compatibility entry point; implementation: financial_economics.portfolio.execution."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.portfolio.execution", "costs.py", globals())
