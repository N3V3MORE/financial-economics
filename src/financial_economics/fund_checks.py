"""Compatibility entry point; implementation: financial_economics.portfolio.fund_checks."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.portfolio.fund_checks", "fund_checks.py", globals())
