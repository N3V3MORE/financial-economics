"""Compatibility entry point; implementation: financial_economics.studies.paper."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.studies.paper", "paper.py", globals())
