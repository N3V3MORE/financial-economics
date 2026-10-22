"""Compatibility entry point; implementation: financial_economics.studies.realistic."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.studies.realistic", "paper_realistic.py", globals())
