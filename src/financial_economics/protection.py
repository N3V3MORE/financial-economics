"""Compatibility entry point; implementation: financial_economics.studies.protection."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.studies.protection", "protection.py", globals())
