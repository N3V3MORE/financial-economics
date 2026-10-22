"""Compatibility entry point; implementation: financial_economics.evaluation.protection."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.evaluation.protection", "protection_inference.py", globals())
