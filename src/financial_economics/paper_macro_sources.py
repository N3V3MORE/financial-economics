"""Compatibility entry point; implementation: financial_economics.datasets.sources.macro."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.datasets.sources.macro", "paper_macro_sources.py", globals())
