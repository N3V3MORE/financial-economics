"""Compatibility entry point; implementation: financial_economics.models.paper_gru."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.models.paper_gru", "paper_gru.py", globals())
