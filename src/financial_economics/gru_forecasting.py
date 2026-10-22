"""Compatibility entry point; implementation: financial_economics.models.gpu."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.models.gpu", "gru_forecasting.py", globals())
