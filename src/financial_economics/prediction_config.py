"""Compatibility entry point; implementation: financial_economics.settings.prediction."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.settings.prediction", "prediction_config.py", globals())
