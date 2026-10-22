"""Compatibility entry point; implementation: financial_economics.evaluation.prediction."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.evaluation.prediction", "prediction_inference.py", globals())
