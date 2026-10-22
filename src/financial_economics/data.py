"""Compatibility entry point; implementation: financial_economics.datasets.public."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.datasets.public", "data.py", globals())
