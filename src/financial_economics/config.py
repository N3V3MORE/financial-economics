"""Compatibility entry point; implementation: financial_economics.settings.study."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.settings.study", "config.py", globals())
