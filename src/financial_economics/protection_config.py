"""Compatibility entry point; implementation: financial_economics.settings.protection."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.settings.protection", "protection_config.py", globals())
