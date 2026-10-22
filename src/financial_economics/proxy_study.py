"""Compatibility entry point; implementation: financial_economics.studies.proxy."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.studies.proxy", "proxy_study.py", globals())
