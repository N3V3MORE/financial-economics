"""Compatibility entry point; implementation: financial_economics.models.networks."""

from ._compat import expose_module as _expose_module

_expose_module(__name__, "financial_economics.models.networks", "paper_networks.py", globals())
