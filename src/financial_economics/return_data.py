"""Compatibility entry point; implementation: financial_economics.datasets.returns."""

from ._compat import install_alias as _install_alias

_implementation = _install_alias(__name__, "financial_economics.datasets.returns")
