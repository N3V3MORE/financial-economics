"""Compatibility entry point; implementation: financial_economics.studies.historical."""

from ._compat import install_alias as _install_alias

_implementation = _install_alias(__name__, "financial_economics.studies.historical")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
