"""Keep original imports, monkeypatches, runpy execution, and pickle namespaces."""

from importlib import import_module
from importlib.machinery import SourceFileLoader
import sys


class _AliasSourceLoader(SourceFileLoader):
    """Read the same implementation under either its original or grouped name."""

    def get_code(self, fullname):
        return SourceFileLoader(fullname, self.path).get_code(fullname)

    def get_source(self, fullname):
        return SourceFileLoader(fullname, self.path).get_source(fullname)

    def get_filename(self, fullname):
        return self.path


def install_alias(original, canonical):
    implementation = import_module(canonical)
    loader = _AliasSourceLoader(canonical, implementation.__file__)
    implementation.__loader__ = loader
    implementation.__spec__.loader = loader
    # Original imports continue to produce pickle-compatible names for symbols
    # defined in this module. Imported symbols retain their defining namespace.
    for value in vars(implementation).values():
        if getattr(value, "__module__", None) == canonical:
            value.__module__ = original
    sys.modules[original] = implementation
    return implementation
