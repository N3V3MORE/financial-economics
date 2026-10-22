"""Keep original imports, monkeypatches, runpy execution, and pickle namespaces."""

from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
import sys


_ORIGINAL_NAMES = {}


def _restore_symbol_names(implementation, original, canonical):
    for value in vars(implementation).values():
        if getattr(value, "__module__", None) == canonical:
            value.__module__ = original


class _AliasSourceLoader(SourceFileLoader):
    """Read the same implementation under either its original or grouped name."""

    def get_code(self, fullname):
        return SourceFileLoader(fullname, self.path).get_code(fullname)

    def get_source(self, fullname):
        return SourceFileLoader(fullname, self.path).get_source(fullname)

    def get_filename(self, fullname):
        return self.path


    def exec_module(self, module):
        super().exec_module(module)
        original = _ORIGINAL_NAMES.get(module.__name__)
        if original:
            _restore_symbol_names(module, original, module.__name__)


class _ReloadFinder(MetaPathFinder):
    """Restore alias-aware loading only when an aliased module is reloaded."""

    def find_spec(self, fullname, path=None, target=None):
        if target is None or fullname not in _ORIGINAL_NAMES:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _AliasSourceLoader(fullname, spec.origin)
        return spec


sys.meta_path.insert(0, _ReloadFinder())


def install_alias(original, canonical):
    implementation = import_module(canonical)
    if original != "__main__":
        _ORIGINAL_NAMES[canonical] = original
    loader = _AliasSourceLoader(canonical, implementation.__file__)
    implementation.__loader__ = loader
    implementation.__spec__.loader = loader
    # Original imports continue to produce pickle-compatible names for symbols
    # defined in this module. Imported symbols retain their defining namespace.
    _restore_symbol_names(implementation, original, canonical)
    sys.modules[original] = implementation
    return implementation
