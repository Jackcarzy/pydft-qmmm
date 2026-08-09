"""Functionality for importing interfaces to external software.

Attributes:
    DISCOVERED_INTERFACES: A tuple of entry points into the interface
        architecture of PyDFT-QMMM from installed package metadata.
    BUNDLED_INTERFACES: The names of the interfaces shipped in this
        package, in the order they are registered.
    LOADED_INTERFACES: The loaded interface modules.
    UNAVAILABLE_INTERFACES: The bundled interfaces that could not be
        imported, mapped to the reason, so that an engine going missing
        is diagnosable rather than silent.
"""
from __future__ import annotations

__all__ = ["get_interfaces"]

import importlib
from importlib.metadata import entry_points
from types import ModuleType
from typing import TYPE_CHECKING

BUNDLED_INTERFACES = ("psi4", "openmm", "sparc", "vasp")

if TYPE_CHECKING:
    from typing import TypeAlias
    from .interface import QMFactory
    from .interface import MMFactory
    from pydft_qmmm.utils import TheoryLevel

    Factory: TypeAlias = QMFactory | MMFactory


try:
    # This is for Python 3.10-3.11.
    DISCOVERED_INTERFACES = tuple(
        entry_points(
        ).get("pydft_qmmm.interfaces", []),  # type: ignore[attr-defined]
    )
except AttributeError:
    # This is for Python +3.12, importlib.metadata now uses a selectable
    # EntryPoints object.
    DISCOVERED_INTERFACES = tuple(
        entry_points(group="pydft_qmmm.interfaces"),
    )

UNAVAILABLE_INTERFACES: dict[str, str] = {}


def _load_bundled(name: str) -> ModuleType | None:
    """Import a bundled interface, or skip it if its engine is absent.

    Each interface imports the package it wraps at module scope, so an
    engine that is not installed would otherwise make `import
    pydft_qmmm` fail outright.  One tree serves several engines, and no
    environment is expected to hold all of them at once.

    Args:
        name: The name of the interface subpackage.

    Returns:
        The interface module, or None when its engine is unavailable.
    """
    try:
        return importlib.import_module(f"pydft_qmmm.interfaces.{name}")
    except ImportError as exc:
        UNAVAILABLE_INTERFACES[name] = str(exc)
        return None


LOADED_INTERFACES = tuple(
    map(lambda x: x.load(), DISCOVERED_INTERFACES),
) + tuple(
    module for module in map(_load_bundled, BUNDLED_INTERFACES)
    if module is not None
)


def get_interfaces() -> dict[str, tuple[TheoryLevel, Factory]]:
    """Get PyDFT-QMMM interfaces to external packages.

    Returns:
        A dictionary of interface theory levels and factory functions
        indexed by interface name.
    """
    interfaces = dict(
        map(
            lambda y: (y.NAME, (y.THEORY_LEVEL, y.FACTORY)),
            LOADED_INTERFACES,
        ),
    )
    return interfaces
