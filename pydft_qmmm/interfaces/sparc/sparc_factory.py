"""Functionality for building the SPARC interface.
"""
from __future__ import annotations

__all__ = ["sparc_interface_factory"]

from typing import Any
from typing import TYPE_CHECKING

from sparc.calculator import SPARC

from .sparc_interface import SPARCPotential

if TYPE_CHECKING:
    from pydft_qmmm import System


def sparc_interface_factory(
        system: System,
        /,
        charge: int = 0,
        **options: Any,
) -> SPARCPotential:
    r"""Build the interface to SPARC.

    Args:
        system: The system which will be tied to the SPARC interface.
        charge: The net charge (:math:`e`) of the QM subsystem.
        options: Additional keyword arguments forwarded to
            ``sparc.calculator.SPARC`` (e.g. ``h``, ``kpts``, ``xc``,
            ``directory``, ``command``).

    Returns:
        The SPARC interface.
    """
    calculator = SPARC(**options)
    return SPARCPotential(system, calculator, charge)
