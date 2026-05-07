"""The SPARC software interface and potential.

This module wraps the SPARC ASE calculator from SPARC-X-API and exposes
it as a QM potential within PyDFT-QMMM.  Only mechanical embedding is
supported; QM-region atoms are passed to SPARC and energies/forces are
returned with no electrostatic coupling to the MM subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import system_cache

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sparc.calculator import SPARC
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401

# 1 eV expressed in kJ/mol (CODATA-derived).
KJMOL_PER_EV = 96.48533212331


@dataclass(frozen=True)
class SPARCInterface(QMInterface):
    r"""A mix-in for storing and manipulating SPARC data types.

    Args:
        system: The system that will inform the interface to SPARC.
        calculator: A configured ``sparc.calculator.SPARC`` instance.
        charge: The net charge (:math:`e`) of the QM subsystem.  SPARC
            via SPARC-X-API does not expose a total-charge keyword, so
            only ``charge=0`` is currently accepted.
    """
    calculator: SPARC
    charge: int

    def __post_init__(self) -> None:
        if self.charge:
            raise NotImplementedError(
                "SPARC-X-API does not expose a system-charge parameter; "
                "only charge=0 is supported by the SPARC interface.",
            )

    def add_electronic_potential(
            self, potential: ElectronicPotential,
    ) -> None:
        """Electrostatic embedding is not supported for SPARC."""
        raise NotImplementedError(
            "The SPARC interface does not support electrostatic "
            "embedding; use a mechanical-embedding QM/MM scheme.",
        )

    @system_cache("positions", "elements", "subsystems", "box")
    def _run(self) -> tuple[float, NDArray[np.float64]]:
        r"""Run SPARC on the QM subsystem.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) and an Nx3 force
            array (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) for
            the QM atoms only, in the order returned by
            ``system.select("subsystem I")`` after sorting.
        """
        qm_indices = sorted(self.system.select("subsystem I"))
        symbols = [str(self.system.elements[i]) for i in qm_indices]
        positions = np.asarray(self.system.positions)[qm_indices]
        # pydft-qmmm stores box vectors as columns; ASE wants rows.
        cell = np.asarray(self.system.box).T
        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=cell,
            pbc=True,
        )
        atoms.calc = self.calculator
        energy = atoms.get_potential_energy() * KJMOL_PER_EV
        forces = atoms.get_forces() * KJMOL_PER_EV
        return energy, forces


class SPARCPotential(SPARCInterface, AtomicPotential):
    """A potential wrapping SPARC functionality.

    Args:
        system: The system that will inform the interface to SPARC.
        calculator: A configured ``sparc.calculator.SPARC`` instance.
        charge: The net charge (:math:`e`) of the QM subsystem.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using SPARC.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the QM
            subsystem.
        """
        energy, _ = self._run()
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using SPARC.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            on every atom in the system, with non-QM atoms zeroed.
        """
        _, qm_forces = self._run()
        forces = np.zeros(self.system.positions.shape)
        qm_indices = sorted(self.system.select("subsystem I"))
        forces[qm_indices, :] = qm_forces
        return forces

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of energy using SPARC.

        Returns:
            An empty dict; SPARC does not expose meaningful sub-components
            via SPARC-X-API.
        """
        return {}
