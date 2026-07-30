"""The VASP software interface and potential.

This module wraps VASP as a QM potential within PyDFT-QMMM.  Each energy
or force evaluation writes a fresh set of VASP input files into a
persistent working directory, launches VASP as a subprocess for a single
point calculation, and parses the resulting ``vasprun.xml``.  The
``WAVECAR`` and ``CHGCAR`` left behind by one step are reused to
initialize the next, which makes the sequence of single points behave
much like a continued SCF.

Only mechanical embedding is supported at present: VASP sees the QM
atoms alone, and QM/MM coupling is carried by the MM force field.
Electrostatic embedding requires a VASP binary compiled with
``-DPLUGINS`` and is implemented separately.
"""
from __future__ import annotations

__all__ = ["VaspInterface", "VaspPotential"]

import os
from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential
from pydft_qmmm.utils import KJMOL_PER_EV
from pydft_qmmm.utils import system_cache

from . import vasp_utils

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential
    from pydft_qmmm import System  # noqa: F401


@dataclass(frozen=True)
class VaspInterface(QMInterface):
    r"""A mix-in for storing and manipulating VASP data types.

    Args:
        system: The system that will inform the interface to VASP.
        charge: The net charge (:math:`e`) of the QM subsystem.  VASP
            has no molecular charge concept, so a non-zero charge must
            be realized by setting NELECT together with a compensating
            background; only ``charge=0`` is currently accepted.
        directory: The working directory for VASP calculations.
        command: The shell command that launches VASP.
        incar: INCAR tags applied to every calculation.
        kpts: The number of k-points along each reciprocal lattice
            vector.
        pp_path: The directory containing per-species POTCAR
            subdirectories.
        potcar_map: A mapping from element symbol to POTCAR
            subdirectory name, for selecting non-default potentials.

    Attributes:
        potentials: A list of electronic potentials to incorporate into
            QM calculations.
        frame: The number of calculations performed so far, used to
            decide whether a restart file is available.
    """
    charge: int
    directory: str
    command: str
    incar: dict[str, Any]
    kpts: tuple[int, int, int]
    pp_path: str
    potcar_map: dict[str, str]
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )

    def add_electronic_potential(
            self, potential: ElectronicPotential,
    ) -> None:
        """Electrostatic embedding is not yet supported for VASP.

        Args:
            potential: The electronic potential that would be
                incorporated into QM calculations.
        """
        raise NotImplementedError(
            "Electrostatic embedding with VASP requires a VASP binary "
            "compiled with -DPLUGINS, which is not yet wired up in this "
            "interface.  Use a mechanical-embedding QM/MM scheme, i.e. "
            "QMMMHamiltonian('mechanical', 'mechanical').",
        )

    def _write_input(self) -> list[int]:
        """Write the VASP input files for the current QM geometry.

        Returns:
            The permutation mapping POSCAR order onto the sorted QM
            atom indices.
        """
        os.makedirs(self.directory, exist_ok=True)
        qm_indices = sorted(self.system.select("subsystem I"))
        symbols = [str(self.system.elements[i]) for i in qm_indices]
        positions = np.asarray(self.system.positions)[qm_indices]
        # PyDFT-QMMM stores lattice vectors as columns; VASP wants rows.
        cell = np.asarray(self.system.box).T
        species, _, order = vasp_utils.write_poscar(
            os.path.join(self.directory, "POSCAR"),
            symbols,
            positions,
            cell,
        )
        vasp_utils.write_potcar(
            os.path.join(self.directory, "POTCAR"),
            species,
            self.pp_path,
            self.potcar_map,
        )
        vasp_utils.write_kpoints(
            os.path.join(self.directory, "KPOINTS"),
            self.kpts,
        )
        tags = dict(self.incar)
        # Single point only; PyDFT-QMMM owns the dynamics.
        tags.update({"NSW": 0, "IBRION": -1})
        # Reuse the previous step's orbitals and density once they exist.
        if self.frame[0] and os.path.isfile(
                os.path.join(self.directory, "WAVECAR"),
        ):
            tags.setdefault("ISTART", 1)
        vasp_utils.write_incar(
            os.path.join(self.directory, "INCAR"),
            tags,
        )
        return order

    @system_cache("positions", "elements", "subsystems", "box")
    def _run(self) -> tuple[float, NDArray[np.float64]]:
        r"""Run VASP on the QM subsystem.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) and an Nx3 force
            array (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) for
            the QM atoms only, ordered to match
            ``sorted(system.select("subsystem I"))``.
        """
        order = self._write_input()
        vasp_utils.run_vasp(self.command, self.directory)
        energy, forces = vasp_utils.read_vasprun(
            os.path.join(self.directory, "vasprun.xml"),
        )
        if len(forces) != len(order):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                f"VASP returned forces for {len(forces)} atoms, but "
                f"{len(order)} QM atoms were submitted.",
            )
        # Undo the species grouping applied when writing the POSCAR.
        qm_forces = np.empty_like(forces)
        qm_forces[order, :] = forces
        self.frame[0] += 1
        return energy * KJMOL_PER_EV, qm_forces * KJMOL_PER_EV


class VaspPotential(VaspInterface, AtomicPotential):
    """A potential wrapping VASP functionality.

    Args:
        system: The system that will inform the interface to VASP.
        charge: The net charge (:math:`e`) of the QM subsystem.
        directory: The working directory for VASP calculations.
        command: The shell command that launches VASP.
        incar: INCAR tags applied to every calculation.
        kpts: The number of k-points along each reciprocal lattice
            vector.
        pp_path: The directory containing per-species POTCAR
            subdirectories.
        potcar_map: A mapping from element symbol to POTCAR
            subdirectory name.
    """

    def compute_energy(self) -> float:
        r"""Compute the energy of the system using VASP.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the QM
            subsystem.
        """
        energy, _ = self._run()
        return energy

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system using VASP.

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
        r"""Compute the components of energy using VASP.

        Returns:
            An empty dict; the VASP interface does not currently expose
            energy sub-components.
        """
        components: dict[str, float] = {}
        return components
