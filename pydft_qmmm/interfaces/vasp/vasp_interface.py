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
import shutil
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
    embedding: bool = False
    embedding_sigma: float = 0.3
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )

    def configure_electrostatic_embedding(self, enabled: bool) -> None:
        """Align the VASP plugin with the QM/MM coupling Hamiltonian.

        Electrostatic coupling requires VASP's external-potential plugin.
        Mechanical or absent coupling must not run that plugin, because
        OpenMM retains those electrostatic interactions and they would be
        counted twice.

        Args:
            enabled: Whether any QM/MM electrostatics are assigned to
                the QM level of theory.

        Raises:
            ValueError: If embedding was manually enabled for a coupling
                scheme which leaves electrostatics at the MM level.
        """
        if enabled:
            # SoftwareInterface is frozen to keep its external-engine
            # handles stable.  This flag is configuration state finalized
            # while the composite calculator is being built.
            object.__setattr__(self, "embedding", True)
        elif self.embedding:
            raise ValueError(
                "VASP embedding=True conflicts with this QMMMHamiltonian: "
                "no QM/MM electrostatic interaction is assigned to the QM "
                "level. Disable embedding or select electrostatic coupling "
                "to avoid double-counting electrostatics.",
            )

    def add_electronic_potential(
            self, potential: ElectronicPotential,
    ) -> None:
        """Electrostatic embedding is not yet supported for VASP.

        Args:
            potential: The electronic potential that would be
                incorporated into QM calculations.
        """
        if not self.embedding:
            raise NotImplementedError(
                "PME embedding needs the VASP Python plugin.  Build the "
                "potential with embedding=True and a vasp_std compiled "
                "with -DPLUGINS.",
            )
        self.potentials.append(potential)

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
        if self.embedding:
            # PLUGINS/MODE is deliberately left unset.  It defaults to
            # "serial", which is what makes this correct under MPI: only
            # rank 1 calls the plugin, it receives the FULL gathered
            # grid, and the addition is broadcast back.  Under
            # PLUGINS/MODE = parallel every rank would be handed its own
            # grid slab and a full-grid V_ext would be silently wrong.
            tags["PLUGINS/LOCAL_POTENTIAL"] = "T"
            tags["PLUGINS/FORCE_AND_STRESS"] = "T"
            # Both files: VASP imports vasp_plugin as a TOP-LEVEL module
            # from the run directory, so its package-relative import of
            # grid_potential falls back to a plain one, which only
            # resolves if grid_potential.py sits beside it.
            modules = ["vasp_plugin.py", "grid_potential.py"]
            if self.potentials:
                modules.append("pme_external.py")
            for name in modules:
                shutil.copyfile(
                    os.path.join(os.path.dirname(__file__), name),
                    os.path.join(self.directory, name),
                )
            # The plugin writes MM_FORCES during the run, not this
            # method.  Remove any stale copy before launching so that a
            # run which dies before the plugin fires leaves a missing
            # file rather than the previous step's forces to be
            # silently reread.
            force_path = os.path.join(self.directory, "MM_FORCES")
            if os.path.isfile(force_path):
                os.remove(force_path)
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

    def _write_mm_charges(self) -> int:
        r"""Write subsystem II point charges for the plugin.

        Returns:
            The number of MM charges written.
        """
        os.makedirs(self.directory, exist_ok=True)
        indices = sorted(self.system.select("subsystem II"))
        positions = np.asarray(self.system.positions)[indices]
        charges = np.asarray(self.system.charges)[indices]
        path = os.path.join(self.directory, "MM_CHARGES")
        # Delete before rewriting so that a failed write surfaces as a
        # missing file rather than leaving the previous step's charges
        # in place to be silently reused.
        if os.path.isfile(path):
            os.remove(path)
        vasp_utils.write_mm_charges(
            path, positions, charges, self.frame[0], self.embedding_sigma,
        )
        return len(charges)

    def _write_pme_data(self) -> int:
        r"""Write the whole system plus Ewald parameters for the plugin.

        The cutoff scheme ships only subsystem II, but PME is a lattice
        sum: the reciprocal part runs over every charge and a real-space
        adjustment removes those that must not act on the QM region.

        Returns:
            The number of charges written.
        """
        os.makedirs(self.directory, exist_ok=True)
        potential = self.potentials[0]
        excluded = sorted(self.system.select("not subsystem III"))
        path = os.path.join(self.directory, "PME_DATA")
        if os.path.isfile(path):
            os.remove(path)
        vasp_utils.write_pme_data(
            path,
            np.asarray(self.system.positions),
            np.asarray(self.system.charges),
            excluded,
            potential.pme_alpha,
            tuple(potential.pme_gridnumber),
            potential.pme_spline_order,
            self.frame[0],
        )
        return len(self.system.charges)

    def _check_plugin_fired(self) -> None:
        """Verify that the plugin actually ran.

        Raises:
            VaspExecutionError: If the sentinel is absent, which means
                the external potential was never applied and the energy
                is quietly the unembedded one.

        Note that grepping the binary for "not compiled with PLUGINS"
        does NOT work as a check -- that string is present in every
        build, plugin-enabled or not.
        """
        from .vasp_plugin import SENTINEL
        if not os.path.isfile(os.path.join(self.directory, SENTINEL)):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "Electrostatic embedding was requested but the plugin "
                "sentinel is absent, so the external potential was "
                "never applied.  Check that vasp_std is built with "
                "-DPLUGINS and that PYTHONHOME and PATH point at the "
                "environment holding the plugin's Python.",
            )

    @system_cache("positions", "elements", "subsystems", "box", "charges")
    def _run(self) -> tuple[float, NDArray[np.float64]]:
        r"""Run VASP on the QM subsystem.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) and an Nx3 force
            array (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) for
            the QM atoms only, ordered to match
            ``sorted(system.select("subsystem I"))``.
        """
        order = self._write_input()
        if self.embedding:
            # Always write the near-field charges.  Under PME they are
            # excluded from the reciprocal sum and must be reinstated
            # analytically; without them the QM region feels only the
            # long-range tail.
            self._write_mm_charges()
            if self.potentials:
                self._write_pme_data()
        vasp_utils.run_vasp(self.command, self.directory)
        if self.embedding:
            self._check_plugin_fired()
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
