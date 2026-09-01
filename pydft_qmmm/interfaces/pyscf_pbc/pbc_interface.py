"""The periodic PySCF interface and potential.

The QM cell is the OpenMM simulation box itself, so the wavefunction is
periodic rather than molecular.  That rules out ``pyscf.qmmm``, whose
assertions pass for a pbc SCF object but whose integrals carry no
lattice sum, so the MM environment reaches the electrons through a
real-space potential sampled on the cell's own uniform FFT grid.
"""
from __future__ import annotations

__all__ = ["PySCFPBCInterface", "PySCFPBCPotential"]

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.potentials import AtomicPotential

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential


@dataclass(frozen=True)
class PySCFPBCInterface(QMInterface):
    r"""Store and manipulate periodic PySCF data types.

    Args:
        system: The system that will inform the interface.
        basis: The GTH basis set name.
        pseudo: The GTH pseudopotential name.
        functional: The exchange-correlation functional, or None for
            Hartree-Fock.
        ke_cutoff: The kinetic energy cutoff (:math:`\mathrm{E_h}`), or
            None when an explicit mesh is given.
        mesh: An explicit FFT mesh, or None when a cutoff is given.
        embedding_sigma: The Gaussian width
            (:math:`\mathrm{\mathring{A}}`) smearing subsystem II point
            charges onto the grid.
        device: Either ``cpu`` for PySCF or ``gpu`` for GPU4PySCF.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        output_file: The file PySCF output is written to, or None.
        output_interval: The interval at which output is written.
        conv_tol: The SCF convergence threshold (:math:`\mathrm{E_h}`).
        max_cycle: The maximum number of SCF iterations.
        verbose: The PySCF logging verbosity.
        options: Additional attributes to set on the solver.

    Attributes:
        potentials: The electronic potentials folded into the core
            Hamiltonian.
        density_guess: The last compatible density matrix for SCF reuse.
        frame: The estimated current frame, for output writing.
        embedding: Whether the QM/MM Hamiltonian assigns electrostatics
            to the QM level.
    """
    basis: str
    pseudo: str
    functional: str | None
    ke_cutoff: float | None
    mesh: tuple[int, int, int] | None
    embedding_sigma: float
    device: str
    charge: int
    multiplicity: int
    output_file: str | None
    output_interval: int
    conv_tol: float
    max_cycle: int
    verbose: int
    options: dict[str, Any] = field(default_factory=dict)
    potentials: list[ElectronicPotential] = field(
        default_factory=list,
        init=False,
    )
    density_guess: list[Any] = field(
        default_factory=lambda: [None],
        init=False,
    )
    frame: list[int] = field(
        default_factory=lambda: [0],
        init=False,
    )
    embedding: bool = field(default=False, init=False)

    def electrostatic_coupling_mode(self) -> ElectrostaticCouplingMode:
        """Keep electrostatic coupling inside this interface.

        Like VASP, this interface owns the whole electrostatic
        coupling, including the term acting on its own nuclei.
        """
        return ElectrostaticCouplingMode.ENGINE

    def applies_nuclear_potential(self) -> bool:
        """This interface couples V_ext to its own nuclei.

        The SCF state adds ``-sum(Z_I V_ext(R_I))`` with the
        pseudopotential valence charge, so the coupling Hamiltonian must
        not add its own nuclear term on top.

        Returns:
            Whether the nuclear term is already applied.
        """
        return self.embedding

    def configure_electrostatic_embedding(self, enabled: bool) -> None:
        """Align the interface with the QM/MM coupling Hamiltonian.

        Args:
            enabled: Whether the QM/MM Hamiltonian assigns any
                electrostatic interaction to the QM level of theory.

        Raises:
            ValueError: If embedding was enabled for a coupling scheme
                which leaves electrostatics at the MM level.
        """
        if enabled:
            # SoftwareInterface is frozen to keep external-engine
            # handles stable.  This flag is configuration state
            # finalized while the composite calculator is being built.
            object.__setattr__(self, "embedding", True)
        elif self.embedding:
            raise ValueError(
                "pyscf_pbc embedding conflicts with this QMMMHamiltonian:"
                " no QM/MM electrostatic interaction is assigned to the QM"
                " level. Disable embedding or select electrostatic coupling"
                " to avoid double-counting electrostatics.",
            )

    def add_electronic_potential(
            self, potential: ElectronicPotential,
    ) -> None:
        """Register a potential to fold into the core Hamiltonian.

        Args:
            potential: The electronic potential to incorporate into QM
                calculations.
        """
        self.potentials.append(potential)

    def nuclear_charges(self) -> NDArray[np.float64]:
        r"""Get the effective nuclear charges the QM method uses.

        Under a GTH pseudopotential the nucleus carries only the
        valence charge, so the atomic number would over-count every
        nuclear term by the core electrons.

        Returns:
            The valence charges (:math:`e`) of the subsystem I atoms,
            ordered by ascending system index.
        """
        from .pbc_cell import build_cell
        from .pbc_cell import valence_charges
        cell, _ = build_cell(
            self.system, self.basis, self.pseudo, self.ke_cutoff,
            self.mesh, self.charge, self.multiplicity, self.verbose,
        )
        return valence_charges(cell)


class PySCFPBCPotential(PySCFPBCInterface, AtomicPotential):
    """A potential wrapping periodic PySCF."""

    def compute_energy(self) -> float:
        r"""Compute the energy of the system.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        raise NotImplementedError("implemented in Task 6")

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on atoms in the system.
        """
        raise NotImplementedError("implemented in Tasks 7 and 8")

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of the energy.

        Returns:
            The components of the energy
            (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        raise NotImplementedError("implemented in Task 6")
