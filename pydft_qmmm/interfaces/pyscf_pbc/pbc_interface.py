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
from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import system_cache

from ..pyscf.pyscf_backend import load_backend
from ..pyscf.pyscf_backend import load_submodule
from ..pyscf.pyscf_backend import to_like
from ..pyscf.pyscf_backend import to_numpy
from ..vasp.grid_potential import spectral_value_and_gradient
from .pbc_cell import build_cell
from .pbc_cell import valence_charges
from .pbc_embedding import ao_operator
from .pbc_embedding import external_potential
from .pbc_embedding import grid_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm.potentials import ElectronicPotential


@dataclass(frozen=True)
class _SCFState:
    r"""A converged periodic solver and everything derived with it.

    Attributes:
        cell: The periodic cell for subsystem I.
        qm_indices: The original system indices of subsystem I, in cell
            order.
        embed_indices: The original system indices of subsystem II.
        kpts: The k-points, a single point in this release.
        method: The converged KRKS or KUKS solver.
        dm: The converged density matrix.
        coords: The uniform grid coordinates
            (:math:`\mathrm{\mathring{A}}`).
        weights: The uniform quadrature weights (:math:`\mathrm{a_0^3}`).
        potential: V_ext (:math:`\mathrm{E_h}`) at each grid point.
        matrix: The one-electron operator added to the core Hamiltonian.
        nuclear_energy: The energy (:math:`\mathrm{E_h}`) of the QM
            nuclei in V_ext.
        potentials: The registered electronic potentials, kept so the
            reaction forces use the same helPME instance that built
            V_ext rather than a freshly constructed one.
    """
    cell: Any
    qm_indices: tuple[int, ...]
    embed_indices: tuple[int, ...]
    kpts: NDArray[np.float64]
    method: Any
    dm: Any
    coords: NDArray[np.float64]
    weights: NDArray[np.float64]
    potential: NDArray[np.float64]
    matrix: NDArray[np.complex128]
    nuclear_energy: float
    potentials: tuple[Any, ...]


def _nuclear_coupling(
        cell: Any,
        potential: NDArray[np.float64],
        box: NDArray[np.float64],
) -> float:
    r"""The energy of the QM nuclei in the external potential.

    Z_I is the pseudopotential valence charge, not the atomic number:
    under a pseudopotential the nucleus carries only valence charge, so
    the atomic number would over-count by the core electrons.

    Args:
        cell: A built periodic cell.
        potential: V_ext (:math:`\mathrm{E_h}`) at each grid point.
        box: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{\mathring{A}}`).

    Returns:
        The nuclear embedding energy (:math:`\mathrm{E_h}`).
    """
    mesh = tuple(int(n) for n in cell.mesh)
    positions = np.asarray(cell.atom_coords()) / BOHR_PER_ANGSTROM
    value, _ = spectral_value_and_gradient(
        potential.reshape(mesh),
        np.asarray(box, dtype=np.float64),
        positions,
    )
    return float(-np.sum(valence_charges(cell) * value))


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
        cell, _ = build_cell(
            self.system, self.basis, self.pseudo, self.ke_cutoff,
            self.mesh, self.charge, self.multiplicity, self.verbose,
        )
        return valence_charges(cell)

    @system_cache("positions", "charges", "elements", "subsystems", "box")
    def _scf_state(self) -> _SCFState:
        r"""Converge the embedded periodic SCF.

        ``energy_nuc`` is deliberately left alone: ``cell.energy_nuc()``
        is already the correct QM-QM Ewald sum, and the coupling to the
        MM environment is the separate additive ``nuclear_energy``.

        Returns:
            The converged state and everything derived alongside it.

        Raises:
            RuntimeError: If the SCF fails to converge.
        """
        backend = load_backend(self.device)
        cell, qm_indices = build_cell(
            self.system, self.basis, self.pseudo, self.ke_cutoff,
            self.mesh, self.charge, self.multiplicity, self.verbose,
        )
        kpts = cell.make_kpts([1, 1, 1])
        dft = load_submodule(backend, "pbc.dft")
        solver = dft.KUKS if self.multiplicity > 1 else dft.KRKS
        method = solver(cell, kpts=kpts, xc=self.functional)
        method.conv_tol = self.conv_tol
        method.max_cycle = self.max_cycle
        # Honour output_file/output_interval the way the molecular
        # interface does: the frame counter decides whether this call is
        # one of the ones written.
        if self.output_file is not None:
            self.frame[0] += 1
            if self.frame[0] % self.output_interval == 0:
                method.stdout = open(self.output_file, "a")
                method.verbose = max(self.verbose, 4)
        for key, value in self.options.items():
            setattr(method, key, value)
        embed_indices = tuple(sorted(self.system.select("subsystem II")))
        coords, weights = grid_coordinates(cell)
        nuclear_energy = 0.0
        potential = np.zeros(len(coords))
        nao = cell.nao_nr()
        matrix = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
        if self.embedding:
            potential = external_potential(
                self.system, cell, self.potentials, embed_indices,
                self.embedding_sigma,
            )
            matrix = ao_operator(
                backend, cell, kpts, coords, weights, potential,
            )
            core = method.get_hcore()
            shifted = to_like(to_numpy(core) + matrix, core)
            method.get_hcore = lambda *args, **kwargs: shifted
            nuclear_energy = _nuclear_coupling(
                cell, potential, self.system.box,
            )
        method.kernel(dm0=self.density_guess[0])
        if not method.converged:
            raise RuntimeError(
                "the periodic SCF did not converge; raise max_cycle or"
                " loosen conv_tol",
            )
        dm = method.make_rdm1()
        self.density_guess[0] = dm
        return _SCFState(
            cell, qm_indices, embed_indices, kpts, method, dm,
            coords, weights, potential, matrix, nuclear_energy,
            tuple(self.potentials),
        )


class PySCFPBCPotential(PySCFPBCInterface, AtomicPotential):
    """A potential wrapping periodic PySCF."""

    def compute_energy(self) -> float:
        r"""Compute the energy of the system.

        Returns:
            The energy (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        state = self._scf_state()
        return (
            float(state.method.e_tot) + state.nuclear_energy
        ) * KJMOL_PER_EH

    def compute_forces(self) -> NDArray[np.float64]:
        r"""Compute the forces on the system.

        Returns:
            The forces (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`)
            acting on atoms in the system.
        """
        from .pbc_forces import mm_forces
        from .pbc_forces import qm_forces
        state = self._scf_state()
        backend = load_backend(self.device)
        natoms = len(self.system.positions)
        forces = qm_forces(backend, state, self.system.box, natoms)
        if self.embedding:
            forces += mm_forces(
                backend, state, self.system, self.embedding_sigma, natoms,
            )
        return forces * KJMOL_PER_EH * BOHR_PER_ANGSTROM

    def compute_components(self) -> dict[str, float]:
        r"""Compute the components of the energy.

        Returns:
            The components of the energy
            (:math:`\mathrm{kJ\;mol^{-1}}`) of the system.
        """
        state = self._scf_state()
        return {
            "Periodic QM Energy": float(state.method.e_tot) * KJMOL_PER_EH,
            "Nuclear Embedding Energy": (
                state.nuclear_energy * KJMOL_PER_EH
            ),
        }
