"""Analytic forces for the periodic PySCF interface.

Every channel here differentiates the same discrete V_ext that entered
the SCF, sampled on the same uniform grid.  There are four:

1. the SCF gradient at fixed ``V_ao``, from ``nuc_grad_method``;
2. the Pulay term, the AO derivative of the added ``V``, which PySCF
   does not supply and which must be added by hand;
3. the nuclear term ``+Z_I grad V_ext(R_I)``; and
4. the reaction of the QM density on the static MM charges.

Channel 2 is the subtle one.  A patched ``get_hcore`` gradient runs and
returns physically plausible numbers while silently omitting it, so the
finite-difference test is the only thing that catches its absence.
"""
from __future__ import annotations

__all__ = [
    "pulay_forces",
    "nuclear_forces",
    "qm_forces",
    "qm_density_on_grid",
    "mm_forces",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import KJMOL_PER_EV
from ..pyscf.pyscf_backend import load_submodule
from ..pyscf.pyscf_backend import to_numpy
from ..vasp.grid_potential import contract_gaussian_gradient
from ..vasp.grid_potential import poisson_fft
from ..vasp.grid_potential import spectral_value_and_gradient
from ..vasp.grid_potential import spread_gaussian
from .pbc_cell import valence_charges
from .pbc_embedding import GRID_BLOCK_SIZE

if TYPE_CHECKING:
    from types import ModuleType
    from numpy.typing import NDArray
    from pydft_qmmm import System


def pulay_forces(
        backend: ModuleType,
        state: Any,
        natoms: int,
) -> NDArray[np.float64]:
    r"""The AO-derivative term for the added external operator.

    ``nuc_grad_method`` differentiates the SCF holding ``V_ao`` fixed.
    It does not know that ``V_ao`` itself moves with the nuclei, so

    .. math::

        \frac{\partial}{\partial R_A} \sum_g w_g V(g)
        \sum_{\mu\nu} D_{\mu\nu} \chi^*_\mu(g) \chi_\nu(g)

    is missing from it.  Because the basis functions depend on
    :math:`r - R_A`, the derivative with respect to the nucleus is
    minus the derivative with respect to the grid point, which is what
    ``eval_ao_kpts(deriv=1)`` returns.

    Args:
        backend: The package providing ``pbc.dft``.
        state: The converged SCF state.
        natoms: The number of atoms in the full system.

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) on every atom.
    """
    numint = load_submodule(backend, "pbc.dft.numint")
    cell = state.cell
    ao_slices = cell.aoslice_by_atom()
    dm = np.asarray(to_numpy(state.dm))
    nkpts = len(state.kpts)
    bohr = state.coords * BOHR_PER_ANGSTROM
    gradient = np.zeros((cell.natm, 3))
    for start in range(0, len(state.coords), GRID_BLOCK_SIZE):
        stop = start + GRID_BLOCK_SIZE
        block = np.ascontiguousarray(bohr[start:stop])
        scale = state.weights[start:stop] * state.potential[start:stop]
        # deriv=1 gives (4, ngrids, nao): the value, then d/dx, d/dy,
        # d/dz with respect to the grid point.
        ao_kpts = numint.eval_ao_kpts(
            cell, block, kpts=state.kpts, deriv=1,
        )
        for index, ao in enumerate(ao_kpts):
            ao = np.asarray(to_numpy(ao))
            value, derivs = ao[0], ao[1:4]
            for atom in range(cell.natm):
                first, last = ao_slices[atom][2], ao_slices[atom][3]
                # Two terms, one for the bra and one for the ket; the
                # density matrix is Hermitian so they are conjugates and
                # the real part doubles.
                gradient[atom] += -2.0 * np.einsum(
                    "xgi,g,gj,ij->x",
                    derivs[:, :, first:last].conj(),
                    scale,
                    value,
                    dm[index][first:last, :],
                ).real / nkpts
    forces = np.zeros((natoms, 3))
    forces[list(state.qm_indices)] -= gradient
    return forces


def nuclear_forces(
        state: Any,
        box: NDArray[np.float64],
        natoms: int,
) -> NDArray[np.float64]:
    r"""The force on the QM nuclei sitting in the external potential.

    One Fourier series supplies both the value used in the energy and
    the gradient used here, so this force is the exact derivative of
    that energy rather than an approximation of it.

    Args:
        state: The converged SCF state.
        box: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{\mathring{A}}`).
        natoms: The number of atoms in the full system.

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) on every atom.
    """
    cell = state.cell
    mesh = tuple(int(n) for n in cell.mesh)
    positions = np.asarray(cell.atom_coords()) / BOHR_PER_ANGSTROM
    _, gradient = spectral_value_and_gradient(
        state.potential.reshape(mesh),
        np.asarray(box, dtype=np.float64),
        positions,
    )
    forces = np.zeros((natoms, 3))
    # E = -sum(Z V), so F = -dE/dR = +Z grad V.  The gradient is per
    # Angstrom; the caller works in Eh/a0.
    charges = valence_charges(cell)[:, None]
    forces[list(state.qm_indices)] += (
        charges * gradient / BOHR_PER_ANGSTROM
    )
    return forces


def qm_forces(
        backend: ModuleType,
        state: Any,
        box: NDArray[np.float64],
        natoms: int,
) -> NDArray[np.float64]:
    r"""Channels 1 to 3: every force acting on a QM nucleus.

    Args:
        backend: The package providing ``pbc.dft``.
        state: The converged SCF state.
        box: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{\mathring{A}}`).
        natoms: The number of atoms in the full system.

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) on every atom.
    """
    forces = np.zeros((natoms, 3))
    gradient = state.method.nuc_grad_method()
    gradient.verbose = 0
    forces[list(state.qm_indices)] -= np.asarray(
        to_numpy(gradient.kernel()),
    )
    if state.potential.any():
        forces += pulay_forces(backend, state, natoms)
        forces += nuclear_forces(state, box, natoms)
    return forces


def qm_density_on_grid(
        backend: ModuleType,
        state: Any,
) -> NDArray[np.float64]:
    r"""The QM electronic charge at each grid point.

    Electrons carry negative charge, so this is negative where the
    density is.  The nuclei are added separately by ``mm_forces``,
    smeared with the same width the MM charges use so that the two
    halves of channel 4 stay consistent.

    Args:
        backend: The package providing ``pbc.dft``.
        state: The converged SCF state.

    Returns:
        The charge (:math:`e`) at each grid point.
    """
    numint = load_submodule(backend, "pbc.dft.numint")
    cell = state.cell
    dm = np.asarray(to_numpy(state.dm))
    nkpts = len(state.kpts)
    bohr = state.coords * BOHR_PER_ANGSTROM
    charge = np.zeros(len(state.coords))
    for start in range(0, len(state.coords), GRID_BLOCK_SIZE):
        stop = start + GRID_BLOCK_SIZE
        block = np.ascontiguousarray(bohr[start:stop])
        ao_kpts = numint.eval_ao_kpts(cell, block, kpts=state.kpts, deriv=0)
        density = np.zeros(len(block))
        for index, ao in enumerate(ao_kpts):
            ao = np.asarray(to_numpy(ao))
            density += np.einsum(
                "gi,ij,gj->g", ao.conj(), dm[index], ao,
            ).real / nkpts
        charge[start:stop] = -density * state.weights[start:stop]
    return charge


def mm_forces(
        backend: ModuleType,
        state: Any,
        system: System,
        sigma: float,
        natoms: int,
) -> NDArray[np.float64]:
    r"""Channel 4: the reaction of the QM density on the MM sites.

    The split mirrors the construction of V_ext exactly.  Subsystem II
    was embedded as smeared Gaussians, so its reaction comes from the
    adjoint of that smearing; subsystem III and the periodic images
    entered through helPME, so their reaction comes from the same
    helPME instance.  If the two ever disagree about which atoms they
    cover, the interaction is counted twice.

    MM charges are static, so the force is exactly
    :math:`q_j \mathbf{E}_{\mathrm{QM}}(R_j)` with no term from a
    charge that responds.

    Args:
        backend: The package providing ``pbc.dft``.
        state: The converged SCF state.
        system: The system holding positions and static charges.
        sigma: The Gaussian width (:math:`\mathrm{\mathring{A}}`).
        natoms: The number of atoms in the full system.

    Returns:
        The forces (:math:`\mathrm{E_h\;a_0^{-1}}`) on every atom.
    """
    cell = state.cell
    mesh = tuple(int(n) for n in cell.mesh)
    box = np.asarray(system.box, dtype=np.float64)
    forces = np.zeros((natoms, 3))
    charge = qm_density_on_grid(backend, state)

    indices = list(state.embed_indices)
    if indices:
        volume_element = abs(np.linalg.det(box)) / charge.size
        rho = charge.reshape(mesh) / volume_element
        rho = rho + spread_gaussian(
            np.asarray(cell.atom_coords()) / BOHR_PER_ANGSTROM,
            valence_charges(cell), mesh, box, sigma,
        )
        # poisson_fft returns volts; contract_gaussian_gradient then
        # returns eV/Angstrom.
        phi = poisson_fft(rho, box)
        forces[indices] += contract_gaussian_gradient(
            phi,
            np.asarray(system.positions)[indices],
            np.asarray(system.charges)[indices],
            mesh, box, sigma,
        ) * KJMOL_PER_EV / KJMOL_PER_EH / BOHR_PER_ANGSTROM

    # Subsystem III and the periodic images, through the same helPME
    # instance that built the reciprocal half of V_ext.
    #
    # The source is the WHOLE QM charge distribution: the electrons on
    # the grid and the valence nuclei as point charges.  Leaving the
    # nuclei out drops the reaction to channel 3 entirely, because
    # applies_nuclear_potential() tells the coupling Hamiltonian to skip
    # its own PMENuclearPotential -- the very term that would otherwise
    # have supplied it.  Owning the nuclear coupling means owning both
    # sides of it.
    if state.potentials:
        sources = np.vstack([
            state.coords,
            np.asarray(cell.atom_coords()) / BOHR_PER_ANGSTROM,
        ])
        strengths = np.concatenate([charge, valence_charges(cell)])
        for potential in state.potentials:
            reciprocal = np.asarray(
                potential.compute_source_forces(sources, strengths),
            )
            # compute_source_forces returns -grad(phi) * system.charges
            # for EVERY atom, and the QM atoms still carry their
            # force-field charges: conservative coupling zeroes those
            # inside OpenMM without touching System.charges.  The force
            # on a QM atom from V_rec is channels 2 and 3, so those rows
            # here are both spurious and wrongly weighted.
            reciprocal[list(state.qm_indices)] = 0.0
            forces += reciprocal / KJMOL_PER_EH / BOHR_PER_ANGSTROM
    return forces
