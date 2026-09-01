"""Electrostatic embedding for periodic PySCF.

The MM environment reaches the QM electrons as one real-space potential
sampled on the solver's own uniform FFT grid.  A uniform mesh cannot
integrate an all-electron density -- during the molecular work a mesh at
PME resolution integrated an sto-3g water density to 170 electrons
instead of 10 -- but GTH pseudopotentials remove the nuclear cusps, so
the uniform grid is adequate here.  This is the same reason the VASP
interface can use its plane-wave FFT grid, and it is why
``test_constant_potential_recovers_the_electron_count`` guards this
module.
"""
from __future__ import annotations

__all__ = [
    "GRID_BLOCK_SIZE",
    "grid_coordinates",
    "near_potential",
    "external_potential",
    "ao_operator",
]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.utils import BOHR_PER_ANGSTROM
from pydft_qmmm.utils import KJMOL_PER_EH
from pydft_qmmm.utils import KJMOL_PER_EV
from ..pyscf.pyscf_backend import load_submodule
from ..pyscf.pyscf_backend import to_numpy
from ..vasp.grid_potential import poisson_fft
from ..vasp.grid_potential import spread_gaussian

#: Quadrature points per AO batch.  The AO array is
#: (nkpts, block, nao) complex, so this bounds peak memory.
GRID_BLOCK_SIZE = 16384

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType
    from numpy.typing import NDArray
    from pydft_qmmm import System
    from pydft_qmmm.potentials import ElectronicPotential


def grid_coordinates(
        cell: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Get the cell's uniform FFT grid.

    Reusing the solver's own grid means the embedding integral and the
    exchange-correlation integral share one quadrature.

    Args:
        cell: A built periodic cell.

    Returns:
        The coordinates (:math:`\mathrm{\mathring{A}}`), shaped
        ``(ngrids, 3)`` in C order over the mesh, and the uniform
        quadrature weights (:math:`\mathrm{a_0^3}`).
    """
    coords = np.asarray(cell.get_uniform_grids())     # Bohr
    weights = np.full(len(coords), cell.vol / len(coords))
    return coords / BOHR_PER_ANGSTROM, weights


def _smeared_potential_grid(
        positions: NDArray[np.float64],
        charges: NDArray[np.float64],
        mesh: tuple[int, int, int],
        box: NDArray[np.float64],
        sigma: float,
) -> NDArray[np.float64]:
    r"""The electrostatic potential of smeared point charges.

    A point charge cannot be represented on a finite FFT grid, so each
    is smeared with width sigma and the periodic Poisson equation is
    solved by FFT.

    Args:
        positions: An Nx3 array of positions
            (:math:`\mathrm{\mathring{A}}`).
        charges: An N array of charges (:math:`e`).
        mesh: The grid dimensions.
        box: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{\mathring{A}}`).
        sigma: The Gaussian width (:math:`\mathrm{\mathring{A}}`).

    Returns:
        The electrostatic potential on the mesh, in volts.
    """
    rho = spread_gaussian(positions, charges, mesh, box, sigma)
    return poisson_fft(rho, box)


def near_potential(
        system: System,
        embed_indices: Sequence[int],
        mesh: tuple[int, int, int],
        box: NDArray[np.float64],
        sigma: float,
) -> NDArray[np.float64]:
    r"""The subsystem II near field as an electron potential energy.

    Args:
        system: The system holding positions and static charges.
        embed_indices: The original system indices of subsystem II.
        mesh: The grid dimensions.
        box: A 3x3 array whose rows are lattice vectors
            (:math:`\mathrm{\mathring{A}}`).
        sigma: The Gaussian width (:math:`\mathrm{\mathring{A}}`).

    Returns:
        The potential energy of one electron (:math:`\mathrm{E_h}`) on
        the mesh.
    """
    indices = list(embed_indices)
    if not indices:
        return np.zeros(mesh)
    positions = np.asarray(system.positions)[indices]
    charges = np.asarray(system.charges)[indices]
    volts = _smeared_potential_grid(positions, charges, mesh, box, sigma)
    # An electron carries charge -1, so its potential ENERGY is the
    # negative of the electrostatic potential.  Volts -> eV -> Eh.
    return -volts * KJMOL_PER_EV / KJMOL_PER_EH


def external_potential(
        system: System,
        cell: Any,
        potentials: Sequence[ElectronicPotential],
        embed_indices: Sequence[int],
        sigma: float,
) -> NDArray[np.float64]:
    r"""Assemble V_ext on the cell's uniform grid.

    The reciprocal half comes from helPME and already excludes
    ``not subsystem III``; the near-field half covers subsystem II.
    The two exclusion sets are complementary by construction, and the
    reaction forces in ``pbc_forces`` must split the same way or the
    interaction is counted twice.

    Args:
        system: The system holding positions, charges, and the box.
        cell: A built periodic cell.
        potentials: The registered electronic potentials.
        embed_indices: The original system indices of subsystem II.
        sigma: The Gaussian width (:math:`\mathrm{\mathring{A}}`).

    Returns:
        The potential energy of one electron (:math:`\mathrm{E_h}`) at
        every grid point, shaped ``(ngrids,)``.
    """
    coords, _ = grid_coordinates(cell)
    box = np.asarray(system.box, dtype=np.float64)
    mesh = tuple(int(n) for n in cell.mesh)
    total = np.zeros(len(coords))
    for potential in potentials:
        # compute_potential already returns Eh per electron and already
        # excludes "not subsystem III".
        total += np.asarray(potential.compute_potential(coords)).reshape(-1)
    # get_uniform_grids ravels the mesh in C order; so does reshape(-1).
    total += near_potential(
        system, embed_indices, mesh, box, sigma,
    ).reshape(-1)
    return total


def ao_operator(
        backend: ModuleType,
        cell: Any,
        kpts: NDArray[np.float64],
        coords: NDArray[np.float64],
        weights: NDArray[np.float64],
        potential: NDArray[np.float64],
) -> NDArray[np.complex128]:
    r"""Contract a real-space potential into a one-electron operator.

    This follows the template in ``pbc/df/fft.py`` for turning a
    real-space potential into an AO matrix: evaluate the AOs on the
    grid with the lattice sum, then contract against the weighted
    potential.

    Args:
        backend: The package providing ``pbc.dft``, i.e. ``pyscf`` or
            ``gpu4pyscf``, not one of their sub-modules.
        cell: A built periodic cell.
        kpts: The k-points, shaped ``(nkpts, 3)``.
        coords: The quadrature coordinates
            (:math:`\mathrm{\mathring{A}}`).
        weights: The quadrature weights (:math:`\mathrm{a_0^3}`).
        potential: The potential energy of one electron
            (:math:`\mathrm{E_h}`) at each coordinate.

    Returns:
        The one-electron operator (:math:`\mathrm{E_h}`), shaped
        ``(nkpts, nao, nao)``, on the host.
    """
    numint = load_submodule(backend, "pbc.dft.numint")
    nao = cell.nao_nr()
    kpts = np.asarray(kpts).reshape(-1, 3)
    matrix = np.zeros((len(kpts), nao, nao), dtype=np.complex128)
    bohr = coords * BOHR_PER_ANGSTROM
    for start in range(0, len(coords), GRID_BLOCK_SIZE):
        stop = start + GRID_BLOCK_SIZE
        block = np.ascontiguousarray(bohr[start:stop])
        scale = weights[start:stop] * potential[start:stop]
        ao_kpts = numint.eval_ao_kpts(cell, block, kpts=kpts, deriv=0)
        for index, ao in enumerate(ao_kpts):
            ao = to_numpy(ao)
            matrix[index] += np.einsum(
                "gi,g,gj->ij", ao.conj(), scale, ao,
            )
    return matrix
