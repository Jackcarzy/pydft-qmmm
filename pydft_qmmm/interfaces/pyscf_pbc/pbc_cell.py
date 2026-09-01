"""Construction of the periodic cell from the simulation box."""
from __future__ import annotations

__all__ = ["build_cell", "valence_charges"]

from typing import Any
from typing import TYPE_CHECKING

import numpy as np
from pyscf.pbc import gto

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pydft_qmmm import System


def build_cell(
        system: System,
        basis: str,
        pseudo: str,
        ke_cutoff: float | None,
        mesh: tuple[int, int, int] | None,
        charge: int,
        multiplicity: int,
        verbose: int,
) -> tuple[Any, tuple[int, ...]]:
    r"""Build the periodic cell for subsystem I.

    The QM cell is the simulation box itself, so the cell is rebuilt
    whenever the box or the positions change.

    The cell always comes from PySCF, on either device.  GPU4PySCF's
    ``pbc.gto`` is empty: it accelerates the solver, not the cell, and
    its ``pbc.dft`` solvers take an ordinary PySCF ``Cell``.

    Args:
        system: The system holding the box, positions, and elements.
        basis: The GTH basis set name.
        pseudo: The GTH pseudopotential name.
        ke_cutoff: The kinetic energy cutoff (:math:`\mathrm{E_h}`), or
            None when an explicit mesh is given.
        mesh: An explicit FFT mesh, or None when a cutoff is given.
        charge: The net charge (:math:`e`) of the QM subsystem.
        multiplicity: The spin multiplicity of the QM subsystem.
        verbose: The PySCF logging verbosity.

    Returns:
        The built cell and the original system indices of its atoms, in
        the order they appear in the cell.
    """
    qm_indices = tuple(sorted(system.select("subsystem I")))
    positions = np.asarray(system.positions)
    elements = system.elements
    cell = gto.Cell()
    cell.atom = [
        (str(elements[index]), tuple(float(x) for x in positions[index]))
        for index in qm_indices
    ]
    # PyDFT-QMMM owns Angstrom; Cell defaults to it, but say so.
    cell.unit = "Angstrom"
    cell.a = np.asarray(system.box, dtype=np.float64)
    cell.basis = basis
    cell.pseudo = pseudo
    cell.charge = charge
    cell.spin = multiplicity - 1
    cell.verbose = verbose
    if ke_cutoff is not None:
        cell.ke_cutoff = ke_cutoff
    else:
        cell.mesh = list(mesh)          # type: ignore[arg-type]
    cell.build()
    return cell, qm_indices


def valence_charges(cell: Any) -> NDArray[np.float64]:
    r"""Get the pseudopotential valence charge of every cell atom.

    Under a pseudopotential the nucleus carries only the valence
    charge.  Using the atomic number would over-count every nuclear
    term by the core electrons -- 8 rather than 6 for oxygen under
    gth-pbe.

    Args:
        cell: A built periodic cell.

    Returns:
        The valence charges (:math:`e`), ordered as ``cell.atom``.
    """
    return np.array(
        [cell.atom_charge(index) for index in range(cell.natm)],
        dtype=np.float64,
    )
