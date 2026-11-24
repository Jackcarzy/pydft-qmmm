"""Region III reciprocal embedding for a periodic QM cell."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydft_qmmm.potentials.pme_potential import helpme_py
from pydft_qmmm.potentials.pme_potential import PMEElectronicPotential
from pydft_qmmm.utils import KJMOL_PER_EH


@dataclass(frozen=True)
class PeriodicPMEElectronicPotential(PMEElectronicPotential):
    """Use region III-only reciprocal sources, including periodic images.

    I has its own periodic QM Hamiltonian; II has a periodic Gaussian field.
    Local erf exclusions would leave unwanted I/II image contributions.
    """

    excluded_indices: tuple[int, ...] = ()

    def _source_indices(self):
        region_iii = set(self.system.select("subsystem III"))
        return sorted(region_iii - set(self.excluded_indices))

    def _native_mesh(self, shape):
        """Whether the PME grid is the QM mesh and helPME can probe it directly.

        Same conditions as pme_grid.build_pme_potential / pme_source_forces
        (the VASP and SPARC native-grid route): matching dimensions with small
        prime factors, a positive diagonal box, spline order >= 4, and
        helpme-py >= 0.3.1 with compute_P_rec_grid.
        """
        from pydft_qmmm.potentials.embedding.pme_grid import _native_grid_shape
        box = np.asarray(self.system.box, dtype=np.float64)
        return (
            hasattr(helpme_py.PMEInstanceD, "compute_P_rec_grid")
            and np.lib.NumpyVersion(helpme_py.__version__) >= "0.3.1"
            and tuple(int(n) for n in self.pme_gridnumber) == tuple(shape)
            and _native_grid_shape(shape)
            and np.array_equal(box, np.diag(np.diag(box)))
            and np.all(np.diag(box) > 0)
            and int(self.pme_spline_order) >= 4
        )

    def compute_potential_on_mesh(self, shape):
        """Region III electron potential (Hartree) at every QM mesh node, C order.

        One helPME reciprocal pass on the native grid (compute_P_rec_grid)
        instead of B-spline interpolation at N_grid points. Returns None when
        the PME grid is not the QM mesh; the caller then uses compute_potential.
        Mesh node (i, j, k) sits at (i/Nx, j/Ny, k/Nz) in fractional
        coordinates, the same points as the PySCF uniform grid up to lattice
        translations.
        """
        shape = tuple(int(n) for n in shape)
        if not self._native_mesh(shape):
            return None
        indices = self._source_indices()
        values = np.zeros((int(np.prod(shape)), 1))
        if indices:
            self.pme.compute_P_rec_grid(
                0,
                helpme_py.MatrixD(np.ascontiguousarray(
                    self.system.charges[indices].reshape(-1, 1),
                )),
                helpme_py.MatrixD(np.ascontiguousarray(
                    self.system.positions[indices],
                )),
                helpme_py.MatrixD(values),
            )
        # helPME stores x fastest; the PySCF grid is C order (z fastest).
        potential = values[:, 0].reshape(shape[::-1]).transpose(2, 1, 0)
        return -np.ascontiguousarray(potential).reshape(-1) / KJMOL_PER_EH

    def compute_source_forces_on_mesh(self, grid_charges, shape):
        """Forces (kJ/mol/Å) on region III sources from QM charges on the mesh.

        The exact transpose of compute_potential_on_mesh
        (pme_grid_forces.native_grid_source_forces), so the MM forces are the
        derivative of the same discrete energy. Rows outside region III sources
        are zero. Returns None when the native route does not apply.
        """
        shape = tuple(int(n) for n in shape)
        if not self._native_mesh(shape):
            return None
        from pydft_qmmm.potentials.embedding.pme_grid import COULOMB_CONSTANT
        from pydft_qmmm.potentials.embedding.pme_grid_forces import (
            native_grid_source_forces,
        )
        forces = np.zeros((len(self.system), 3))
        indices = self._source_indices()
        if indices:
            forces[indices] = native_grid_source_forces(
                np.asarray(self.system.positions)[indices],
                np.asarray(self.system.charges)[indices],
                np.asarray(grid_charges, dtype=np.float64).reshape(shape),
                shape,
                np.diag(np.asarray(self.system.box, dtype=np.float64)),
                float(self.pme_alpha),
                int(self.pme_spline_order),
                COULOMB_CONSTANT,
            )
        return forces

    def compute_potential(self, coordinates):
        """Return the region III electron potential in Hartree."""
        indices = self._source_indices()
        potential = np.zeros((len(coordinates), 1))
        if indices:
            self.pme.compute_P_rec(
                0,
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        self.system.charges[indices].reshape(-1, 1),
                    ),
                ),
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        self.system.positions[indices],
                    ),
                ),
                helpme_py.MatrixD(np.ascontiguousarray(coordinates)),
                0,
                helpme_py.MatrixD(potential),
            )
        return -potential / KJMOL_PER_EH

    def _source_potential_and_derivs(self, coordinates, weights):
        """Return the reaction field at III sites; other system rows are zero.

        Columns: potential (kJ/mol/e), then x/y/z derivatives (kJ/mol/e/Å).
        """
        indices = self._source_indices()
        result = np.zeros((len(self.system), 4))
        if indices:
            potential = np.zeros((len(indices), 4))
            self.pme.compute_P_rec(
                0,
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        np.asarray(weights, dtype=float).reshape(-1, 1),
                    ),
                ),
                helpme_py.MatrixD(np.ascontiguousarray(coordinates)),
                helpme_py.MatrixD(
                    np.ascontiguousarray(
                        self.system.positions[indices],
                    ),
                ),
                1,
                helpme_py.MatrixD(potential),
            )
            result[indices] = potential
        return result
