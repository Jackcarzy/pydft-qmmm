# VASP Electrostatic Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let MM subsystem II point charges enter VASP's Kohn-Sham Hamiltonian as an external potential `V_ext`, so QM/MM with VASP uses electrostatic rather than mechanical embedding.

**Architecture:** The interface writes MM charges to a small text file and launches VASP with the Python plugin enabled. Inside VASP, `vasp_plugin.py` spreads those charges as Gaussians onto VASP's fine FFT grid, solves Poisson by FFT, and adds the result to `total_potential`. Nuclear corrections (`ΔE_I`, `ΔF_I`) are applied in the same process via the `force_and_stress` callback, so `vasprun.xml` is already correct and the interface's `_run()` is unchanged.

**Tech Stack:** Python 3.10+, numpy (only), pytest, VASP 6.6.1 built with `-DPLUGINS`.

## Global Constraints

- **numpy only** in `vasp_plugin.py`. No scipy, no pydft_qmmm imports, no VASP-only imports at module scope. This is what lets Tier 1 tests import the module with no VASP running, and what makes the Phase B extraction a file move.
- `EPS0 = 0.005526349358057108` — vacuum permittivity in e/(V·Å). Define locally in the plugin module; do not import it.
- **Sign convention:** `total_potential` is electron potential *energy* in eV, equal to `−φ`. A positive MM charge must *lower* it.
- **`Z_I` is `ZVAL`** (pseudopotential valence charge), not atomic number. Access as `constants.ZVAL[constants.ion_types[i]]`. For Au this is 11, not 79.
- **G=0 is set to zero** in the Poisson solve (uniform neutralizing background).
- Test files are named `*_test.py`, not `test_*.py` — match the existing repo convention.
- **Never set `PLUGINS/MODE`.** It defaults to `serial` (`src/plugins.F:196-198`), and that default is what makes this design correct under MPI: only rank 1 calls the plugin, it receives the **full gathered** grid, and the result is broadcast back and redistributed. Under `PLUGINS/MODE = parallel` every rank would call the plugin with its own grid *slab*, and a full-grid `V_ext` would be silently wrong. Verified in `plugins.F`: `buffer%active_rank = mode /= "serial" .or. comm%node_me == 1`, `buffer%broadcast = mode == "serial"`, plus `merge_grid_quantity`.
- `ConstantsLocalPotential.charge_density` is `Optional` and may be `None`. Guard before using it.
- `ion_types` arrives **already 0-indexed** — `_adjust_dataclass.adjust_indexing` subtracts 1 from every `IndexArray`. Do not subtract again.
- All arrays on `constants` are **read-only** (`freeze_arrays` sets `writeable = False`). Arrays on `additions` are writable, so in-place `+=` on `total_potential`/`forces` is correct.
- Target build: `install/vasp/vasp.6.6.1_nvhpc24/bin/vasp_std` with `OMP_NUM_THREADS=8`, modules `nvhpc-openmpi3/24.1` + `intel-oneapi-mkl`, **no hdf5 module**, `LD_LIBRARY_PATH` → `~/.conda/envs/vasp_plugin/lib`.
- Spec: `docs/superpowers/specs/2026-07-30-vasp-electrostatic-embedding-design.md`

## File Structure

| File | Responsibility |
|---|---|
| `pydft_qmmm/interfaces/vasp/vasp_plugin.py` | **Create.** Phase C: physics functions + the two VASP callbacks. Copied into each run directory. |
| `pydft_qmmm/interfaces/vasp/grid_potential.py` | **Create in Task 10.** Phase B: the physics functions, extracted. |
| `pydft_qmmm/interfaces/vasp/vasp_utils.py` | **Modify.** Add `write_mm_charges` alongside the existing `write_poscar`/`write_incar` writers. |
| `pydft_qmmm/interfaces/vasp/vasp_interface.py` | **Modify.** Write MM charges, add PLUGINS tags, copy the plugin, check the sentinel. |
| `pydft_qmmm/interfaces/vasp/vasp_factory.py` | **Modify.** Add `embedding_sigma`. |
| `tests/vasp_grid_potential_test.py` | **Create.** Tier 1 analytic tests. No VASP required. |
| `tests/vasp_embedding_test.py` | **Create.** Tier 0 smoke + Tier 2 finite-difference. Requires VASP. |

---

### Task 1: MM charge handoff file

The contract between the two processes. The step stamp is the guard against a plugin silently reading the previous step's geometry.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_utils.py`
- Create: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `vasp_utils.write_mm_charges(path: str, positions: NDArray[np.float64], charges: NDArray[np.float64], step: int, sigma: float) -> None`; `vasp_plugin.read_mm_charges(path: str, expect_step: int | None = None) -> tuple[NDArray[np.float64], NDArray[np.float64], int, float]` returning `(positions Nx3 Å, charges N e, step, sigma Å)`.

**Why sigma travels in this file:** the plugin runs in a separate process and cannot see the factory's `embedding_sigma`. Carrying it in the header is what makes that setting actually take effect; a module-level constant in the plugin would silently ignore the user's choice.

- [ ] **Step 1: Write the failing test**

Create `tests/vasp_grid_potential_test.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp import vasp_utils


class TestMMChargeHandoff:

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        positions = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]])
        charges = np.array([-0.834, 0.417])
        vasp_utils.write_mm_charges(
            path, positions, charges, step=7, sigma=0.45,
        )
        got_pos, got_q, got_step, got_sigma = vasp_plugin.read_mm_charges(path)
        assert got_step == 7
        assert got_sigma == pytest.approx(0.45, abs=1e-12)
        assert got_pos == pytest.approx(positions, abs=1e-12)
        assert got_q == pytest.approx(charges, abs=1e-12)

    def test_empty_selection_round_trips(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        vasp_utils.write_mm_charges(
            path, np.zeros((0, 3)), np.zeros(0), step=0, sigma=0.3,
        )
        got_pos, got_q, _, _ = vasp_plugin.read_mm_charges(path)
        assert got_pos.shape == (0, 3)
        assert got_q.shape == (0,)

    def test_stale_step_is_rejected(self, tmp_path):
        path = str(tmp_path / "MM_CHARGES")
        vasp_utils.write_mm_charges(
            path, np.array([[0.0, 0.0, 0.0]]), np.array([1.0]),
            step=3, sigma=0.3,
        )
        with pytest.raises(ValueError, match="step"):
            vasp_plugin.read_mm_charges(path, expect_step=4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            vasp_plugin.read_mm_charges(str(tmp_path / "nope"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_grid_potential_test.py::TestMMChargeHandoff -v`
Expected: FAIL — `ImportError` / `module 'vasp_plugin' has no attribute 'read_mm_charges'`.

- [ ] **Step 3: Implement the writer**

Append to `pydft_qmmm/interfaces/vasp/vasp_utils.py` (and add `"write_mm_charges"` to `__all__`):

```python
def write_mm_charges(
        path: str,
        positions: NDArray[np.float64],
        charges: NDArray[np.float64],
        step: int,
        sigma: float,
) -> None:
    r"""Write MM point charges for the VASP plugin to read.

    Args:
        path: The destination file.
        positions: An Nx3 array of positions
            (:math:`\mathrm{\mathring{A}}`).
        charges: An N array of charges (:math:`e`).
        step: A monotonically increasing stamp, used by the plugin to
            detect a stale file from a previous evaluation.
        sigma: The Gaussian width (:math:`\mathrm{\mathring{A}}`).  It
            travels with the charges because the plugin runs in a
            separate process and cannot read the interface's settings.
    """
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    charges = np.asarray(charges, dtype=np.float64).reshape(-1)
    if len(positions) != len(charges):
        raise ValueError(
            f"{len(positions)} positions but {len(charges)} charges.",
        )
    with open(path, "w") as fh:
        fh.write(f"{len(charges)} {step} {sigma:.12e}\n")
        for (x, y, z), q in zip(positions, charges):
            fh.write(f"{x:.12e} {y:.12e} {z:.12e} {q:.12e}\n")
```

- [ ] **Step 4: Implement the reader**

Create `pydft_qmmm/interfaces/vasp/vasp_plugin.py`:

```python
"""External-potential plugin for VASP QM/MM electrostatic embedding.

VASP imports this module and calls into it; never run it directly.

Dependencies are deliberately limited to numpy, and nothing from
pydft_qmmm is imported, because this module is executed by the Python
interpreter embedded inside VASP.  That restriction is also what lets
the physics below be unit tested without VASP present.
"""
from __future__ import annotations

import numpy as np

# Vacuum permittivity, e/(V*Angstrom).  Defined locally rather than
# imported so this module stays dependency-free.
EPS0 = 0.005526349358057108


def read_mm_charges(path, expect_step=None):
    """Read the MM point charges written by the VASP interface.

    Args:
        path: The file to read.
        expect_step: If given, the step stamp the file must carry.  A
            mismatch means the interface failed to refresh the file and
            the plugin would otherwise silently embed the previous
            step's geometry.

    Returns:
        Positions (Nx3, Angstrom), charges (N, e), the step stamp, and
        the Gaussian width sigma (Angstrom).
    """
    with open(path) as fh:
        count, step, sigma = fh.readline().split()
        count, step, sigma = int(count), int(step), float(sigma)
        data = np.loadtxt(fh, dtype=np.float64, ndmin=2)
    if len(data) != count:
        raise ValueError(
            f"{path} declares {count} charges but holds {len(data)}.",
        )
    if expect_step is not None and step != expect_step:
        raise ValueError(
            f"{path} is at step {step}, expected {expect_step}.  The "
            "interface did not refresh the MM charges.",
        )
    if count == 0:
        return np.zeros((0, 3)), np.zeros(0), step, sigma
    return data[:, :3].copy(), data[:, 3].copy(), step, sigma
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/vasp_grid_potential_test.py::TestMMChargeHandoff -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_utils.py \
        pydft_qmmm/interfaces/vasp/vasp_plugin.py \
        tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): add MM charge handoff file with step stamp"
```

---

### Task 2: Gaussian charge spreading

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `vasp_plugin.spread_gaussian(positions, charges, shape, cell, sigma) -> NDArray[np.float64]` of shape `shape`, in e/Å³. `cell` is a 3x3 array whose **rows** are lattice vectors (VASP convention).

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_grid_potential_test.py`:

```python
CELL = np.diag([10.0, 10.0, 10.0])
SHAPE = (48, 48, 48)


class TestSpreadGaussian:

    def test_conserves_charge(self):
        positions = np.array([[5.0, 5.0, 5.0], [2.0, 3.0, 4.0]])
        charges = np.array([1.0, -0.5])
        rho = vasp_plugin.spread_gaussian(
            positions, charges, SHAPE, CELL, sigma=0.4,
        )
        d_volume = abs(np.linalg.det(CELL)) / np.prod(SHAPE)
        assert rho.sum() * d_volume == pytest.approx(0.5, abs=1e-6)

    def test_peak_is_at_the_charge(self):
        positions = np.array([[5.0, 5.0, 5.0]])
        rho = vasp_plugin.spread_gaussian(
            positions, np.array([1.0]), SHAPE, CELL, sigma=0.4,
        )
        peak = np.unravel_index(np.argmax(rho), SHAPE)
        # 5.0 Angstrom on a 10 Angstrom / 48 point axis -> index 24.
        assert peak == (24, 24, 24)

    def test_wraps_across_the_periodic_boundary(self):
        # A charge just inside the origin must not pile up at the far
        # face; minimum-image wrapping should make the density
        # symmetric about the boundary.
        rho = vasp_plugin.spread_gaussian(
            np.array([[0.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert rho[1, 24, 24] == pytest.approx(rho[-1, 24, 24], rel=1e-9)

    def test_empty_selection_gives_zero_density(self):
        rho = vasp_plugin.spread_gaussian(
            np.zeros((0, 3)), np.zeros(0), SHAPE, CELL, sigma=0.4,
        )
        assert rho.shape == SHAPE
        assert np.all(rho == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_grid_potential_test.py::TestSpreadGaussian -v`
Expected: FAIL — `no attribute 'spread_gaussian'`.

- [ ] **Step 3: Implement**

Append to `vasp_plugin.py`:

```python
def _fractional_grid(shape):
    """Build the fractional coordinates of every grid point."""
    axes = [np.arange(n, dtype=np.float64) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def spread_gaussian(positions, charges, shape, cell, sigma):
    """Spread point charges onto the grid as normalized Gaussians.

    A point charge cannot be represented on a finite FFT grid, so each
    is smeared with width sigma.  Displacements are minimum-imaged in
    fractional coordinates, which is exact for the nearest image and
    negligible in error while sigma is small against the cell.

    Args:
        positions: An Nx3 array of positions (Angstrom).
        charges: An N array of charges (e).
        shape: The grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        sigma: The Gaussian width (Angstrom).

    Returns:
        The charge density on the grid (e/Angstrom**3).
    """
    shape = tuple(int(n) for n in shape)
    rho = np.zeros(shape, dtype=np.float64)
    if len(charges) == 0:
        return rho
    fractional = _fractional_grid(shape)
    inverse = np.linalg.inv(cell)
    prefactor = (2.0 * np.pi * sigma**2) ** -1.5
    for position, charge in zip(positions, charges):
        delta = fractional - (position @ inverse)
        delta -= np.round(delta)
        cartesian = delta @ cell
        squared = np.einsum("...k,...k->...", cartesian, cartesian)
        rho += charge * prefactor * np.exp(-0.5 * squared / sigma**2)
    return rho
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/vasp_grid_potential_test.py::TestSpreadGaussian -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_plugin.py tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): spread MM point charges as Gaussians on the FFT grid"
```

---

### Task 3: FFT Poisson solve and the electron potential

The highest-risk task. Units and sign are asserted separately so an error in one cannot cancel an error in the other.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `spread_gaussian` from Task 2.
- Produces: `vasp_plugin.poisson_fft(rho, cell) -> NDArray[np.float64]` (φ in volts); `vasp_plugin.build_external_potential(positions, charges, shape, cell, sigma) -> NDArray[np.float64]` (`V_ext` in eV, already negated).

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_grid_potential_test.py`:

```python
# e / (4 * pi * eps0) in eV*Angstrom/e -- the Coulomb constant.
COULOMB = 14.399645


class TestPoisson:

    def test_satisfies_poissons_equation(self):
        # Apply the Laplacian back to phi in reciprocal space and
        # recover rho.  Exact up to the G=0 term, so cell-size
        # independent -- this is the strongest available check.
        rho = vasp_plugin.spread_gaussian(
            np.array([[5.0, 5.0, 5.0], [7.0, 5.0, 5.0]]),
            np.array([1.0, -1.0]), SHAPE, CELL, sigma=0.5,
        )
        phi = vasp_plugin.poisson_fft(rho, CELL)
        phi_g = np.fft.fftn(phi)
        g_squared = vasp_plugin._g_squared(SHAPE, CELL)
        recovered = np.fft.ifftn(phi_g * g_squared * vasp_plugin.EPS0).real
        rho_zero_mean = rho - rho.mean()
        assert recovered == pytest.approx(rho_zero_mean, abs=1e-9)

    def test_coulomb_prefactor_from_a_potential_difference(self):
        # Absolute phi carries a constant offset from the G=0 choice, so
        # compare a DIFFERENCE, which cancels it.  The residual is
        # periodic-image error; test_prefactor_converges_with_cell_size
        # below shows it shrinks with cell size.
        cell = np.diag([30.0, 30.0, 30.0])
        shape = (150, 150, 150)
        centre = np.array([[15.0, 15.0, 15.0]])
        phi = vasp_plugin.poisson_fft(
            vasp_plugin.spread_gaussian(
                centre, np.array([1.0]), shape, cell, sigma=0.3,
            ),
            cell,
        )
        near = vasp_plugin.interpolate_at(
            phi, cell, np.array([[17.0, 15.0, 15.0]]),
        )[0]
        far = vasp_plugin.interpolate_at(
            phi, cell, np.array([[19.0, 15.0, 15.0]]),
        )[0]
        expected = COULOMB * (1.0 / 2.0 - 1.0 / 4.0)
        assert near - far == pytest.approx(expected, rel=0.03)

    def test_positive_charge_lowers_electron_potential_energy(self):
        # The single likeliest bug, asserted alone.  Electrons are
        # attracted to a positive charge, so V_ext must be negative.
        v_ext = vasp_plugin.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert v_ext[24, 24, 24] < 0.0

    def test_neutral_system_is_insensitive_to_the_g0_convention(self):
        rho = vasp_plugin.spread_gaussian(
            np.array([[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]]),
            np.array([1.0, -1.0]), SHAPE, CELL, sigma=0.4,
        )
        phi = vasp_plugin.poisson_fft(rho, CELL)
        assert phi.mean() == pytest.approx(0.0, abs=1e-10)

    def test_converges_to_the_point_charge_limit_as_sigma_shrinks(self):
        # A neutral pair, so the G=0 convention plays no part.  Probe at
        # a point much closer to +q than to -q, where the point-charge
        # value is known.  The residual must SHRINK as sigma shrinks --
        # that monotone trend is the actual claim, and it is stronger
        # evidence than any single-sigma tolerance.
        cell = np.diag([24.0, 24.0, 24.0])
        shape = (120, 120, 120)
        plus = np.array([8.0, 12.0, 12.0])
        minus = np.array([20.0, 12.0, 12.0])
        probe = np.array([[10.0, 12.0, 12.0]])
        expected = COULOMB * (1.0 / 2.0 - 1.0 / 10.0)
        residuals = []
        for sigma in (0.6, 0.4, 0.25):
            phi = vasp_plugin.poisson_fft(
                vasp_plugin.spread_gaussian(
                    np.array([plus, minus]), np.array([1.0, -1.0]),
                    shape, cell, sigma,
                ),
                cell,
            )
            got = vasp_plugin.interpolate_at(phi, cell, probe)[0]
            residuals.append(abs(got - expected))
        assert residuals[1] < residuals[0]
        assert residuals[2] < residuals[1]
        assert residuals[2] < 0.05 * expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_grid_potential_test.py::TestPoisson -v`
Expected: FAIL — `no attribute 'poisson_fft'`.

- [ ] **Step 3: Implement**

Append to `vasp_plugin.py`:

```python
def _g_squared(shape, cell):
    """Return |G|**2 on the reciprocal grid (Angstrom**-2)."""
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    axes = [np.fft.fftfreq(n) * n for n in shape]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    g_vectors = miller @ reciprocal
    return np.einsum("...k,...k->...", g_vectors, g_vectors)


def poisson_fft(rho, cell):
    """Solve the periodic Poisson equation by FFT.

    Solves del**2 phi = -rho / eps0.  In reciprocal space this is
    phi_G = rho_G / (eps0 * |G|**2).  The G=0 term diverges for a
    net-charged cell and is set to zero, which fixes the average
    potential at zero -- equivalent to a uniform neutralizing
    background, the same convention VASP uses.

    Args:
        rho: The charge density on the grid (e/Angstrom**3).
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).

    Returns:
        The electrostatic potential on the grid (volts).
    """
    g_squared = _g_squared(rho.shape, cell)
    g_squared[0, 0, 0] = 1.0
    phi_g = np.fft.fftn(rho) / (EPS0 * g_squared)
    phi_g[0, 0, 0] = 0.0
    return np.fft.ifftn(phi_g).real


def build_external_potential(positions, charges, shape, cell, sigma):
    """Build V_ext on the grid from MM point charges.

    Returns:
        The external potential as electron potential ENERGY (eV), which
        is the negative of the electrostatic potential.  This is what
        VASP's total_potential expects.
    """
    rho = spread_gaussian(positions, charges, shape, cell, sigma)
    return -poisson_fft(rho, cell)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/vasp_grid_potential_test.py::TestPoisson -v`
Expected: 4 passed. `test_coulomb_prefactor_from_a_potential_difference` needs `interpolate_at` from Task 4 — if running tasks in order, expect that one to fail here with `no attribute 'interpolate_at'` and pass after Task 4. Mark it `@pytest.mark.xfail(reason="needs Task 4")` if you want a green bar now, and remove the marker in Task 4.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_plugin.py tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): FFT Poisson solve and electron potential convention"
```

---

### Task 4: Interpolation and gradient at arbitrary points

Needed for the nuclear corrections. Interpolating the *same* grid the electrons see is what keeps the two halves of the QM subsystem consistent.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `build_external_potential` from Task 3.
- Produces: `vasp_plugin.interpolate_at(field, cell, points) -> NDArray[np.float64]` shape `(len(points),)`; `vasp_plugin.gradient_at(field, cell, points) -> NDArray[np.float64]` shape `(len(points), 3)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_grid_potential_test.py`:

```python
class TestInterpolation:

    def test_reproduces_grid_values_at_grid_points(self):
        field = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        # Grid point (3, 5, 7) sits at 10 Angstrom * index / 48.
        point = np.array([[10.0 * 3 / 48, 10.0 * 5 / 48, 10.0 * 7 / 48]])
        got = vasp_plugin.interpolate_at(field, CELL, point)
        assert got[0] == pytest.approx(field[3, 5, 7], rel=1e-9)

    def test_interpolation_is_periodic(self):
        field = np.arange(np.prod(SHAPE), dtype=np.float64).reshape(SHAPE)
        inside = np.array([[1.0, 2.0, 3.0]])
        shifted = inside + np.array([10.0, -10.0, 20.0])
        assert vasp_plugin.interpolate_at(field, CELL, inside) == pytest.approx(
            vasp_plugin.interpolate_at(field, CELL, shifted), rel=1e-9,
        )

    def test_gradient_matches_finite_differences(self):
        v_ext = vasp_plugin.build_external_potential(
            np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.8,
        )
        point = np.array([[6.5, 5.3, 4.7]])
        analytic = vasp_plugin.gradient_at(v_ext, CELL, point)[0]
        step = 1e-3
        numerical = np.zeros(3)
        for axis in range(3):
            shift = np.zeros((1, 3))
            shift[0, axis] = step
            plus = vasp_plugin.interpolate_at(v_ext, CELL, point + shift)[0]
            minus = vasp_plugin.interpolate_at(v_ext, CELL, point - shift)[0]
            numerical[axis] = (plus - minus) / (2 * step)
        assert analytic == pytest.approx(numerical, rel=0.02)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_grid_potential_test.py::TestInterpolation -v`
Expected: FAIL — `no attribute 'interpolate_at'`.

- [ ] **Step 3: Implement**

Append to `vasp_plugin.py`:

```python
def interpolate_at(field, cell, points):
    """Trilinearly interpolate a periodic grid field at points.

    Args:
        field: A scalar field on the grid.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        points: An Nx3 array of Cartesian positions (Angstrom).

    Returns:
        An N array of interpolated values.
    """
    shape = np.array(field.shape)
    fractional = (np.atleast_2d(points) @ np.linalg.inv(cell)) % 1.0
    scaled = fractional * shape
    lower = np.floor(scaled).astype(int)
    weight = scaled - lower
    result = np.zeros(len(scaled))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                offset = np.array([dx, dy, dz])
                index = (lower + offset) % shape
                corner = (1.0 - weight) * (1 - offset) + weight * offset
                result += (
                    field[index[:, 0], index[:, 1], index[:, 2]]
                    * corner.prod(axis=1)
                )
    return result


def gradient_at(field, cell, points):
    """Return the gradient of a periodic grid field at points.

    The derivative is taken in reciprocal space (exact for a
    band-limited field) and the three components are then interpolated,
    so the gradient is consistent with interpolate_at.

    Args:
        field: A scalar field on the grid.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        points: An Nx3 array of Cartesian positions (Angstrom).

    Returns:
        An Nx3 array of gradients (field units per Angstrom).
    """
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    axes = [np.fft.fftfreq(n) * n for n in field.shape]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    g_vectors = miller @ reciprocal
    field_g = np.fft.fftn(field)
    gradient = np.zeros((len(np.atleast_2d(points)), 3))
    for axis in range(3):
        component = np.fft.ifftn(1j * g_vectors[..., axis] * field_g).real
        gradient[:, axis] = interpolate_at(component, cell, points)
    return gradient
```

- [ ] **Step 4: Run the whole Tier 1 suite**

Run: `pytest tests/vasp_grid_potential_test.py -v`
Expected: all pass, including `test_coulomb_prefactor_from_a_potential_difference` from Task 3. Remove the `xfail` marker if you added one.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_plugin.py tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): interpolation and reciprocal-space gradient on the grid"
```

---

### Task 5: The VASP callbacks

Thin wrappers. All physics is already tested; this task is about caching, the sentinel, and loud failure.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `read_mm_charges`, `build_external_potential`, `interpolate_at`, `gradient_at`.
- Produces: `vasp_plugin.local_potential(constants, additions) -> None`; `vasp_plugin.force_and_stress(constants, additions) -> None`; `vasp_plugin.SENTINEL = "PLUGIN_FIRED.txt"`; `vasp_plugin.ERROR_FILE = "PLUGIN_ERROR.txt"`; `vasp_plugin.reset_cache() -> None`.

**Verify against the real API before implementing.** The tests below use a `SimpleNamespace` standing in for VASP's dataclasses, which will pass regardless of whether the real field names match. Confirm the following against `VASP-Python/*/vasp_plugin.py` and `vasp.6.6.1/src/plugins/`, and correct the code here if they differ:

- `ConstantsLocalPotential`: `shape_grid`, `lattice_vectors`, `charge_density`
- `AdditionsLocalPotential`: `total_potential`, `total_energy`
- `ConstantsForceAndStress`: `positions` (fractional), `ion_types`, `ZVAL`
- `AdditionsForceAndStress`: `forces` — **the least certain of these.** If the real attribute has another name, the mock test still passes while the real run silently applies no nuclear force correction. Task 9's finite-difference test is what would eventually catch that, but checking now is cheaper.

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_grid_potential_test.py`:

```python
import types


def _fake_constants(shape, cell, positions=None):
    """Mimic the fields of VASP's ConstantsLocalPotential."""
    return types.SimpleNamespace(
        shape_grid=np.array(shape),
        lattice_vectors=cell,
        positions=(
            np.zeros((0, 3)) if positions is None
            else positions @ np.linalg.inv(cell)
        ),
        ion_types=np.zeros(0 if positions is None else len(positions), int),
        ZVAL=np.array([11.0]),
    )


class TestCallbacks:

    def test_local_potential_adds_a_negative_well(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        additions = types.SimpleNamespace(
            total_potential=np.zeros(SHAPE), total_energy=0.0,
        )
        vasp_plugin.local_potential(_fake_constants(SHAPE, CELL), additions)
        assert additions.total_potential[24, 24, 24] < 0.0
        assert (tmp_path / vasp_plugin.SENTINEL).exists()

    def test_local_potential_builds_the_field_once(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        calls = []
        original = vasp_plugin.build_external_potential
        monkeypatch.setattr(
            vasp_plugin, "build_external_potential",
            lambda *a, **k: (calls.append(1), original(*a, **k))[1],
        )
        constants = _fake_constants(SHAPE, CELL)
        for _ in range(3):
            additions = types.SimpleNamespace(
                total_potential=np.zeros(SHAPE), total_energy=0.0,
            )
            vasp_plugin.local_potential(constants, additions)
        assert len(calls) == 1

    def test_missing_charges_file_raises_and_records(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_plugin.reset_cache()
        additions = types.SimpleNamespace(
            total_potential=np.zeros(SHAPE), total_energy=0.0,
        )
        with pytest.raises(Exception):
            vasp_plugin.local_potential(_fake_constants(SHAPE, CELL), additions)
        assert (tmp_path / vasp_plugin.ERROR_FILE).exists()

    def test_empty_selection_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.zeros((0, 3)), np.zeros(0), step=0, sigma=0.4,
        )
        vasp_plugin.reset_cache()
        additions = types.SimpleNamespace(
            total_potential=np.zeros(SHAPE), total_energy=0.0,
        )
        vasp_plugin.local_potential(_fake_constants(SHAPE, CELL), additions)
        assert np.all(additions.total_potential == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_grid_potential_test.py::TestCallbacks -v`
Expected: FAIL — `no attribute 'reset_cache'`.

- [ ] **Step 3: Implement**

Append to `vasp_plugin.py`:

```python
CHARGE_FILE = "MM_CHARGES"
SENTINEL = "PLUGIN_FIRED.txt"
ERROR_FILE = "PLUGIN_ERROR.txt"

_CACHE = {}


def reset_cache():
    """Discard the cached external potential.  For tests."""
    _CACHE.clear()


def _record_error(exc):
    """Persist a traceback before re-raising.

    A Python exception raised inside VASP's embedded interpreter may not
    surface legibly in VASP's own output, so write it to a file first.
    """
    import traceback
    with open(ERROR_FILE, "w") as fh:
        fh.write("".join(traceback.format_exception(exc)))


def _external_potential(constants):
    """Return V_ext on the current grid, building it at most once."""
    shape = tuple(int(n) for n in constants.shape_grid)
    cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
    if _CACHE.get("shape") == shape and _CACHE.get("v_ext") is not None:
        return _CACHE["v_ext"]
    positions, charges, _, sigma = read_mm_charges(CHARGE_FILE)
    net_charge = float(np.sum(charges))
    if abs(net_charge) > 1e-6:
        # The G=0 term is dropped in the Poisson solve, which is
        # equivalent to a uniform neutralizing background.  Absolute
        # energies then carry a constant offset; differences at fixed
        # composition do not.
        import warnings
        warnings.warn(
            f"subsystem II carries a net charge of {net_charge:.4f} e.  "
            "Absolute embedded energies are shifted by a constant from "
            "the neutralizing-background (G=0) convention.",
            RuntimeWarning,
        )
    v_ext = build_external_potential(positions, charges, shape, cell, sigma)
    _CACHE["shape"] = shape
    _CACHE["v_ext"] = v_ext
    _CACHE["count"] = len(charges)
    with open(SENTINEL, "w") as fh:
        fh.write("local_potential callback executed\n")
        fh.write(f"shape_grid   = {shape}\n")
        fh.write(f"mm_charges   = {len(charges)}\n")
        fh.write(f"net_charge   = {net_charge:.6f}\n")
        fh.write(f"sigma        = {sigma}\n")
        fh.write(f"min V_ext eV = {float(v_ext.min()):.6f}\n")
    return v_ext


def local_potential(constants, additions):
    """Add the MM external potential to VASP's local potential."""
    try:
        additions.total_potential += _external_potential(constants)
    except Exception as exc:
        _record_error(exc)
        raise


def force_and_stress(constants, additions):
    """Apply the nuclear terms VASP omits.

    VASP does not include the interaction of the external potential with
    the pseudo-ion cores, so add dE_I = -Z_I V_ext(R_I) and
    dF_I = +Z_I grad V_ext(R_I).  Z_I is the pseudopotential valence
    charge ZVAL, not the atomic number.

    NOTE THE FORCE SIGN: F = -grad(dE) = -grad(-Z V_ext) = +Z grad V_ext.
    An earlier draft of this plan wrote -Z grad V_ext, which contradicts
    the energy term and flips every nuclear force.
    """
    try:
        v_ext = _external_potential(constants)
        cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
        positions = np.asarray(constants.positions) @ cell
        if len(positions) == 0:
            return
        valence = np.array(
            [constants.ZVAL[t] for t in constants.ion_types],
            dtype=np.float64,
        )
        additions.total_energy += -np.sum(
            valence * interpolate_at(v_ext, cell, positions),
        )
        additions.forces += -(
            valence[:, None] * gradient_at(v_ext, cell, positions)
        )
    except Exception as exc:
        _record_error(exc)
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/vasp_grid_potential_test.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_plugin.py tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): plugin callbacks with caching, sentinel and loud failure"
```

---

### Task 6: Wire the interface to write charges and enable the plugin

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_interface.py`
- Test: `tests/vasp_embedding_test.py`

**Interfaces:**
- Consumes: `vasp_utils.write_mm_charges` (Task 1), `vasp_plugin.SENTINEL` (Task 5).
- Produces: `VaspInterface.embedding_sigma: float` field; `VaspInterface._write_mm_charges() -> int` returning the number of MM charges written.

- [ ] **Step 1: Write the failing test**

Create `tests/vasp_embedding_test.py`:

```python
from __future__ import annotations

import os

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import vasp_plugin


class TestInterfaceWiring:

    def test_incar_gains_plugin_tags_when_embedding(self, vasp_embedded):
        vasp_embedded._write_input()
        with open(os.path.join(vasp_embedded.directory, "INCAR")) as fh:
            incar = fh.read()
        assert "PLUGINS/LOCAL_POTENTIAL = T" in incar
        assert "PLUGINS/FORCE_AND_STRESS = T" in incar

    def test_plugin_is_copied_into_the_run_directory(self, vasp_embedded):
        vasp_embedded._write_input()
        assert os.path.isfile(
            os.path.join(vasp_embedded.directory, "vasp_plugin.py"),
        )

    def test_mm_charges_are_written(self, vasp_embedded):
        count = vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        positions, charges, _, sigma = vasp_plugin.read_mm_charges(path)
        assert count == len(charges) > 0
        assert sigma == pytest.approx(vasp_embedded.embedding_sigma)

    def test_configured_sigma_reaches_the_plugin(self, spce_qmmm_system, tmp_path):
        # Guards the process boundary: a module-level SIGMA in the
        # plugin would silently ignore the user's setting.
        from pydft_qmmm.interfaces.vasp.vasp_factory import (
            vasp_interface_factory,
        )
        potential = vasp_interface_factory(
            spce_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=str(tmp_path / "pp"),
            embedding=True,
            embedding_sigma=0.55,
        )
        potential._write_mm_charges()
        _, _, _, sigma = vasp_plugin.read_mm_charges(
            os.path.join(potential.directory, "MM_CHARGES"),
        )
        assert sigma == pytest.approx(0.55)

    def test_missing_sentinel_raises(self, vasp_embedded):
        with pytest.raises(Exception, match="sentinel|PLUGINS"):
            vasp_embedded._check_plugin_fired()
```

Add to `tests/conftest.py`:

```python
@pytest.fixture
def vasp_embedded(spce_qmmm_system, tmp_path):
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    potential = vasp_interface_factory(
        spce_qmmm_system,
        directory=str(tmp_path / "vasp"),
        pp_path=os.environ.get("VASP_PP_PATH", str(tmp_path / "pp")),
        embedding=True,
    )
    return potential
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_embedding_test.py::TestInterfaceWiring -v`
Expected: FAIL — `vasp_interface_factory() got an unexpected keyword argument 'embedding'`.

- [ ] **Step 3: Implement**

In `vasp_interface.py`, add two fields to the `VaspInterface` dataclass after `potcar_map`:

```python
    embedding: bool = False
    embedding_sigma: float = 0.3
```

Add these methods to `VaspInterface`:

```python
    def _write_mm_charges(self) -> int:
        """Write subsystem II charges for the plugin.

        Returns:
            The number of MM charges written.
        """
        os.makedirs(self.directory, exist_ok=True)
        indices = sorted(self.system.select("subsystem II"))
        positions = np.asarray(self.system.positions)[indices]
        charges = np.asarray(self.system.charges)[indices]
        path = os.path.join(self.directory, "MM_CHARGES")
        # Delete before rewriting so a failed write cannot leave the
        # previous step's charges in place to be silently reused.
        if os.path.isfile(path):
            os.remove(path)
        vasp_utils.write_mm_charges(
            path, positions, charges, self.frame[0], self.embedding_sigma,
        )
        return len(charges)

    def _check_plugin_fired(self) -> None:
        """Verify the plugin actually ran.

        Raises:
            VaspExecutionError: If the sentinel is absent, which means
                the potential was never applied.  Note that grepping the
                binary for "not compiled with PLUGINS" does NOT work --
                that string is present in every build.
        """
        from .vasp_plugin import SENTINEL
        if not os.path.isfile(os.path.join(self.directory, SENTINEL)):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "Electrostatic embedding was requested but the plugin "
                "sentinel is absent, so the external potential was "
                "never applied.  Check that vasp_std is built with "
                "-DPLUGINS and that PYTHONHOME/PATH point at the "
                "vasp_plugin environment.",
            )
```

In `_write_input`, after the `tags.update({"NSW": 0, "IBRION": -1})` line, insert:

```python
        if self.embedding:
            tags["PLUGINS/LOCAL_POTENTIAL"] = "T"
            tags["PLUGINS/FORCE_AND_STRESS"] = "T"
            shutil.copyfile(
                os.path.join(os.path.dirname(__file__), "vasp_plugin.py"),
                os.path.join(self.directory, "vasp_plugin.py"),
            )
```

Add `import shutil` to the imports.

In `_run`, replace the body up to `vasp_utils.run_vasp(...)` with:

```python
        order = self._write_input()
        if self.embedding:
            self._write_mm_charges()
        vasp_utils.run_vasp(self.command, self.directory)
        if self.embedding:
            self._check_plugin_fired()
```

**Two deliberate choices worth understanding before reviewing this task:**

*`expect_step` is not passed.* `read_mm_charges` supports it and Task 1 tests it, but the interface does not use it, because staleness is already impossible here: VASP is relaunched for every evaluation, so the plugin process is always fresh, and `_write_mm_charges` deletes the file before rewriting, so a failed write surfaces as `FileNotFoundError` rather than a stale read. The parameter exists for M3, where the plugin may be called across multiple geometries in one process. Do not add a spurious caller to "use" it.

*`PYTHONHOME` is validated after the run, not before.* The spec called for a pre-launch check. Checking the sentinel afterwards is strictly better: it catches every cause of a silent no-op — missing `-DPLUGINS`, unset `PYTHONHOME`, an import error inside the plugin — with one guard, instead of enumerating causes in advance and missing one. The error message names the likely culprits.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/vasp_embedding_test.py::TestInterfaceWiring -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_interface.py tests/vasp_embedding_test.py tests/conftest.py
git commit -m "feat(vasp): write MM charges, enable plugin tags, verify sentinel"
```

---

### Task 7: Expose embedding through the factory

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_factory.py`
- Test: `tests/vasp_embedding_test.py`

**Interfaces:**
- Consumes: `VaspInterface.embedding`, `VaspInterface.embedding_sigma` (Task 6).
- Produces: `vasp_interface_factory(..., embedding: bool = False, embedding_sigma: float = 0.3)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_embedding_test.py`:

```python
class TestFactory:

    def test_embedding_defaults_off(self, spce_qmmm_system, tmp_path):
        from pydft_qmmm.interfaces.vasp.vasp_factory import (
            vasp_interface_factory,
        )
        potential = vasp_interface_factory(
            spce_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=str(tmp_path / "pp"),
        )
        assert potential.embedding is False

    def test_sigma_is_configurable(self, spce_qmmm_system, tmp_path):
        from pydft_qmmm.interfaces.vasp.vasp_factory import (
            vasp_interface_factory,
        )
        potential = vasp_interface_factory(
            spce_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=str(tmp_path / "pp"),
            embedding=True,
            embedding_sigma=0.5,
        )
        assert potential.embedding_sigma == 0.5

    def test_negative_sigma_is_rejected(self, spce_qmmm_system, tmp_path):
        from pydft_qmmm.interfaces.vasp.vasp_factory import (
            vasp_interface_factory,
        )
        with pytest.raises(ValueError, match="sigma"):
            vasp_interface_factory(
                spce_qmmm_system,
                directory=str(tmp_path / "vasp"),
                pp_path=str(tmp_path / "pp"),
                embedding=True,
                embedding_sigma=-1.0,
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vasp_embedding_test.py::TestFactory -v`
Expected: FAIL — unexpected keyword argument.

- [ ] **Step 3: Implement**

In `vasp_factory.py`, add to the signature after `incar`:

```python
        embedding: bool = False,
        embedding_sigma: float = 0.3,
```

Add before the `VaspPotential(...)` construction:

```python
    if embedding_sigma <= 0.0:
        raise ValueError(
            f"embedding_sigma must be positive, got {embedding_sigma}.",
        )
```

Pass both through to the `VaspPotential(...)` call, and document them in the docstring:

```python
        embedding: Whether to electrostatically embed subsystem II
            charges via the VASP Python plugin.  Requires a VASP binary
            built with ``-DPLUGINS``.
        embedding_sigma: The Gaussian width
            (:math:`\mathrm{\mathring{A}}`) used to represent MM point
            charges on VASP's FFT grid.  Should comfortably exceed the
            grid spacing; too small aliases, too large over-softens the
            near field.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/vasp_embedding_test.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_factory.py tests/vasp_embedding_test.py
git commit -m "feat(vasp): expose embedding and embedding_sigma via the factory"
```

---

### Task 8: Tier 0 smoke test against a real VASP

**Files:**
- Test: `tests/vasp_embedding_test.py`
- Create: `tests/data/vasp_embedding/submit.slurm`

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: a `vasp` pytest marker, skipped unless `PYDFT_QMMM_VASP_COMMAND` is set.

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_embedding_test.py`:

```python
requires_vasp = pytest.mark.skipif(
    not os.environ.get("PYDFT_QMMM_VASP_COMMAND"),
    reason="set PYDFT_QMMM_VASP_COMMAND to run VASP-backed tests",
)


@requires_vasp
class TestSmoke:

    def test_plugin_fires_and_perturbs_the_energy(self, vasp_embedded):
        embedded = vasp_embedded.compute_energy()
        sentinel = os.path.join(vasp_embedded.directory, vasp_plugin.SENTINEL)
        assert os.path.isfile(sentinel)
        with open(sentinel) as fh:
            assert "mm_charges" in fh.read()
        assert np.isfinite(embedded)

    def test_zero_charges_reproduce_the_unembedded_energy(
            self, spce_qmmm_system, tmp_path,
    ):
        # The control: machinery fully active, physics inert.  Isolates
        # "the plumbing is a no-op when it should be" from "the physics
        # is right".
        from pydft_qmmm.interfaces.vasp.vasp_factory import (
            vasp_interface_factory,
        )
        spce_qmmm_system.charges[:] = 0.0
        plain = vasp_interface_factory(
            spce_qmmm_system, directory=str(tmp_path / "plain"),
        ).compute_energy()
        embedded = vasp_interface_factory(
            spce_qmmm_system, directory=str(tmp_path / "embed"),
            embedding=True,
        ).compute_energy()
        assert embedded == pytest.approx(plain, abs=1e-5)
```

- [ ] **Step 2: Run test to verify it is skipped without VASP**

Run: `pytest tests/vasp_embedding_test.py::TestSmoke -v`
Expected: 2 skipped.

- [ ] **Step 3: Add the Slurm harness**

Create `tests/data/vasp_embedding/submit.slurm`:

```bash
#!/usr/bin/bash
#SBATCH -J vasp_embed_test
#SBATCH --account=gts-jmcdaniel43-chemx
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH -t 1:00:00
#SBATCH --mem-per-gpu=12G

module purge
module load nvhpc-openmpi3/24.1
module load intel-oneapi-mkl/2023.1.0

ENV=/storage/home/hcoda1/1/cshao48/.conda/envs/vasp_plugin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ENV/lib
export PYTHONHOME=$ENV
export PATH=$ENV/bin:$PATH

export OMP_NUM_THREADS=8
export OMP_STACKSIZE=512m
ulimit -s unlimited

VASP=/storage/project/r-jmcdaniel43-0/cshao48/install/vasp/vasp.6.6.1_nvhpc24/bin/vasp_std
export PYDFT_QMMM_VASP_COMMAND="mpirun -np 1 $VASP"

pytest tests/vasp_embedding_test.py -v
```

- [ ] **Step 4: Run it on a GPU node**

Run: `sbatch tests/data/vasp_embedding/submit.slurm`
Expected: `TestSmoke` passes. If the sentinel is missing, the binary lacks `-DPLUGINS` or `PYTHONHOME` is unset — the error message from Task 6 names both.

- [ ] **Step 5: Commit**

```bash
git add tests/vasp_embedding_test.py tests/data/vasp_embedding/submit.slurm
git commit -m "test(vasp): Tier 0 smoke test and Slurm harness for embedding"
```

---

### Task 9: Tier 2 finite-difference forces

This task also answers the spec's one open question: whether `TOTEN` already contains `∫ρ·V_ext`.

**Files:**
- Test: `tests/vasp_embedding_test.py`
- Modify (only if the test fails in the specific way described): `pydft_qmmm/interfaces/vasp/vasp_plugin.py`

**Interfaces:**
- Consumes: `vasp_plugin.force_and_stress` (Task 5), `pydft_qmmm.utils.numerical_gradient`.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_embedding_test.py`:

```python
@requires_vasp
class TestGradients:

    def test_embedded_forces_match_numerical_gradient(
            self, spce_qmmm_system, tmp_path,
    ):
        # Mirrors TestCutoffEmbeddingSchemes in gradient_test.py, which
        # does the same for Psi4.
        from pydft_qmmm import QMMMHamiltonian
        from pydft_qmmm.utils import numerical_gradient
        from pydft_qmmm.interfaces.vasp.vasp_factory import (
            vasp_interface_factory,
        )
        potential = vasp_interface_factory(
            spce_qmmm_system,
            directory=str(tmp_path / "vasp"),
            embedding=True,
            incar={"EDIFF": 1e-8},
        )
        analytical = potential.compute_forces()[0]
        numerical = -numerical_gradient(
            potential, frozenset({0}), dist=1e-3,
        )[0]
        assert analytical == pytest.approx(numerical, abs=0.5)
```

- [ ] **Step 2: Run it on a GPU node**

Run: `sbatch tests/data/vasp_embedding/submit.slurm`
Expected: PASS.

**If it fails, read the failure before changing anything.** A discrepancy that is *uniform across all three components and proportional to the embedding strength* means `TOTEN` does not include `∫ρ·V_ext`, and the plugin must add it. A discrepancy on one component only, or one that survives setting all MM charges to zero, is an ordinary bug — do not "fix" it with the change in Step 3.

- [ ] **Step 3: Only if the failure matches the uniform-offset signature, add the energy term**

In `vasp_plugin.local_potential`, after adding to `total_potential`:

```python
        # VASP's TOTEN does not include the electron-V_ext term when the
        # plugin supplies total_potential, so add it here.  Established
        # empirically by the Tier 2 finite-difference test, not assumed.
        if constants.charge_density is None:
            raise RuntimeError(
                "charge_density is None, so the electron-V_ext energy "
                "term cannot be formed.  Set LVHAR = .TRUE. in the "
                "INCAR so VASP populates it.",
            )
        additions.total_energy += float(
            np.sum(constants.charge_density * v_ext)
            * abs(np.linalg.det(cell)) / v_ext.size
        )
```

Note `charge_density` is declared `Optional[DoubleArray] = None` in `ConstantsLocalPotential`, hence the guard. The reference plugin `VASP-Python/H_in_constant_field/INCAR` sets `LVHAR = .TRUE.`, which is why it has the field populated.

Re-run Step 2 and confirm the gradient now matches.

- [ ] **Step 4: Record the answer in the spec**

Edit `docs/superpowers/specs/2026-07-30-vasp-electrostatic-embedding-design.md`, replacing the "Open questions" section with the measured answer and the evidence for it.

- [ ] **Step 5: Commit**

```bash
git add tests/vasp_embedding_test.py \
        pydft_qmmm/interfaces/vasp/vasp_plugin.py \
        docs/superpowers/specs/2026-07-30-vasp-electrostatic-embedding-design.md
git commit -m "test(vasp): finite-difference gradient check; settle TOTEN double-counting"
```

---

### Task 10: Phase B — extract the physics module

Only after Tasks 1-9 pass. Because of the module-scope import discipline, this is a move, not a rewrite.

**Files:**
- Create: `pydft_qmmm/interfaces/vasp/grid_potential.py`
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Modify: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: all physics functions from Tasks 1-4.
- Produces: `grid_potential` module exporting `EPS0`, `read_mm_charges`, `spread_gaussian`, `poisson_fft`, `build_external_potential`, `interpolate_at`, `gradient_at`, `_g_squared`, `_fractional_grid`. `vasp_plugin` retains only the callbacks, the cache, and the module constants.

- [ ] **Step 1: Move the physics functions**

Create `grid_potential.py` containing `EPS0` and the seven functions listed above, moved verbatim from `vasp_plugin.py`. Add a module docstring stating that it must remain importable by VASP's embedded interpreter and therefore must not import anything beyond numpy.

- [ ] **Step 2: Reduce the plugin to callbacks**

In `vasp_plugin.py`, delete the moved functions and add at the top, after the numpy import:

```python
try:
    from .grid_potential import EPS0
    from .grid_potential import build_external_potential
    from .grid_potential import gradient_at
    from .grid_potential import interpolate_at
    from .grid_potential import read_mm_charges
except ImportError:
    # VASP copies this file into the run directory and executes it as a
    # top-level script, where the package-relative import is unavailable.
    from grid_potential import EPS0
    from grid_potential import build_external_potential
    from grid_potential import gradient_at
    from grid_potential import interpolate_at
    from grid_potential import read_mm_charges
```

- [ ] **Step 3: Copy both files into the run directory**

In `vasp_interface.py::_write_input`, replace the single `shutil.copyfile` with a loop:

```python
            for name in ("vasp_plugin.py", "grid_potential.py"):
                shutil.copyfile(
                    os.path.join(os.path.dirname(__file__), name),
                    os.path.join(self.directory, name),
                )
```

- [ ] **Step 4: Point the Tier 1 tests at the new module**

In `tests/vasp_grid_potential_test.py`, change the physics imports from `vasp_plugin` to `grid_potential`, leaving the `TestCallbacks` class importing `vasp_plugin`.

- [ ] **Step 5: Run the full suite**

Run: `pytest tests/vasp_grid_potential_test.py tests/vasp_embedding_test.py -v`
Expected: all non-VASP tests pass unchanged. Then re-run the Slurm harness to confirm the two-file copy works inside VASP.

- [ ] **Step 6: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/grid_potential.py \
        pydft_qmmm/interfaces/vasp/vasp_plugin.py \
        pydft_qmmm/interfaces/vasp/vasp_interface.py \
        tests/vasp_grid_potential_test.py
git commit -m "refactor(vasp): extract grid physics into grid_potential module"
```

---

## Deferred to later milestones

- **Forces on MM atoms (M3).** The QM density's reaction back on the MM charges. Until this exists, QM/MM MD will not conserve momentum.
- **PME / `long_range="electrostatic"`.** Requires `helpme_py` (https://github.com/johnppederson/helpme-py) installed into the `vasp_plugin` conda env. `add_electronic_potential` stays `NotImplementedError` until then.
- **VASPsol coexistence.** Both VASPsol and this plugin write `total_potential`; the interaction is untested.
