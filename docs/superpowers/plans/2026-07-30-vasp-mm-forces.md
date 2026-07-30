# VASP QM/MM Forces on MM Atoms Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the force the QM charge distribution exerts on each embedded MM point charge, so QM/MM MD conserves momentum.

**Architecture:** The plugin caches VASP's `hartree_potential + ion_potential` during `local_potential`, then in `force_and_stress` contracts that potential against each MM charge's own Gaussian derivative over a 6σ box — the transpose of `spread_gaussian`. Forces return to the driver through an `MM_FORCES` file, because VASP's `additions.forces` is sized for QM ions only.

**Tech Stack:** Python 3.10 (`vasp_qmmm` env), numpy, helpme_py 0.2.2, VASP 6.6.1 built with `-DPLUGINS` (nvhpc 24.1).

## Global Constraints

- **numpy only** in `grid_potential.py`. No scipy, nothing from `pydft_qmmm`, no VASP-only imports at module scope. This is what keeps the physics testable in milliseconds instead of 22-minute GPU jobs.
- **Run tests with** `~/.conda/envs/vasp_qmmm/bin/python -m pytest`. Python 3.10.20, matching the `libpython3.10` the VASP binary links.
- **Test files are named** `*_test.py`, not `test_*.py`.
- **`EPS0 = 0.005526349358057108`** e/(V·Å). `KJMOL_PER_EV = 96.48533212331`.
- **Sign convention:** VASP's potentials are electron-referenced. `V = −φ`. A positive charge *lowers* the electron potential energy. Getting this wrong flipped every nuclear force in M2.
- **The contraction must use the same σ and cutoff as the spreading.** Reading σ from `MM_CHARGES` rather than hardcoding is what makes Newton's third law hold term-by-term.
- **`PLUGINS/MODE` stays unset** (defaults to `serial`, giving the full gathered grid on rank 1).
- **Finite-difference tests must set `ISTART = 0`.** Restart reuse poisoned M2's gradient test and took three GPU jobs to diagnose.
- Spec: `docs/superpowers/specs/2026-07-30-vasp-mm-forces-design.md`

## File Structure

| File | Responsibility |
|---|---|
| `pydft_qmmm/interfaces/vasp/grid_potential.py` | **Modify.** Add `contract_gaussian_gradient` and `electrostatic_potential_from_vasp`. |
| `pydft_qmmm/interfaces/vasp/vasp_plugin.py` | **Modify.** Cache potentials in `local_potential`; compute and write `MM_FORCES` in `force_and_stress`. |
| `pydft_qmmm/interfaces/vasp/vasp_utils.py` | **Modify.** `read_mm_forces`. |
| `pydft_qmmm/interfaces/vasp/vasp_interface.py` | **Modify.** `compute_forces()` fills subsystem II rows. |
| `tests/vasp_grid_potential_test.py` | **Modify.** Tier 1: transpose identity, synthetic third law, σ consistency. |
| `tests/vasp_embedding_test.py` | **Modify.** Tier 2: third law, then MM finite difference. |

---

### Task 1: The Gaussian-gradient contraction

The core kernel. Everything else is plumbing.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/grid_potential.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `_axis_spacings(shape, cell)`, `_fractional_grid(shape)` — both already exist.
- Produces: `contract_gaussian_gradient(field, positions, charges, shape, cell, sigma, cutoff=6.0) -> NDArray` of shape `(N, 3)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/vasp_grid_potential_test.py`:

```python
class TestGaussianContraction:
    """The transpose of spread_gaussian: forces from a grid potential."""

    def test_recovers_the_gradient_of_a_linear_field(self):
        # For phi = c.r the convolution with a normalized Gaussian is
        # exact, so contracting must return exactly q*c per charge --
        # independent of sigma.
        gradient = np.array([0.3, -0.7, 0.2])
        axes = [np.arange(n) * 10.0 / n for n in SHAPE]
        mesh = np.meshgrid(*axes, indexing="ij")
        field = sum(g * m for g, m in zip(gradient, mesh))
        positions = np.array([[5.0, 5.0, 5.0], [3.0, 7.0, 4.0]])
        charges = np.array([1.0, -0.5])
        got = grid_potential.contract_gaussian_gradient(
            field, positions, charges, SHAPE, CELL, sigma=0.4,
        )
        assert got == pytest.approx(np.outer(charges, gradient), rel=1e-6)

    def test_scales_linearly_with_charge(self):
        field = np.random.RandomState(0).normal(size=SHAPE)
        position = np.array([[5.0, 5.0, 5.0]])
        one = grid_potential.contract_gaussian_gradient(
            field, position, np.array([1.0]), SHAPE, CELL, sigma=0.5,
        )
        two = grid_potential.contract_gaussian_gradient(
            field, position, np.array([2.0]), SHAPE, CELL, sigma=0.5,
        )
        assert two == pytest.approx(2.0 * one, rel=1e-12)

    def test_uniform_field_exerts_no_force(self):
        field = np.full(SHAPE, 3.7)
        got = grid_potential.contract_gaussian_gradient(
            field, np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            SHAPE, CELL, sigma=0.4,
        )
        assert got[0] == pytest.approx(np.zeros(3), abs=1e-9)

    def test_empty_selection_returns_an_empty_array(self):
        got = grid_potential.contract_gaussian_gradient(
            np.zeros(SHAPE), np.zeros((0, 3)), np.zeros(0),
            SHAPE, CELL, sigma=0.4,
        )
        assert got.shape == (0, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestGaussianContraction -q`
Expected: FAIL — `module 'grid_potential' has no attribute 'contract_gaussian_gradient'`

- [ ] **Step 3: Implement**

Append to `grid_potential.py`:

```python
def contract_gaussian_gradient(
        field, positions, charges, shape, cell, sigma, cutoff=6.0,
):
    """Force on each Gaussian charge sitting in a grid potential.

    This is the transpose of spread_gaussian.  That function builds a
    potential by summing q*g_sigma over a 6 sigma box; this one
    contracts a potential against grad g_sigma over the same box:

        F_j = -q_j * integral( field(r) * grad g_sigma(r - r_j) dr )

    Using the SAME kernel and box is what makes Newton's third law hold
    term by term rather than approximately: the force is differentiated
    with respect to exactly the representation the QM density felt.

    Evaluating instead the field's gradient at each charge position
    would be O(N_charges * N_grid) -- about 2.2 hours per ionic step for
    2685 charges on a 320**3 grid, versus seconds here.

    Args:
        field: A scalar potential on the grid, in volts.
        positions: An Nx3 array of charge positions (Angstrom).
        charges: An N array of charges (e).
        shape: The grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        sigma: The Gaussian width (Angstrom).  MUST match the value used
            to spread the charges.
        cutoff: Truncation radius in units of sigma.  MUST match too.

    Returns:
        An Nx3 array of forces (e*V/Angstrom == eV/Angstrom).
    """
    shape = tuple(int(n) for n in shape)
    forces = np.zeros((len(charges), 3), dtype=np.float64)
    if len(charges) == 0:
        return forces
    inverse = np.linalg.inv(cell)
    dimensions = np.array(shape)
    volume = abs(np.linalg.det(cell))
    d_volume = volume / float(np.prod(dimensions))
    prefactor = (2.0 * np.pi * sigma**2) ** -1.5
    half = np.ceil(cutoff * sigma / _axis_spacings(shape, cell)).astype(int)
    if np.any(2 * half + 1 >= dimensions):
        raise ValueError(
            "The cutoff box wraps the cell; contraction would "
            "double-count periodic images.  Use a larger cell or a "
            "smaller sigma.",
        )
    offsets = [np.arange(-h, h + 1) for h in half]
    for index, (position, charge) in enumerate(zip(positions, charges)):
        centre = np.round((position @ inverse) * dimensions).astype(int)
        select = [(centre[i] + offsets[i]) % dimensions[i] for i in range(3)]
        block = np.stack(
            np.meshgrid(
                *[(centre[i] + offsets[i]) / dimensions[i] for i in range(3)],
                indexing="ij",
            ),
            axis=-1,
        )
        delta = block - (position @ inverse)
        delta -= np.round(delta)
        cartesian = delta @ cell
        squared = np.einsum("...k,...k->...", cartesian, cartesian)
        gaussian = prefactor * np.exp(-0.5 * squared / sigma**2)
        # grad_r g(r - r_j) = -(r - r_j)/sigma**2 * g, and the derivative
        # with respect to r_j carries the opposite sign, so the two
        # minus signs leave a plain +(r - r_j)/sigma**2 here.
        weight = gaussian * field[np.ix_(*select)]
        forces[index] = charge * d_volume * np.einsum(
            "ijk,ijkc->c", weight, cartesian,
        ) / sigma**2
    return forces
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestGaussianContraction -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/grid_potential.py tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): Gaussian-gradient contraction, the transpose of spreading"
```

---

### Task 2: Newton's third law on a synthetic pair

Task 1 proves the kernel is a gradient. This proves it is the *right* gradient — that spreading and contracting are genuinely adjoint.

**Files:**
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `contract_gaussian_gradient` (Task 1), `spread_gaussian`, `poisson_fft`, `spectral_value_and_gradient` — all existing.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
class TestSyntheticThirdLaw:

    def test_two_gaussians_push_each_other_apart(self):
        # Charge A makes a potential; charge B sits in it.  Compute the
        # force on B by contraction, and the force on A by contracting
        # B's potential.  They must be equal and opposite.
        a_pos = np.array([[4.0, 5.0, 5.0]])
        b_pos = np.array([[6.0, 5.0, 5.0]])
        a_q = np.array([1.0])
        b_q = np.array([1.0])
        sigma = 0.5
        phi_a = poisson_of(a_pos, a_q, sigma)
        phi_b = poisson_of(b_pos, b_q, sigma)
        force_on_b = grid_potential.contract_gaussian_gradient(
            phi_a, b_pos, b_q, SHAPE, CELL, sigma,
        )[0]
        force_on_a = grid_potential.contract_gaussian_gradient(
            phi_b, a_pos, a_q, SHAPE, CELL, sigma,
        )[0]
        assert force_on_a == pytest.approx(-force_on_b, rel=1e-6)
        # Like charges: B (at larger x) is pushed to +x.
        assert force_on_b[0] > 0.0

    def test_mismatched_sigma_breaks_the_third_law(self):
        # The guard that matters: contraction and spreading must agree
        # on sigma.  If they drift apart the forces stop balancing, and
        # this test is what would catch it.
        a_pos = np.array([[4.0, 5.0, 5.0]])
        b_pos = np.array([[6.0, 5.0, 5.0]])
        charge = np.array([1.0])
        phi_a = poisson_of(a_pos, charge, 0.5)
        phi_b = poisson_of(b_pos, charge, 0.5)
        good = grid_potential.contract_gaussian_gradient(
            phi_a, b_pos, charge, SHAPE, CELL, 0.5,
        )[0]
        bad = grid_potential.contract_gaussian_gradient(
            phi_a, b_pos, charge, SHAPE, CELL, 0.9,
        )[0]
        reference = grid_potential.contract_gaussian_gradient(
            phi_b, a_pos, charge, SHAPE, CELL, 0.5,
        )[0]
        assert good == pytest.approx(-reference, rel=1e-6)
        assert not np.allclose(bad, -reference, rtol=1e-3)
```

Add this helper just above the class:

```python
def poisson_of(positions, charges, sigma):
    """Electrostatic potential (volts) of Gaussian charges on the grid."""
    rho = grid_potential.spread_gaussian(
        positions, charges, SHAPE, CELL, sigma,
    )
    return grid_potential.poisson_fft(rho, CELL)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestSyntheticThirdLaw -q`
Expected: FAIL — `NameError: name 'poisson_of' is not defined` until the helper is added, then a real assertion result.

- [ ] **Step 3: No implementation needed**

Task 1's kernel should already satisfy this. If `test_two_gaussians_push_each_other_apart` fails on the **sign**, the `cartesian` term in `contract_gaussian_gradient` has the wrong sign — flip it and re-run Task 1's tests, which do not pin the sign on their own because `test_recovers_the_gradient_of_a_linear_field` would also flip.

- [ ] **Step 4: Run the whole Tier 1 suite**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py -q -W "ignore::RuntimeWarning"`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/vasp_grid_potential_test.py
git commit -m "test(vasp): synthetic third law pins the contraction sign and sigma"
```

---

### Task 3: Convert VASP's potentials to an electrostatic potential

One function, one place, so the sign convention is stated once.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/grid_potential.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `electrostatic_potential_from_vasp(hartree, ion) -> NDArray` in volts.

- [ ] **Step 1: Write the failing test**

```python
class TestVaspPotentialConversion:

    def test_negates_the_electron_referenced_sum(self):
        hartree = np.full((4, 4, 4), -2.0)
        ion = np.full((4, 4, 4), -3.0)
        got = grid_potential.electrostatic_potential_from_vasp(hartree, ion)
        # VASP reports what an ELECTRON feels, in eV.  The electrostatic
        # potential is the negative of that.
        assert got == pytest.approx(np.full((4, 4, 4), 5.0))

    def test_rejects_missing_fields(self):
        with pytest.raises(RuntimeError, match="PLUGINS/LOCAL_POTENTIAL"):
            grid_potential.electrostatic_potential_from_vasp(
                None, np.zeros((4, 4, 4)),
            )

    def test_rejects_shape_mismatch(self):
        with pytest.raises(RuntimeError, match="shape"):
            grid_potential.electrostatic_potential_from_vasp(
                np.zeros((4, 4, 4)), np.zeros((8, 8, 8)),
            )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestVaspPotentialConversion -q`
Expected: FAIL — no attribute `electrostatic_potential_from_vasp`.

- [ ] **Step 3: Implement**

```python
def electrostatic_potential_from_vasp(hartree, ion):
    """Electrostatic potential of the QM system, in volts.

    VASP hands the plugin the Hartree potential of the electrons and the
    local potential of the ion cores, both ELECTRON-REFERENCED and in
    eV: they are what an electron feels.  The electrostatic potential a
    positive test charge feels is the negative of their sum.

    That single sign is the same convention as V_ext = -phi elsewhere in
    this package, and getting it backwards is what flipped every nuclear
    force in Milestone 2.

    Both fields are allocated by VASP whenever
    PLUGINS/LOCAL_POTENTIAL = T, so a None here means the callback was
    reached some other way rather than that an INCAR tag is missing.

    Args:
        hartree: constants.hartree_potential, or None.
        ion: constants.ion_potential, or None.

    Returns:
        The electrostatic potential on the grid (volts).
    """
    if hartree is None or ion is None:
        missing = "hartree_potential" if hartree is None else "ion_potential"
        raise RuntimeError(
            f"{missing} is None.  VASP allocates both whenever "
            "PLUGINS/LOCAL_POTENTIAL = T, so this means the potentials "
            "were never captured -- check that local_potential ran "
            "before force_and_stress.",
        )
    hartree = np.asarray(hartree, dtype=np.float64)
    ion = np.asarray(ion, dtype=np.float64)
    if hartree.shape != ion.shape:
        raise RuntimeError(
            f"hartree_potential has shape {hartree.shape} but "
            f"ion_potential has {ion.shape}.",
        )
    return -(hartree + ion)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestVaspPotentialConversion -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/grid_potential.py tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): convert VASP's electron-referenced potentials to phi"
```

---

### Task 4: Cache the potentials and write MM_FORCES

Wires the kernel into the plugin.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Modify: `pydft_qmmm/interfaces/vasp/vasp_utils.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `contract_gaussian_gradient` (Task 1), `electrostatic_potential_from_vasp` (Task 3), `read_mm_charges` (existing).
- Produces: `vasp_plugin.FORCE_FILE = "MM_FORCES"`; `vasp_utils.read_mm_forces(path, expect_step=None) -> tuple[NDArray, int]` returning `(forces Nx3 in kJ/mol/Å, step)`.

- [ ] **Step 1: Write the failing test**

```python
class TestMMForceHandoff:

    def test_force_and_stress_writes_mm_forces(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[7.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.5,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL)
        # A positive Gaussian at the origin side: its potential pushes a
        # positive MM charge at +x further along +x.
        phi = poisson_of(np.array([[5.0, 5.0, 5.0]]), np.array([1.0]), 0.5)
        constants.hartree_potential = -phi
        constants.ion_potential = np.zeros(SHAPE)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        vasp_plugin.force_and_stress(constants, additions)
        forces, step = vasp_utils.read_mm_forces("MM_FORCES")
        assert step == 0
        assert forces.shape == (1, 3)
        assert forces[0, 0] > 0.0
        assert forces[0, 1] == pytest.approx(0.0, abs=1e-9)

    def test_missing_potentials_raise(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        vasp_utils.write_mm_charges(
            "MM_CHARGES", np.array([[7.0, 5.0, 5.0]]), np.array([1.0]),
            step=0, sigma=0.5,
        )
        vasp_plugin.reset_cache()
        constants = _constants(SHAPE, CELL)
        additions = _additions(SHAPE)
        vasp_plugin.local_potential(constants, additions)
        with pytest.raises(Exception):
            vasp_plugin.force_and_stress(constants, additions)
        assert (tmp_path / vasp_plugin.ERROR_FILE).exists()
```

Extend the `_constants` helper so the mock carries the two fields:

```python
        charge_density=np.full(shape, 8.0),
        hartree_potential=None,
        ion_potential=None,
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestMMForceHandoff -q`
Expected: FAIL — `read_mm_forces` does not exist.

- [ ] **Step 3: Implement the reader**

The **writer stays inline in the plugin**, not in `vasp_utils`. The plugin is
copied standalone into each run directory and imported as a top-level module by
VASP's interpreter, so it cannot import `vasp_utils` — only `grid_potential` and
`pme_external` travel with it. The reader lives in `vasp_utils` because only the
driver uses it.

Append to `vasp_utils.py`, and add `"read_mm_forces"` to `__all__`:

```python
def read_mm_forces(path: str, expect_step: int | None = None):
    r"""Read the MM forces the plugin computed.

    Args:
        path: The file to read.
        expect_step: If given, the step stamp the file must carry.

    Returns:
        An Nx3 array of forces
        (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) and the step.
    """
    with open(path) as fh:
        count, step = fh.readline().split()
        count, step = int(count), int(step)
        if count == 0:
            return np.zeros((0, 3)), step
        data = np.loadtxt(fh, dtype=np.float64, ndmin=2)
    if len(data) != count:
        raise ValueError(
            f"{path} declares {count} forces but holds {len(data)}.",
        )
    if expect_step is not None and step != expect_step:
        raise ValueError(
            f"{path} is at step {step}, expected {expect_step}.",
        )
    return data.copy(), step
```

- [ ] **Step 4: Implement the plugin side**

In `vasp_plugin.py`, add `FORCE_FILE = "MM_FORCES"` beside the other file names, and in `local_potential` cache the potentials right after the `try:`:

```python
        if getattr(constants, "hartree_potential", None) is not None:
            _CACHE["hartree"] = np.array(constants.hartree_potential)
            _CACHE["ion"] = np.array(constants.ion_potential)
```

Note the explicit copies: VASP marks these arrays read-only and may reuse the buffer between calls, so holding a reference is not safe.

Then append to `force_and_stress`, just before the `except` clause:

```python
        # The QM->MM back-reaction.  VASP's additions.forces is sized
        # 3 x number_ions, i.e. QM ions only -- the MM atoms do not
        # exist in VASP's calculation -- so these go back to the driver
        # through a file, the way the charges came in.
        phi_qm = electrostatic_potential_from_vasp(
            _CACHE.get("hartree"), _CACHE.get("ion"),
        )
        mm_positions, mm_charges, step, sigma = read_mm_charges(CHARGE_FILE)
        mm_forces = contract_gaussian_gradient(
            phi_qm, mm_positions, mm_charges, shape, cell, sigma,
        ) * KJMOL_PER_EV
        with open(FORCE_FILE, "w") as fh:
            fh.write(f"{len(mm_forces)} {step}\n")
            for fx, fy, fz in mm_forces:
                fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
```

`force_and_stress` needs `shape` and `cell` in scope; it already computes `cell`, so add `shape = tuple(int(n) for n in constants.shape_grid)` beside it. Add `KJMOL_PER_EV = 96.48533212331` next to `EPS0` in `grid_potential.py` and import it in the plugin alongside the other names.

- [ ] **Step 5: Run test to verify it passes**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py -q -W "ignore::RuntimeWarning"`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_plugin.py \
        pydft_qmmm/interfaces/vasp/vasp_utils.py \
        pydft_qmmm/interfaces/vasp/grid_potential.py \
        tests/vasp_grid_potential_test.py
git commit -m "feat(vasp): plugin computes MM forces and writes MM_FORCES"
```

---

### Task 5: Return the forces through the interface

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/vasp_interface.py`
- Test: `tests/vasp_embedding_test.py`

**Interfaces:**
- Consumes: `vasp_utils.read_mm_forces` (Task 4).
- Produces: `VaspInterface._read_mm_forces() -> NDArray` of shape `(n_subsystem_II, 3)`.

- [ ] **Step 1: Write the failing test**

```python
class TestMMForceReturn:

    def test_reads_forces_into_subsystem_ii_rows(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        rows = np.arange(3 * len(indices), dtype=float).reshape(-1, 3)
        with open(
            os.path.join(vasp_embedded.directory, "MM_FORCES"), "w",
        ) as fh:
            fh.write(f"{len(rows)} 0\n")
            for fx, fy, fz in rows:
                fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
        got = vasp_embedded._read_mm_forces()
        assert got == pytest.approx(rows)

    def test_missing_file_raises(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        with pytest.raises(Exception, match="MM_FORCES"):
            vasp_embedded._read_mm_forces()

    def test_all_zero_forces_raise(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        with open(
            os.path.join(vasp_embedded.directory, "MM_FORCES"), "w",
        ) as fh:
            fh.write(f"{len(indices)} 0\n")
            for _ in indices:
                fh.write("0.0 0.0 0.0\n")
        # Identically zero is indistinguishable from Milestone 2's
        # behaviour, which is exactly the failure this milestone exists
        # to remove.  It must not pass silently.
        with pytest.raises(Exception, match="zero"):
            vasp_embedded._read_mm_forces()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_embedding_test.py::TestMMForceReturn -q`
Expected: FAIL — no attribute `_read_mm_forces`.

- [ ] **Step 3: Implement**

Add to `VaspInterface`:

```python
    def _read_mm_forces(self) -> NDArray[np.float64]:
        r"""Read the QM->MM forces the plugin wrote.

        Returns:
            An Nx3 array of forces
            (:math:`\mathrm{kJ\;mol^{-1}\;\mathring{A}^{-1}}`) ordered to
            match ``sorted(system.select("subsystem II"))``.

        Raises:
            VaspExecutionError: If the file is absent or the forces are
                identically zero, either of which means the back-reaction
                was never applied.
        """
        path = os.path.join(self.directory, "MM_FORCES")
        if not os.path.isfile(path):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "MM_FORCES is absent, so the QM->MM back-reaction was "
                "never computed and momentum will not be conserved.",
            )
        forces, _ = vasp_utils.read_mm_forces(path)
        indices = sorted(self.system.select("subsystem II"))
        if len(forces) != len(indices):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                f"MM_FORCES holds {len(forces)} rows but subsystem II "
                f"has {len(indices)} atoms.",
            )
        if len(forces) and not np.any(forces):
            raise vasp_utils.VaspExecutionError(
                self.directory,
                "MM_FORCES is identically zero, which is what the "
                "unimplemented back-reaction looked like.",
            )
        return forces
```

And in `VaspPotential.compute_forces`, after the QM rows are filled:

```python
        if self.embedding:
            embed_indices = sorted(self.system.select("subsystem II"))
            forces[embed_indices, :] = self._read_mm_forces()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_embedding_test.py -q`
Expected: all pass, VASP-backed ones skipped.

- [ ] **Step 5: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/vasp_interface.py tests/vasp_embedding_test.py
git commit -m "feat(vasp): return MM forces into subsystem II rows"
```

---

### Task 6: Newton's third law against real VASP

The acceptance test, and the cheap one.

**Files:**
- Test: `tests/vasp_embedding_test.py`
- Create: `tests/data/vasp_embedding/thirdlaw.slurm`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
@requires_vasp
class TestThirdLaw:
    """Momentum conservation: the reason Milestone 3 exists."""

    def test_qm_and_mm_forces_cancel(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            incar={"ENCUT": 250, "EDIFF": 1e-7},
        )
        forces = potential.compute_forces()
        qm = sorted(vasp_qmmm_system.select("subsystem I"))
        mm = sorted(vasp_qmmm_system.select("subsystem II"))
        # Only over I + II.  Subsystem III feels nothing through the
        # near-field channel, so including it would show a violation
        # that is expected rather than a bug.
        total = forces[qm].sum(axis=0) + forces[mm].sum(axis=0)
        scale = np.abs(forces[qm + mm]).max()
        print(f"\n  sum |F| over I+II : {total}")
        print(f"  largest |F|       : {scale:.3f} kJ/mol/A")
        print(f"  relative violation: {np.abs(total).max() / scale:.3e}")
        assert np.abs(total).max() < 0.01 * scale
```

- [ ] **Step 2: Verify it skips without VASP**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_embedding_test.py::TestThirdLaw -q`
Expected: 1 skipped.

- [ ] **Step 3: Write the job script**

```bash
cd /storage/project/r-jmcdaniel43-0/cshao48/install/pydft-vasp-qmmm/pydft-qmmm
sed -e 's/#SBATCH -J vasp_fig2b/#SBATCH -J vasp_thirdlaw/' \
    -e 's|vasp_embedding/fig2b.slurm|vasp_embedding/thirdlaw.slurm|' \
    -e 's/-k "Figure2b"/-k "TestThirdLaw"/' \
    -e 's/#SBATCH --mem-per-gpu=12G/#SBATCH --mem-per-gpu=64G/' \
    tests/data/vasp_embedding/fig2b.slurm > tests/data/vasp_embedding/thirdlaw.slurm
bash -n tests/data/vasp_embedding/thirdlaw.slurm
```

- [ ] **Step 4: Run it**

Run: `sbatch tests/data/vasp_embedding/thirdlaw.slurm`
Expected: PASS. A violation of a few percent rather than <1% suggests the contraction and spreading disagree on σ or cutoff — check that `sigma` is read from `MM_CHARGES` in both, not hardcoded.

- [ ] **Step 5: Commit**

```bash
git add tests/vasp_embedding_test.py tests/data/vasp_embedding/thirdlaw.slurm
git commit -m "test(vasp): Newton's third law over subsystems I and II"
```

---

### Task 7: Finite difference on an MM atom

The strong per-atom test. Run only after Task 6 passes.

**Files:**
- Test: `tests/vasp_embedding_test.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing.

- [ ] **Step 1: Write the failing test**

```python
@requires_vasp
class TestMMFiniteDifference:

    def test_mm_force_matches_the_numerical_gradient(
            self, vasp_qmmm_system, vasp_workdir, vasp_pp_library,
    ):
        from pydft_qmmm.calculators import PotentialCalculator
        from pydft_qmmm.utils import numerical_gradient
        mm = sorted(vasp_qmmm_system.select("subsystem II"))
        target = mm[0]
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(vasp_workdir / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            # ISTART=0 on every evaluation: restart reuse across
            # displacements poisoned the Milestone 2 gradient test and
            # took three GPU jobs to diagnose.
            incar={"ENCUT": 250, "EDIFF": 1e-7, "ISTART": 0},
        )
        calculator = PotentialCalculator(vasp_qmmm_system, potential)
        analytical = calculator.calculate().forces[target]
        numerical = -numerical_gradient(
            calculator, frozenset({target}), dist=1e-3,
        )[0]
        print(f"\n  analytical: {analytical}")
        print(f"  numerical : {numerical}")
        print(f"  residual  : {analytical - numerical}")
        assert analytical == pytest.approx(numerical, abs=1.0)
```

- [ ] **Step 2: Verify it skips without VASP**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_embedding_test.py::TestMMFiniteDifference -q`
Expected: 1 skipped.

- [ ] **Step 3: Run it on a GPU node**

```bash
sed -e 's/-k "TestThirdLaw"/-k "TestMMFiniteDifference"/' \
    -e 's/#SBATCH -t 1:00:00/#SBATCH -t 4:00:00/' \
    tests/data/vasp_embedding/thirdlaw.slurm \
    > tests/data/vasp_embedding/mmgradient.slurm
sbatch tests/data/vasp_embedding/mmgradient.slurm
```

Expected: PASS. Seven VASP launches.

**If it fails, do not tune the tolerance.** Compare the residual against the unembedded control from Milestone 2, which agreed to 0.03 kJ/mol/Å: a residual of that order is the interface's own noise floor, while one comparable to the force itself is a real defect in the contraction.

- [ ] **Step 4: Commit**

```bash
git add tests/vasp_embedding_test.py tests/data/vasp_embedding/mmgradient.slurm
git commit -m "test(vasp): finite-difference check on an MM atom"
```

---

### Task 8: PME-region forces

Subsystem III acts on the QM through helPME's reciprocal sum, so its reaction must return the same way. Do this only after Tasks 1-7 pass, so a failure here is isolated to the new channel.

**Files:**
- Modify: `pydft_qmmm/interfaces/vasp/pme_external.py`
- Modify: `pydft_qmmm/interfaces/vasp/vasp_plugin.py`
- Test: `tests/vasp_grid_potential_test.py`

**Interfaces:**
- Consumes: `read_pme_data`, `grid_coordinates` (existing in `pme_external`).
- Produces: `pme_external.pme_forces_on_charges(path, charge_density, shape, cell) -> NDArray` of shape `(N_all, 3)` in eV/Å.

- [ ] **Step 1: Write the failing test**

```python
class TestPMEForces:

    def test_returns_one_row_per_charge(self, tmp_path):
        from pydft_qmmm.interfaces.vasp import pme_external
        rng = np.random.RandomState(0)
        length = 20.0
        cell = np.diag([length] * 3)
        shape = (32, 32, 32)
        positions = rng.uniform(0.0, length, (40, 3))
        charges = rng.uniform(-1.0, 1.0, 40)
        charges -= charges.mean()
        path = str(tmp_path / "PME_DATA")
        vasp_utils.write_pme_data(
            path, positions, charges, [], 0.35, (32, 32, 32), 6, 0,
        )
        density = np.zeros(shape)
        density[16, 16, 16] = 1.0
        got = pme_external.pme_forces_on_charges(path, density, shape, cell)
        assert got.shape == (40, 3)
        assert np.isfinite(got).all()

    def test_zero_density_gives_zero_force(self, tmp_path):
        from pydft_qmmm.interfaces.vasp import pme_external
        cell = np.diag([20.0] * 3)
        shape = (32, 32, 32)
        path = str(tmp_path / "PME_DATA")
        vasp_utils.write_pme_data(
            path, np.array([[5.0, 5.0, 5.0]]), np.array([1.0]),
            [], 0.35, (32, 32, 32), 6, 0,
        )
        got = pme_external.pme_forces_on_charges(
            path, np.zeros(shape), shape, cell,
        )
        assert got == pytest.approx(np.zeros((1, 3)), abs=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestPMEForces -q`
Expected: FAIL — no attribute `pme_forces_on_charges`.

- [ ] **Step 3: Implement**

Append to `pme_external.py`:

```python
def pme_forces_on_charges(path, charge_density, shape, cell, chunk=1 << 20):
    """Force on every MM charge from the QM density, via PME.

    The mirror of build_pme_potential.  There the MM charges are the
    source and VASP's grid points are the probes; here VASP's grid
    carries the source density and the MM charges are the probes.

    Args:
        path: The PME_DATA file written by the interface.
        charge_density: VASP's grid density, normalized so that
            sum(rho) = NELECT * N_grid.
        shape: VASP's FFT grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        chunk: Grid points per helPME call.

    Returns:
        An Nx3 array of forces (eV/Angstrom), one row per charge in
        PME_DATA.
    """
    import helpme_py

    positions, charges, _, alpha, gridnumber, spline_order, _ = (
        read_pme_data(path)
    )
    pme = helpme_py.PMEInstanceD()
    pme.setup(1, alpha, spline_order, *gridnumber, COULOMB_CONSTANT, 0)
    pme.set_lattice_vectors(
        *_lattice_constants(np.asarray(cell, dtype=np.float64)),
        helpme_py.LatticeType.XAligned,
    )
    # The grid density in e/Angstrom**3 * cell volume, i.e. the charge
    # carried by each grid point, with the electron sign.
    density = -np.asarray(charge_density, dtype=np.float64).reshape(-1)
    density = density / float(np.prod(shape))
    coordinates = grid_coordinates(shape, cell)
    result = np.zeros((len(charges), 4))
    matrix = helpme_py.MatrixD(result)
    probe_charges = helpme_py.MatrixD(charges.reshape(-1, 1))
    probe_positions = helpme_py.MatrixD(positions)
    for start in range(0, len(coordinates), chunk):
        block = np.ascontiguousarray(coordinates[start:start + chunk])
        block_charges = np.ascontiguousarray(
            density[start:start + chunk].reshape(-1, 1),
        )
        pme.compute_P_rec(
            0, helpme_py.MatrixD(block_charges), helpme_py.MatrixD(block),
            probe_positions, 1, matrix,
        )
    # column 0 is the potential; columns 1-3 its gradient.  F = q * E,
    # and E = -grad phi.
    return -(result[:, 1:].T * charges).T / KJMOL_PER_EV
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/.conda/envs/vasp_qmmm/bin/python -m pytest tests/vasp_grid_potential_test.py::TestPMEForces -q`
Expected: 2 passed.

- [ ] **Step 5: Add the PME rows to the plugin's force output**

In `force_and_stress`, when `PME_FILE` exists, add the reciprocal contribution to the rows for subsystem III. The near-field rows already come from the contraction; PME supplies the rest.

```python
        if os.path.isfile(PME_FILE):
            try:
                from .pme_external import pme_forces_on_charges
            except ImportError:
                from pme_external import pme_forces_on_charges
            pme_rows = pme_forces_on_charges(
                PME_FILE, constants.charge_density, shape, cell,
            ) * KJMOL_PER_EV
            with open(PME_FORCE_FILE, "w") as fh:
                fh.write(f"{len(pme_rows)} {step}\n")
                for fx, fy, fz in pme_rows:
                    fh.write(f"{fx:.12e} {fy:.12e} {fz:.12e}\n")
```

Add `PME_FORCE_FILE = "PME_FORCES"` beside the other file names, and have the interface read it into the subsystem III rows the same way `_read_mm_forces` fills subsystem II.

- [ ] **Step 6: Extend the third-law test to all three subsystems**

Change `TestThirdLaw` to sum over `I + II + III` when a PME potential is attached, and keep the `I + II` form when it is not. With both channels present the full sum must cancel.

- [ ] **Step 7: Commit**

```bash
git add pydft_qmmm/interfaces/vasp/pme_external.py \
        pydft_qmmm/interfaces/vasp/vasp_plugin.py \
        pydft_qmmm/interfaces/vasp/vasp_interface.py \
        tests/vasp_grid_potential_test.py tests/vasp_embedding_test.py
git commit -m "feat(vasp): PME-region forces on subsystem III"
```

---

## Deferred

- **Stress contributions from the MM charges.** The plugin can add to `additions.stress`, but nothing downstream consumes it.
- **The ~16 kJ/mol/Å QM/MM gradient residual from Milestone 2.** Measured, unexplained, and out of scope here.
