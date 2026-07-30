# VASP Electrostatic Embedding (Milestone 2) — Design

**Date:** 2026-07-30
**Status:** Approved for planning
**Scope:** Cutoff-scheme electrostatic embedding of MM point charges into VASP's
Kohn-Sham Hamiltonian via the VASP Python plugin interface.

## Goal

Let MM subsystem II charges enter the VASP SCF as an external potential
`V_ext`, so that QM/MM with VASP as the QM engine uses electrostatic rather than
mechanical embedding. This is §3.4 of the Todorova manuscript
(arXiv:2510.23328).

## Scope

**In scope**

- Energy and QM forces correct under electrostatic embedding.
- The `close_range="electrostatic"` (cutoff) scheme.
- The Eq. 3-4 corrections VASP omits: `ΔE_I = −Z_I·V_ext(R_I)` and
  `ΔF_I = +Z_I·∇V_ext(R_I)`.

  **Correction, 2026-07-30 (found during implementation).** This document
  originally wrote the force as `ΔF_I = −Z_I·∇V_ext(R_I)`, matching the
  manuscript's Eq. 4 as transcribed. That is **inconsistent with Eq. 3** and
  flips every nuclear force. The force must be the negative gradient of the
  energy it accompanies:

  `F = −∇(ΔE_I) = −∇(−Z_I·V_ext) = +Z_I·∇V_ext`

  Verified physically: for an MM charge of +1 e and a pseudo-ion of +11 e two
  Ångström away, `V_ext` is negative and rising with r, so `∇V_ext > 0` and the
  ion is pushed away — repulsion, as two positive charges must. With the
  original sign the ion was attracted, at −38.4 eV/Å. Caught by the Tier 1
  analytic tests rather than waiting for the Tier 2 finite-difference run.

  **`Z_I` is the pseudopotential valence charge `ZVAL`, not the atomic number.**
  VASP's nuclei are pseudo-ions carrying only valence charge, so using the
  atomic number would over-count by the core electrons — for Au that is 79 vs
  11, a factor of seven. Obtain it as `constants.ZVAL[constants.ion_types[i]]`.

**Out of scope**

- **Forces on MM atoms** (the QM density's reaction back on the MM charges).
  That is Milestone 3. Without it a QM/MM MD run will not conserve momentum;
  this is a deliberate staging choice, not an oversight.
- **PME / `long_range="electrostatic"`**. Deferred to phase 2. Requires
  `helpme_py` (https://github.com/johnppederson/helpme-py), which is not
  currently installed.
- Cross-code validation against Psi4, and reproduction of published Au/H₂O
  results.

## Established facts this design rests on

Each was verified against the code or a real run, not assumed.

- `add_electronic_potential` is called from exactly one place,
  `hamiltonians/qmmm_hamiltonian.py:216`, inside the **PME branch only**. It is
  therefore *not* the hook for cutoff embedding, and remains
  `NotImplementedError` in this milestone.
- `PMEElectronicPotential.compute_potential(coordinates)`
  (`potentials/pme_potential.py:111`) evaluates at **arbitrary** coordinates;
  helPME performs B-spline interpolation from its own PME grid internally. No
  hand-written PME→DFT-grid interpolation is ever required.
- VASP maintains two FFT grids. `NGX/NGY/NGZ` (coarse) carries wavefunctions;
  `NGXF/NGYF/NGZF` (fine, "support grid") carries the charge density and local
  potentials. For the Au(111) reference system: coarse `72×80×168`, fine
  `144×160×336`, `PREC = normal`.
- The plugin's `constants.shape_grid` is the **fine** grid — confirmed from a
  real run's sentinel: `shape_grid = (144, 160, 336)` = 7,741,440 points.
- `vasp_interface.py:111` builds the VASP cell from `self.system.box`, the whole
  system box. **QM and MM share one periodic cell**, so MM charges lie inside
  the VASP cell and "cutoff" is a selection criterion, not a boundary problem.
- `strings vasp_std | grep "not compiled with PLUGINS"` is **not** a valid
  plugin check — that string is present in all builds. Use the runtime sentinel
  instead.

## Architecture

The organizing insight: **only the electron–MM coupling needs the grid.**
Electrons live on the FFT grid, so their coupling to MM charges must pass
through `total_potential`. The nucleus–MM term does not — but it must be
evaluated from the *same* `V_ext` field, or the two halves of the QM subsystem
feel inconsistent potentials.

**PyDFT-QMMM process** (`vasp_interface.py`), per evaluation:

1. Select `subsystem II`; write `MM_CHARGES` into the run directory.
2. Copy `vasp_plugin.py` into the run directory; add
   `PLUGINS/LOCAL_POTENTIAL = T` and `PLUGINS/FORCE_AND_STRESS = T` to the INCAR.
3. Launch VASP; parse `vasprun.xml` as today.

**VASP process** (`vasp_plugin.py`):

4. On the **first** `local_potential` call: read `MM_CHARGES`, spread each charge
   as a Gaussian onto `constants.shape_grid`, FFT-Poisson solve, negate to
   electron potential energy in eV, cache in a module global. MM positions are
   fixed across the SCF, so this runs once per VASP launch, not once per SCF
   step.
5. On **every** `local_potential` call: `additions.total_potential += V_ext`.
6. In `force_and_stress`: evaluate `V_ext` and `∇V_ext` at each QM nucleus by
   interpolating the same grid; apply `ΔE_I` and `ΔF_I`.

Because the corrections are applied inside VASP, the energy and forces in
`vasprun.xml` are already correct and `_run()` needs no modification.

### Why interpolate `V_ext` at the nuclei rather than sum pairwise

A direct `Σ_I Σ_j Z_I q_j / r_Ij` sum is exact and cheap, and is the obvious
alternative. It is **rejected** because the grid solve is periodic (FFT) while a
pairwise sum is not: electrons would feel a periodic field and nuclei a
non-periodic one. Interpolating the same grid keeps them consistent by
construction, and is what `ΔE_I = −Z_I·V_ext(R_I)` literally specifies.

## Components

### Physics functions — pure numpy, no VASP imports

The unit under test. In phase C these are module-level functions in
`vasp_plugin.py`; in phase B they move unchanged to
`interfaces/vasp/grid_potential.py`.

| function | responsibility |
|---|---|
| `read_mm_charges(path)` | → positions (N×3, Å), charges (N, e), step stamp |
| `spread_gaussian(pos, q, shape, cell, sigma)` | → ρ on grid (e/Å³), periodic minimum-image |
| `poisson_fft(rho, cell)` | → φ (V), solving `∇²φ = −ρ/ε₀` in reciprocal space |
| `interpolate_at(v, cell, points)` | → `V_ext` at arbitrary points |
| `gradient_at(v, cell, points)` | → `∇V_ext` at points (reciprocal-space derivative, then interpolate) |

The callbacks are thin wrappers containing no physics.

**Hard constraint:** no VASP-only imports at module scope, and **numpy only** —
no scipy. This is what lets the Tier 1 tests import the module directly with no
VASP in the loop, and what makes the phase C → B move a file cut. Every extra
import is a way for the embedded interpreter to fail at runtime inside VASP.

### Interface-side changes

- `vasp_interface.py`: add `_write_mm_charges()`; `_write_input()` adds the two
  `PLUGINS/` tags and copies the plugin file when embedding is active.
- `vasp_factory.py`: add `embedding_sigma` and a plugin-source path.

### Handoff file

Plain text: one header line carrying the count and step stamp, then `x y z q` at
`%.12e`. Binary `.npy` would be marginally more exact; for a spike, being able
to eyeball the file when a sign looks wrong is worth more than the last few ULPs.

## Physics conventions

These are the highest-risk part of the implementation.

- **Units chain:** ρ in e/Å³ → `poisson_fft` with
  `eps0 = 0.005526349358057108` e/(V·Å) → φ in volts → **negate** → `V_ext` as
  electron potential *energy* in eV, which is what `total_potential` expects.
- **Sign:** a *positive* MM charge must *lower* the electron potential energy.
- **G = 0 is set to zero.** The reciprocal-space solve diverges at G = 0 for a
  net-charged cell. Setting `φ(G=0) = 0` fixes the average potential at zero,
  equivalent to a uniform neutralizing background — the same convention VASP
  uses for charged cells. Consequence, to be documented rather than fixed: a
  net-charged subsystem II shifts absolute energies by a constant. Energy
  differences at fixed composition are unaffected, which is what MD needs.
- **No self-energy subtraction.** Each Gaussian does interact with itself in the
  FFT solve, but `V_ext` is only ever evaluated at *QM* positions; MM–MM
  interactions belong to OpenMM and never enter through this path. The one real
  approximation is that QM electrons within ~σ of an MM charge see a smeared
  rather than a point charge.
- **σ** is user-facing, defaulting to ≈2× the grid spacing (σ ≈ 0.3 Å for the
  reference system, comfortably resolved on a 144×160×336 grid). Too small
  aliases; too large over-softens the near field.

## Error handling

**Governing principle: never silently apply a zero potential.** Every failure
mode below has a benign-looking variant in which VASP completes, `vasprun.xml`
parses, and the energy is quietly the unembedded one. That produces plausible
wrong science, which is worse than a crash. All guards fail loudly.

| failure | guard |
|---|---|
| VASP not built with `-DPLUGINS`, or plugin never invoked | Plugin writes a sentinel on first call; interface checks for it after every embedded run and raises `VaspExecutionError` if absent |
| `MM_CHARGES` missing or unreadable | Plugin writes `PLUGIN_ERROR.txt` then raises — a Python exception inside VASP's embedded interpreter may not surface legibly in VASP's own output |
| Stale `MM_CHARGES` from a previous step | Interface stamps a step counter and deletes before rewriting; plugin verifies the stamp |
| Grid changed between steps (NPT, INCAR edit) | Plugin cache keyed on `shape_grid`; mismatch rebuilds |
| `subsystem II` empty | Explicit no-op with `V_ext = 0`, not a crash |
| Net-charged `subsystem II` | Warn once, documenting the G=0 constant offset |
| `PYTHONHOME`/`PATH` unset for the plugin env | Interface validates before launching |

## Testing

**Tier 0 — smoke.** Sentinel present, `V_ext` non-zero, run completes.

**Tier 1 — analytic, no VASP.** The first gate; runs in milliseconds.

- **Point charge:** single MM charge, large cell, evaluate at r ≫ σ against
  `−q/(4πε₀r)`.
- **Sign:** positive MM charge lowers electron potential energy. Asserted on its
  own — it is the likeliest bug and is invisible to any magnitude-only check.
- **Units:** assert on intermediate φ in volts, not only the final eV array, so
  a unit error cannot cancel against a sign error.
- **σ convergence:** sweep σ; approach the point-charge limit at fixed large r.
- **Neutral dipole:** two opposite charges; also confirms G=0 is irrelevant when
  neutral.
- **Charge conservation:** `Σρ·dV == Σq`.
- **Gradient consistency:** `gradient_at` vs finite differences of
  `interpolate_at`.

**Tier 2 — finite-difference forces, VASP in the loop.** Small QM system plus a
few MM charges. Displace one QM atom by ±h; compare `−(E(+h) − E(−h))/2h`
against the returned force. Requires tight `EDIFF` (~1e-8) and h ≈ 1e-3 Å so SCF
noise does not swamp the signal.

This tier validates the Eq. 3-4 corrections and **settles the open question
below**: if `TOTEN` already contains `∫ρ·V_ext`, energy and forces agree; if not,
they disagree by exactly that term.

**Zero-charge control.** All MM charges set to zero with the plugin fully
active — energy must reproduce the unembedded run to ~1e-8 eV. Separates "the
machinery is inert when it should be" from "the physics is right."

## Phasing

**Phase C (first):** all physics in `vasp_plugin.py`, written as pure module-level
functions with thin callbacks. Fastest path to a working end-to-end run; the
`VASP-Python/H_in_constant_field/vasp_plugin.py` reference is already most of the
way there. Tier 1 tests import the plugin module directly, so the fast test loop
is preserved even in this phase.

**Phase B (after C works):** cut the physics functions into
`interfaces/vasp/grid_potential.py`; the plugin imports them. Because of the
module-scope import discipline above, this is a file move rather than a rewrite.
Phase B is also what makes the PME path tractable, since `helpme_py` installed
into the `vasp_plugin` conda env is importable from the plugin process.

## Open questions

1. **Does VASP's `TOTEN` already include `∫ρ·V_ext`** once the plugin modifies
   `total_potential`, or must the plugin add it via `additions.total_energy`?
   Deliberately not guessed. Tier 2 measures it.

## Decisions already taken

- **Target build:** `install/vasp/vasp.6.6.1_nvhpc24/bin/vasp_std` with
  `OMP_NUM_THREADS=8`. Fastest measured (76.6 s vs 107.2 s plain; 90.2 s vs
  121.6 s with the plugin active) and energy-identical to the nvhpc 25.5 build
  to 8 digits. Run it with `nvhpc-openmpi3/24.1` + `intel-oneapi-mkl`, **no hdf5
  module** (that build has `-DVASP_HDF5` off), and `LD_LIBRARY_PATH` pointing at
  the `vasp_plugin` conda env's `lib`.
- **`NSIM = 16`** in the INCAR (~5% faster than the default 4; 32 and 64 are
  worse).

## References

- Todorova et al., arXiv:2510.23328, §3.4
- `VASP-Python/H_in_constant_field/vasp_plugin.py` — charge spreading, FFT
  Poisson, and the sign convention
- `interfaces/psi4/psi4_interface.py:64-167` — the two-mechanism reference
  (native point charges vs `EMBPOT` grid)
- `install/vasp_test/BENCHMARK_RESULTS.md` — build/performance context
