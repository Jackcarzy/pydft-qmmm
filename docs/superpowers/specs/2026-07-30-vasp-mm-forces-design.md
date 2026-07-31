# VASP QM/MM: Forces on MM Atoms (Milestone 3) — Design

**Date:** 2026-07-30
**Status:** Approved for planning
**Scope:** The QM→MM back-reaction — forces exerted by the QM charge
distribution on the embedded MM point charges.

## Goal

Milestone 2 sends MM charges *into* VASP as an external potential. M3 sends the
reaction *back*. Without it the QM region is pushed by the MM charges while the
MM charges feel nothing, so **momentum is not conserved** and QM/MM MD is not
usable for production.

## Scope

**In scope**

- Forces on **subsystem II** (the near-field charges) via the analytic channel.
- Forces on **subsystem III** via the PME reciprocal channel, staged second.
- Newton's third law and finite-difference validation.

**Out of scope**

- Stress contributions from the MM charges.
- Re-opening the unexplained ~16 kJ/mol/Å QM/MM gradient residual from M2.

## Why this must live in the plugin

The force needs the **converged QM charge distribution**, which exists only
inside VASP's process. The alternative — having VASP write `LOCPOT`/`CHGCAR`
and parsing it driver-side — is impractical: 320³ is 33 million values as text
per evaluation.

VASP's `additions.forces` **cannot** carry the result: it is sized
`3 × number_ions`, the QM ions only. MM atoms do not exist in VASP's
calculation; they enter only as a potential. So the forces return the way the
charges went out — a file — reusing the handoff already carrying `MM_CHARGES`
and `PME_DATA`.

## The computational constraint that determined the design

The obvious formulation — evaluate the QM field at each MM position — is not
viable. `spectral_value_and_gradient` is O(N_points × N_grid): ~9 s for **3** QM
nuclei at 360³, hence **~2.2 hours** for 2685 MM atoms per ionic step.

Use the symmetry of the interaction instead:

```
E_int = ∫ ρ_QM(r)·φ_MM(r) dr  =  ∫ φ_QM(r)·ρ_MM(r) dr
```

Differentiating the **second** form with respect to `r_j` puts the derivative on
the MM charge's own Gaussian, which is **local**:

```
F_j = −q_j ∫ φ_QM(r)·∇_{r_j} g_σ(r − r_j) dr    ~21³ grid points per atom
```

| approach | cost per ionic step |
|---|---|
| spectral evaluation at 2685 points | ~8×10¹⁰ ops → hours |
| **local-box contraction** | ~2.5×10⁷ ops → seconds |

The gradient is taken with respect to **`r_j`**, the charge position, not with
respect to `r`. The two differ by a sign, and `F = −∂E/∂r_j` supplies a second
one; applying only one of the two returns the energy gradient and inverts every
MM force.

This is the **transpose of `spread_gaussian`** — same kernel, same σ, same 6σ
box, `∇g_σ` in place of `g_σ`. The same locality that made charge spreading
1870× faster makes the back-reaction affordable.

It is also the more accurate choice, not a compromise: the force is
differentiated with respect to precisely the representation the QM felt, so
`Σ F_QM + Σ F_MM = 0` holds **term by term**. A third-law violation then means a
bug, not an expected decomposition artifact.

## Source of φ_QM

VASP hands the plugin `hartree_potential` and `ion_potential` on the grid.
Reconstructing them instead would mean approximating PAW cores as point charges
— the same modelling error suspected behind M2's unexplained residual.

**Constraint:** both fields exist only on `ConstantsLocalPotential`, not on
`ConstantsForceAndStress`.

| | `local_potential` | `force_and_stress` |
|---|---|---|
| `charge_density` | yes | yes |
| `hartree_potential` | **yes** | no |
| `ion_potential` | **yes** | no |
| called | every SCF step | once, **converged** |

So `local_potential` caches the potentials on every call and `force_and_stress`
consumes the most recent. The plugin cannot detect which SCF call is last and
does not need to: `force_and_stress` runs after all of them.

**`LVHAR` is NOT required** — an earlier draft of this spec said it was. Both
fields are allocated whenever `PLUGINS/LOCAL_POTENTIAL = T`:

```fortran
IF (INFO%PLUGIN%LOCAL_POTENTIAL) THEN
   ALLOCATE(HARTREE_POTENTIAL(GRIDC%MPLWV)) ; ... FFT3D(...)
ENDIF
```

Verified in a real run (job 11573109) by probing the callback: all three of
`charge_density`, `hartree_potential` and `ion_potential` arrive populated with
shape (140, 140, 392), no `LVHAR` set. `ion_potential` spans −61.1 to +2.3 eV —
strongly negative near the cores, as an electron-referenced potential must be,
which corroborates the sign convention above.

Cost: two cached 320³ arrays, ~528 MB, about 1% of the 64 GB these runs already
request.

## Architecture

**VASP process** (`vasp_plugin.py`)

1. `local_potential` — cache `hartree_potential + ion_potential`.
2. `force_and_stress` — form `φ_QM`, contract against `∇g_σ` per MM charge,
   write `MM_FORCES`.

**PyDFT-QMMM process** (`vasp_interface.py`)

3. `compute_forces()` reads `MM_FORCES` into the subsystem II rows it currently
   zeroes.

**Subsystem III (PME), staged second.** Those atoms act on the QM through
helPME's reciprocal sum, so the reaction must return the same way: helPME with
`ρ_QM` as the source. This is the one genuinely new piece of machinery. The
near-field channel ships first so that PME failures can be diagnosed against a
working reference — the staging that made M2's PME problems tractable.

## Components

### `grid_potential.py` (numpy only)

| function | responsibility |
|---|---|
| `contract_gaussian_gradient(field, positions, charges, shape, cell, sigma, cutoff=6.0)` | → N×3 forces; transpose of `spread_gaussian` |
| `electrostatic_potential_from_vasp(hartree, ion)` | → `φ_QM` in volts, applying the `V = −φ` conversion once, in one place |

The `numpy`-only discipline holds: this is what has kept the physics testable in
milliseconds rather than in 22-minute GPU jobs.

### Handoff

`vasp_utils.write_mm_forces` / `read_mm_forces`, mirroring the `MM_CHARGES`
contract exactly: count, step stamp, `%.12e`.

### Interface

`_read_mm_forces()`; `compute_forces()` fills subsystem II rows. No INCAR change
is needed.

## Error handling

**Never silently return zeros.** Every failure below has a variant in which the
run completes and the MM forces are quietly zero — which is exactly M2's current
behaviour and therefore invisible without an explicit check.

| failure | guard |
|---|---|
| `hartree_potential`/`ion_potential` are `None` | raise; **no fallback**. Not an INCAR problem — see below |
| `MM_FORCES` absent after an embedded run | raise, as the sentinel check already does |
| stale `MM_FORCES` | step stamp, as `MM_CHARGES` uses |
| forces identically zero | raise — otherwise indistinguishable from M2 |

## Testing

**Tier 1 — no VASP.**

- **Transpose identity:** for an analytic `φ`, the contraction reproduces
  `q·∇φ` at the charge position.
- **Third law, synthetic:** one Gaussian charge in the `φ` of another; forces
  equal and opposite to machine precision.
- **σ/box consistency:** contraction and spreading must use identical σ and
  cutoff; deliberately mismatching them must fail the third-law test.

**Tier 2 — VASP, in this order.**

1. **Newton's third law:** `|Σ F_QM + Σ F_MM| / max|F|` on the SPC/E system.
   One run.
2. **Finite difference on one MM atom:** displace `r_j`, check
   `F_j = −∂E/∂r_j`. ~7 runs, with **`ISTART = 0`** — restart reuse poisoned the
   M2 finite-difference test and cost three GPU jobs to diagnose.

**Recorded trap:** the third law can only balance over atoms whose interaction
is actually represented. With the near-field channel alone, subsystem III feels
nothing, so the sum must be taken over **subsystem I + II only**. Including III
would show a violation that is expected rather than a bug — the same trap as
M2's empty-subsystem-III fixture.

## Discovered during implementation: VASP destroys the net force

Not anticipated by this design, found by job 11580378 failing the third-law
test with a 40% violation, and load-bearing for anything built on top.

`force.F` calls the plugin and then removes the drift:

```fortran
CALL PLUGINS_FORCE_AND_STRESS(..., TIFOR, ...)      ! :1811
...
IF (DYN%IBRION/=0 .OR. LCOMPAT) THEN
   IF (LREMOVE_DRIFT) CALL SYMVEC(T_INFO%NIONS,TIFOR)   ! :1827
ENDIF
```

`SYMVEC` (`dyna.F:257`) subtracts the mean force from every ion. That is right
for an isolated periodic cell, where the total energy really is translationally
invariant so `Σ F` must vanish — and wrong under embedding, where `V_ext` breaks
that invariance and the net force **is** the momentum the MM subsystem transfers
to the QM one.

`LREMOVE_DRIFT` is a hardcoded call-site argument, **not an INCAR tag**, so it
cannot be switched off from input; and `IBRION = -1` satisfies the outer gate,
so the removal always runs for these single points. Because the callback
receives `TIFOR` *before* `SYMVEC`, the plugin is the only place the true net
can be reconstructed: `sum(constants.forces) + sum(nuclear term)`, written to
`QM_NET_FORCE` and spread back over the QM atoms by the driver.

**`SYMVEC` exempts the single-atom case** — `IF (NIONS==1) RETURN`. This is why
the problem hid for so long:

| test | QM atoms | net force |
|---|---|---|
| M2 Figure 2b, H in a constant field | 1 | preserved |
| M2 gradient test, one QM water | 3 | destroyed |
| M3 third law, one QM water | 3 | destroyed |

Milestone 2's headline validation ran on the single geometry in the suite that
is immune, so it gave no warning about the multi-atom case.

This also explains **M2's unexplained ~16 kJ/mol/Å gradient residual**: `SYMVEC`
subtracts the same vector from every atom, so relative forces survive and only
the centre-of-mass component is lost — an error of `|net|/N` per atom, which
here is `|(9.90, -37.70, 29.28)|/3 = 16.3 kJ/mol/Å`.

Measured, job 11580631, in eV/A:

```
sum(VASP's own components) = ( 3.007096150151, 12.73685590952, -3.088900111545)
sum(nuclear term)          = (-2.904533542613,-13.12762391728,  3.392336744396)
true net                   = ( 0.102562,       -0.390768,        0.303437)
```

against `-sum(F_MM) = (0.102481, -0.387915, 0.302087)` — the third law holds to
0.28% once the net is restored, against 40% before.

## Open questions

1. **Does `ion_potential` need a sign or pseudization correction at MM
   positions?** It is the local pseudopotential — what *electrons* feel from the
   cores, including pseudization inside the core radius. MM atoms sit outside
   that radius, so it should approach the bare form, but the convention is
   electron-referenced and that is exactly the trap that produced M2's Eq. 4
   sign error. The Tier 1 third-law test settles it.

## References

- Todorova et al., arXiv:2510.23328, Eqs. 3–4
- Pederson & McDaniel, J. Chem. Phys. **156**, 174105 (2022) — Fig. 4; note
  `PMEExcludedPotential.compute_forces` returns zeros, so the PME back-reaction
  has no working precedent in this codebase
- `interfaces/psi4/psi4_interface.py:228` — `gradient_on_charges()`, the
  reference implementation for the cutoff case
- `docs/superpowers/specs/2026-07-30-vasp-electrostatic-embedding-design.md`
