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
F_j = −q_j ∫ φ_QM(r)·∇g_σ(r − r_j) dr        ~21³ grid points per atom
```

| approach | cost per ionic step |
|---|---|
| spectral evaluation at 2685 points | ~8×10¹⁰ ops → hours |
| **local-box contraction** | ~2.5×10⁷ ops → seconds |

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

`_read_mm_forces()`; `compute_forces()` fills subsystem II rows; `_write_input`
adds `LVHAR = .TRUE.` when embedding is active.

## Error handling

**Never silently return zeros.** Every failure below has a variant in which the
run completes and the MM forces are quietly zero — which is exactly M2's current
behaviour and therefore invisible without an explicit check.

| failure | guard |
|---|---|
| `hartree_potential`/`ion_potential` are `None` | raise, naming `LVHAR = .TRUE.`; **no fallback** |
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
