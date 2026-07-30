"""External-potential plugin for VASP QM/MM electrostatic embedding.

VASP imports this module and calls into it; never run it directly.  The
interface copies this file, together with grid_potential.py, into each
run directory, where VASP's embedded interpreter picks it up as
``vasp_plugin``.

Only the callbacks and their caching live here.  The physics is in
grid_potential, which is deliberately free of any dependency beyond
numpy so that it can be exercised without VASP present.
"""
from __future__ import annotations

import os
import traceback
import warnings

import numpy as np

try:
    from .grid_potential import EPS0
    from .grid_potential import build_external_potential
    from .grid_potential import electron_interaction_energy
    from .grid_potential import gradient_at
    from .grid_potential import interpolant_gradient_at
    from .grid_potential import interpolate_at
    from .grid_potential import read_mm_charges
    from .grid_potential import spectral_value_and_gradient
except ImportError:
    # VASP copies this file into the run directory and imports it as a
    # top-level module, where the package-relative form is unavailable.
    from grid_potential import EPS0
    from grid_potential import build_external_potential
    from grid_potential import electron_interaction_energy
    from grid_potential import gradient_at
    from grid_potential import interpolant_gradient_at
    from grid_potential import interpolate_at
    from grid_potential import read_mm_charges
    from grid_potential import spectral_value_and_gradient

__all__ = [
    "EPS0", "CHARGE_FILE", "SENTINEL", "ERROR_FILE",
    "build_external_potential", "electron_interaction_energy",
    "gradient_at", "interpolant_gradient_at", "interpolate_at",
    "read_mm_charges", "spectral_value_and_gradient",
    "reset_cache", "local_potential", "force_and_stress",
]

CHARGE_FILE = "MM_CHARGES"
SENTINEL = "PLUGIN_FIRED.txt"
ERROR_FILE = "PLUGIN_ERROR.txt"
PME_FILE = "PME_DATA"

_CACHE = {}


def reset_cache():
    """Discard the cached external potential.  For tests."""
    _CACHE.clear()


def _record_error(exc):
    """Persist a traceback before re-raising.

    VASP wraps the callback in a bare ``except`` (see
    ``src/plugins/src/vasp/_apply_interface.py``) and turns any
    exception into a PYTHON_EXCEPTION return code.  It does abort, but
    the traceback goes to stdout where it is easily lost among VASP's
    own output, so write it somewhere findable first.
    """
    with open(ERROR_FILE, "w") as fh:
        fh.write("".join(traceback.format_exception(exc)))


def _external_potential(constants):
    """Return V_ext on the current grid, building it at most once.

    The MM geometry is fixed for the duration of one VASP launch, so the
    field is built on the first callback and reused for every subsequent
    SCF step.  The cache is keyed on the grid shape so that a changed
    grid rebuilds rather than returning a mismatched array.
    """
    shape = tuple(int(n) for n in constants.shape_grid)
    cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
    if _CACHE.get("shape") == shape and _CACHE.get("v_ext") is not None:
        return _CACHE["v_ext"]
    if os.path.isfile(PME_FILE):
        # PME path: the driver shipped the whole system and the Ewald
        # parameters, and helPME evaluates on VASP's own grid.  Imported
        # lazily so the cutoff path keeps working without helpme_py.
        try:
            from .pme_external import build_pme_potential
        except ImportError:
            from pme_external import build_pme_potential
        v_ext = build_pme_potential(PME_FILE, shape, cell)
        _CACHE["shape"] = shape
        _CACHE["v_ext"] = v_ext
        with open(SENTINEL, "w") as fh:
            fh.write("local_potential callback executed (PME)\n")
            fh.write(f"shape_grid   = {shape}\n")
            fh.write(f"min V_ext eV = {float(v_ext.min()):.6f}\n")
            fh.write(f"mean V_ext   = {float(v_ext.mean()):.3e}\n")
        return v_ext
    positions, charges, _, sigma = read_mm_charges(CHARGE_FILE)
    net_charge = float(np.sum(charges)) if len(charges) else 0.0
    if abs(net_charge) > 1e-6:
        # The G=0 term is dropped in the Poisson solve, which is
        # equivalent to a uniform neutralizing background.  Absolute
        # energies then carry a constant offset; differences at fixed
        # composition do not.
        warnings.warn(
            f"subsystem II carries a net charge of {net_charge:.4f} e.  "
            "Absolute embedded energies are shifted by a constant from "
            "the neutralizing-background (G=0) convention.",
            RuntimeWarning,
            stacklevel=2,
        )
    v_ext = build_external_potential(positions, charges, shape, cell, sigma)
    _CACHE["shape"] = shape
    _CACHE["v_ext"] = v_ext
    with open(SENTINEL, "w") as fh:
        fh.write("local_potential callback executed\n")
        fh.write(f"shape_grid   = {shape}\n")
        fh.write(f"mm_charges   = {len(charges)}\n")
        fh.write(f"net_charge   = {net_charge:.6f}\n")
        fh.write(f"sigma        = {sigma}\n")
        fh.write(f"min V_ext eV = {float(v_ext.min()):.6f}\n")
        fh.write(f"mean V_ext   = {float(v_ext.mean()):.3e}  (G=0 dropped)\n")
    return v_ext


def local_potential(constants, additions):
    """Add the MM external potential to VASP's local potential.

    Defines the PLUGINS/LOCAL_POTENTIAL interface.  Called once per SCF
    step.

    additions.total_energy is deliberately LEFT ALONE.  MEASURED, not
    assumed (jobs 11566990 and 11567275):

        without an energy term  analytical - numerical = (14, 14, 5)
        reporting int(rho V_ext) ->                      (-266, -985, 217)
        the nuclear correction itself is                 (-272, -950, 205)

    Adding the term shifted the result by very nearly the whole nuclear
    correction, i.e. by the electronic response that almost cancels it.
    So VASP's TOTEN ALREADY contains int(rho V_ext): the band-structure
    energy picks it up once V_ext is added to the local potential, and
    nothing subtracts it again.  E%EPLUGINS exists for energies VASP
    cannot know about -- a field's self-energy, a constraint term --
    not for this one.

    This contradicts what the design document guessed, and is the reason
    the Tier 2 test was written as a measurement rather than a check.
    """
    try:
        additions.total_potential += _external_potential(constants)
    except Exception as exc:
        _record_error(exc)
        raise


def force_and_stress(constants, additions):
    """Apply the nuclear terms VASP omits.

    Defines the PLUGINS/FORCE_AND_STRESS interface.  VASP does not
    include the interaction between the external potential and the
    pseudo-ion cores, so add

        dE_I = -Z_I V_ext(R_I)
        dF_I = +Z_I grad V_ext(R_I)

    NOTE THE FORCE SIGN.  The force must be the negative gradient of the
    energy it accompanies:

        F = -grad(dE_I) = -grad(-Z_I V_ext) = +Z_I grad V_ext

    Writing dF_I = -Z_I grad V_ext (as the manuscript's Eq. 4 appears to,
    and as this project's spec originally did) is inconsistent with
    Eq. 3 and flips every nuclear force.  Checked physically: for an MM
    charge of +1 e and a pseudo-ion of +11 e two Angstrom away, V_ext is
    negative and rising with r, so grad V_ext > 0 and the ion is pushed
    away -- repulsion, as two positive charges must.  The
    test_energy_and_force_corrections_are_consistent test finite
    differences dE against dF to keep the two locked together.

    Z_I is the pseudopotential valence charge ZVAL, NOT the atomic
    number: VASP's nuclei are pseudo-ions carrying only valence charge,
    so the atomic number would over-count by the core electrons (79 vs
    11 for gold).

    V_ext is interpolated from the same grid the electrons see, rather
    than summed pairwise over MM charges.  A pairwise sum would be exact
    but non-periodic, while the grid solve is periodic, and the two
    halves of the QM subsystem must feel a consistent potential.
    """
    try:
        cell = np.asarray(constants.lattice_vectors, dtype=np.float64)
        positions = np.asarray(constants.positions) @ cell
        if len(positions) == 0:
            return
        v_ext = _external_potential(constants)
        # ion_types is already 0-indexed: adjust_indexing subtracts one
        # from every IndexArray before the plugin sees it.
        valence = np.asarray(constants.ZVAL, dtype=np.float64)[
            np.asarray(constants.ion_types, dtype=int)
        ]
        # One Fourier series supplies both, so the force is exactly the
        # derivative of the energy.  See spectral_value_and_gradient.
        potential, gradient = spectral_value_and_gradient(
            v_ext, cell, positions,
        )
        additions.total_energy += -float(np.sum(valence * potential))
        additions.forces += valence[:, None] * gradient
    except Exception as exc:
        _record_error(exc)
        raise
