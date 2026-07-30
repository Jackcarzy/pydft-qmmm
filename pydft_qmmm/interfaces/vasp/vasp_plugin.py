"""External-potential plugin for VASP QM/MM electrostatic embedding.

VASP imports this module and calls into it; never run it directly.  The
interface copies this file into each run directory, where VASP's
embedded interpreter picks it up as ``vasp_plugin``.

Dependencies are deliberately limited to numpy, and nothing from
pydft_qmmm is imported, because this module is executed by the Python
interpreter embedded inside VASP.  That restriction is also what lets
the physics below be unit tested without VASP present.
"""
from __future__ import annotations

import traceback
import warnings

import numpy as np

# Vacuum permittivity, e/(V*Angstrom).  Defined locally rather than
# imported so this module stays dependency-free.
EPS0 = 0.005526349358057108

CHARGE_FILE = "MM_CHARGES"
SENTINEL = "PLUGIN_FIRED.txt"
ERROR_FILE = "PLUGIN_ERROR.txt"

_CACHE = {}


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
        if count == 0:
            # Skip loadtxt entirely: an empty subsystem II is a normal
            # case, and loadtxt warns on empty input.
            return np.zeros((0, 3)), np.zeros(0), step, sigma
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
    return data[:, :3].copy(), data[:, 3].copy(), step, sigma


def _fractional_grid(shape):
    """Build the fractional coordinates of every grid point."""
    axes = [np.arange(n, dtype=np.float64) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def _axis_spacings(shape, cell):
    """Grid spacing along each axis, as perpendicular widths.

    For a general (non-orthogonal) cell the spacing that matters is the
    distance between adjacent lattice planes, which is the cell volume
    divided by the area of the opposite face.
    """
    volume = abs(np.linalg.det(cell))
    widths = np.array([
        volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ])
    return widths / np.array(shape, dtype=np.float64)


def spread_gaussian(positions, charges, shape, cell, sigma, cutoff=6.0):
    """Spread point charges onto the grid as normalized Gaussians.

    A point charge cannot be represented on a finite FFT grid, so each
    is smeared with width sigma.

    Only grid points within ``cutoff * sigma`` of a charge are touched.
    Without that restriction the cost is O(N_charges * N_grid), which is
    ruinous at production scale: 1149 MM charges on a 160**3 grid took
    935 s per evaluation, versus well under a second here.  A 3D
    Gaussian has about 7e-8 of its mass beyond 6 sigma, so the truncated
    charge is conserved to roughly one part in 1e7.

    Displacements are minimum-imaged in fractional coordinates, which is
    exact for the nearest image and negligible in error while sigma is
    small against the cell.

    Args:
        positions: An Nx3 array of positions (Angstrom).
        charges: An N array of charges (e).
        shape: The grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        sigma: The Gaussian width (Angstrom).
        cutoff: Truncation radius in units of sigma.

    Returns:
        The charge density on the grid (e/Angstrom**3).
    """
    shape = tuple(int(n) for n in shape)
    rho = np.zeros(shape, dtype=np.float64)
    if len(charges) == 0:
        return rho
    inverse = np.linalg.inv(cell)
    prefactor = (2.0 * np.pi * sigma**2) ** -1.5
    dimensions = np.array(shape)
    half = np.ceil(
        cutoff * sigma / _axis_spacings(shape, cell),
    ).astype(int)
    if np.any(2 * half + 1 >= dimensions):
        # The cutoff box wraps the whole cell, so a local block would
        # double-count images.  Fall back to evaluating on every point.
        fractional = _fractional_grid(shape)
        for position, charge in zip(positions, charges):
            delta = fractional - (position @ inverse)
            delta -= np.round(delta)
            cartesian = delta @ cell
            squared = np.einsum("...k,...k->...", cartesian, cartesian)
            rho += charge * prefactor * np.exp(-0.5 * squared / sigma**2)
        return rho
    offsets = [np.arange(-h, h + 1) for h in half]
    for position, charge in zip(positions, charges):
        centre = np.round((position @ inverse) * dimensions).astype(int)
        index = [(centre[i] + offsets[i]) % dimensions[i] for i in range(3)]
        block = np.stack(
            np.meshgrid(
                *[
                    (centre[i] + offsets[i]) / dimensions[i]
                    for i in range(3)
                ],
                indexing="ij",
            ),
            axis=-1,
        )
        delta = block - (position @ inverse)
        delta -= np.round(delta)
        cartesian = delta @ cell
        squared = np.einsum("...k,...k->...", cartesian, cartesian)
        rho[np.ix_(*index)] += (
            charge * prefactor * np.exp(-0.5 * squared / sigma**2)
        )
    return rho


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
    background, the same convention VASP uses for charged cells.

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

    Args:
        positions: An Nx3 array of positions (Angstrom).
        charges: An N array of charges (e).
        shape: The grid dimensions.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        sigma: The Gaussian width (Angstrom).

    Returns:
        The external potential as electron potential ENERGY (eV), which
        is the negative of the electrostatic potential.  This is the
        quantity VASP's total_potential expects.
    """
    rho = spread_gaussian(positions, charges, shape, cell, sigma)
    return -poisson_fft(rho, cell)


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
    for offset in np.ndindex(2, 2, 2):
        offset = np.array(offset)
        index = (lower + offset) % shape
        corner = np.where(offset == 1, weight, 1.0 - weight)
        result += (
            field[index[:, 0], index[:, 1], index[:, 2]]
            * corner.prod(axis=1)
        )
    return result


def gradient_at(field, cell, points):
    """Return the gradient of a periodic grid field at points.

    The derivative is taken in reciprocal space, which is exact for a
    band-limited field, and the three components are then interpolated,
    so the gradient is consistent with interpolate_at.

    Args:
        field: A scalar field on the grid.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        points: An Nx3 array of Cartesian positions (Angstrom).

    Returns:
        An Nx3 array of gradients (field units per Angstrom).
    """
    points = np.atleast_2d(points)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    axes = [np.fft.fftfreq(n) * n for n in field.shape]
    miller = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    g_vectors = miller @ reciprocal
    field_g = np.fft.fftn(field)
    gradient = np.zeros((len(points), 3))
    for axis in range(3):
        component = np.fft.ifftn(1j * g_vectors[..., axis] * field_g).real
        gradient[:, axis] = interpolate_at(component, cell, points)
    return gradient


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


def electron_interaction_energy(charge_density, v_ext):
    """Energy of the electrons in the external potential, in eV.

    VASP normalizes its grid charge density so that

        sum(charge_density) / N_grid == NELECT

    (verified against a real run: 12881756160.000002 / 7741440 ==
    1664.0).  The electron number density is therefore
    n(r) = charge_density / V, and

        integral n V_ext dV = sum(charge_density * V_ext) / N_grid

    with no explicit cell volume: the V from n cancels the V from dV.
    Multiplying by the volume, as an earlier draft did, overshoots by
    ~2.7e4 for this cell.
    """
    if charge_density is None:
        raise RuntimeError(
            "charge_density is None, so the electron-V_ext energy term "
            "cannot be formed.  Set LVHAR = .TRUE. in the INCAR so that "
            "VASP populates it.",
        )
    density = np.asarray(charge_density)
    if density.shape != v_ext.shape:
        raise RuntimeError(
            f"charge_density has shape {density.shape} but V_ext has "
            f"{v_ext.shape}; they must share VASP's fine grid.",
        )
    return float(np.sum(density * v_ext) / density.size)


def local_potential(constants, additions):
    """Add the MM external potential to VASP's local potential.

    Defines the PLUGINS/LOCAL_POTENTIAL interface.  Called once per SCF
    step.

    Also reports the associated energy.  VASP does NOT account for this
    automatically: pot.F passes additions.total_energy straight into
    E%EPLUGINS (assigning, not accumulating), and electron.F sums

        TOTEN = EBANDSTR + DENC + ... + Ediel_sol + ESCPC + EPLUGINS

    so leaving it at zero makes TOTEN inconsistent with the forces,
    which DO include the full effect.  That inconsistency is exactly
    what the Tier 2 finite-difference test measured.
    """
    try:
        v_ext = _external_potential(constants)
        additions.total_potential += v_ext
        additions.total_energy = electron_interaction_energy(
            constants.charge_density, v_ext,
        )
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
        additions.total_energy += -float(
            np.sum(valence * interpolate_at(v_ext, cell, positions)),
        )
        additions.forces += (
            valence[:, None] * gradient_at(v_ext, cell, positions)
        )
    except Exception as exc:
        _record_error(exc)
        raise
