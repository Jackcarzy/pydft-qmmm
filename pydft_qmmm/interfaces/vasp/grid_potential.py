"""Grid physics for VASP QM/MM electrostatic embedding.

Pure numpy.  This module must stay importable by the Python interpreter
embedded inside VASP, so it may not import anything beyond numpy -- not
scipy, and nothing from pydft_qmmm.  That restriction is what lets the
physics here be unit tested with no VASP present, which is how the sign
error in the nuclear force correction and the O(N*grid) charge spreading
were both caught in milliseconds rather than in 22-minute GPU jobs.

Split out of vasp_plugin.py once the embedding was validated end to end
(manuscript Figure 2b: a neutral H atom in a constant field, corrected
force 0.012 vs 5.820 kJ/mol/A uncorrected).
"""
from __future__ import annotations

import numpy as np

# Vacuum permittivity, e/(V*Angstrom).
EPS0 = 0.005526349358057108

# eV -> kJ/mol.
KJMOL_PER_EV = 96.48533212331


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


def _gaussian_box(position, inverse, dimensions, cell, sigma, prefactor, offsets):
    """The local Gaussian box for one charge.

    Factors out the ~25-line block shared, verbatim apart from what is
    done with the result, by spread_gaussian and
    contract_gaussian_gradient.  Using this for both is what makes
    Newton's third law hold term by term rather than approximately: it
    guarantees the two functions see the identical kernel, box and
    minimum-image convention, rather than relying on two hand-maintained
    copies staying in sync.

    Everything that does not vary per charge -- inverse, dimensions,
    prefactor and offsets -- is computed once by the caller and passed
    in here unchanged; this function only does the per-charge geometry
    (the box centre, the wrapped grid indices, and the minimum-imaged
    displacement and kernel on that box).  Recomputing any of those
    per-charge inputs inside this function would reintroduce the
    O(N_charges * N_grid) cost the box was built to avoid.

    Args:
        position: A single Cartesian position (Angstrom).
        inverse: The inverse of `cell`, i.e. cell**-1.
        dimensions: The grid shape, as an int array.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        sigma: The Gaussian width (Angstrom).
        prefactor: The Gaussian normalization, (2*pi*sigma**2)**-1.5.
        offsets: Per-axis arrays of grid-point offsets from the box
            centre, e.g. [np.arange(-h, h + 1) for h in half].

    Returns:
        (index, cartesian, gaussian): `index` is the per-axis list of
        wrapped grid indices suitable for `array[np.ix_(*index)]`;
        `cartesian` is the minimum-imaged Cartesian displacement from
        `position` to each point in the box; `gaussian` is the Gaussian
        kernel evaluated on that same box.
    """
    fractional = position @ inverse
    centre = np.round(fractional * dimensions).astype(int)
    index = [(centre[i] + offsets[i]) % dimensions[i] for i in range(3)]
    block = np.stack(
        np.meshgrid(
            *[(centre[i] + offsets[i]) / dimensions[i] for i in range(3)],
            indexing="ij",
        ),
        axis=-1,
    )
    delta = block - fractional
    delta -= np.round(delta)
    cartesian = delta @ cell
    squared = np.einsum("...k,...k->...", cartesian, cartesian)
    gaussian = prefactor * np.exp(-0.5 * squared / sigma**2)
    return index, cartesian, gaussian


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
        index, _, gaussian = _gaussian_box(
            position, inverse, dimensions, cell, sigma, prefactor, offsets,
        )
        rho[np.ix_(*index)] += charge * gaussian
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


def spectral_value_and_gradient(field, cell, points):
    """Evaluate a periodic grid field and its gradient at points.

    Both come from the same Fourier series,

        f(R)      = sum_G  fhat(G) exp(i G.R)
        grad f(R) = sum_G  i G fhat(G) exp(i G.R)

    so the gradient is EXACTLY the derivative of the value -- which is
    what molecular dynamics needs, and what neither of the alternatives
    provides:

      * interpolate_at + gradient_at mixes a trilinear value with a
        spectral slope.  Measured cost: a reproducible 14 kJ/mol/A
        finite-difference discrepancy (jobs 11566990, 11569337).
      * interpolate_at + interpolant_gradient_at is self-consistent but
        only C0, so at a grid node the slope is one-sided.  Measured
        cost: a 2.0 kJ/mol/A transverse force where symmetry demands
        zero.

    The series is also exact for a band-limited field and correctly
    periodic, unlike a minimum-image pairwise sum.

    Evaluated separably: exp(i G.R) factorizes over the three lattice
    directions, so this costs O(N_grid) work but only O(n) memory per
    point rather than materializing a full complex grid per point.

    Args:
        field: A scalar field on the grid.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        points: An Nx3 array of Cartesian positions (Angstrom).

    Returns:
        (values, gradients) with shapes (N,) and (N, 3).
    """
    points = np.atleast_2d(points)
    shape = field.shape
    coefficients = np.fft.fftn(field) / field.size
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    miller = [np.fft.fftfreq(n) * n for n in shape]
    values = np.zeros(len(points))
    gradients = np.zeros((len(points), 3))
    for index, point in enumerate(points):
        projection = reciprocal @ point
        phase = [
            np.exp(1j * miller[axis] * projection[axis]) for axis in range(3)
        ]
        values[index] = np.einsum(
            "ijk,i,j,k->", coefficients, *phase,
        ).real
        for component in range(3):
            total = 0.0 + 0.0j
            for axis in range(3):
                weighted = list(phase)
                weighted[axis] = (
                    miller[axis] * reciprocal[axis, component] * phase[axis]
                )
                total += np.einsum("ijk,i,j,k->", coefficients, *weighted)
            gradients[index, component] = (1j * total).real
    return values, gradients


def interpolant_gradient_at(field, cell, points):
    """Analytic gradient OF THE TRILINEAR INTERPOLANT at points.

    This is deliberately not the same as gradient_at.  gradient_at
    differentiates in reciprocal space, which is a better approximation
    to the true gradient of the underlying field but is NOT the
    derivative of the value interpolate_at returns -- trilinear
    interpolation is only C0, so its slope inside a cell differs from
    the spectral derivative by O(h).

    For molecular dynamics the energy and the force must be exactly
    consistent, or the integrator sees a non-conservative field.  Since
    the nuclear energy correction uses interpolate_at, the matching
    force must use this function.  Measured cost of getting that wrong:
    a reproducible 14 kJ/mol/A discrepancy in the finite-difference
    test (jobs 11566990 and 11569337, bit-identical), about 1.5% of the
    nuclear correction but roughly half the NET force, because the
    nuclear and electronic terms nearly cancel.

    Args:
        field: A scalar field on the grid.
        cell: A 3x3 array whose rows are lattice vectors (Angstrom).
        points: An Nx3 array of Cartesian positions (Angstrom).

    Returns:
        An Nx3 array of gradients (field units per Angstrom).
    """
    points = np.atleast_2d(points)
    shape = np.array(field.shape)
    inverse = np.linalg.inv(cell)
    fractional = (points @ inverse) % 1.0
    scaled = fractional * shape
    lower = np.floor(scaled).astype(int)
    weight = scaled - lower
    # d(value)/d(scaled coordinate), one column per axis.
    d_scaled = np.zeros((len(points), 3))
    for offset in np.ndindex(2, 2, 2):
        offset = np.array(offset)
        index = (lower + offset) % shape
        values = field[index[:, 0], index[:, 1], index[:, 2]]
        corner = np.where(offset == 1, weight, 1.0 - weight)
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            slope = 1.0 if offset[axis] == 1 else -1.0
            d_scaled[:, axis] += (
                values * slope * corner[:, others].prod(axis=1)
            )
    # Chain rule: scaled = (x @ inverse) * shape.
    return d_scaled @ (inverse * shape).T


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


def contract_gaussian_gradient(
        field, positions, charges, shape, cell, sigma, cutoff=6.0,
):
    """Force on each Gaussian charge sitting in a grid potential.

    This is the transpose of spread_gaussian.  That function builds a
    potential by summing q*g_sigma over a 6 sigma box; this one
    contracts a potential against grad g_sigma over the same box:

        F_j = -q_j * integral( field(r) * grad_{r_j} g_sigma(r - r_j) dr )

    Note grad_{r_j}, the derivative with respect to the CHARGE
    position, not with respect to r.  The two differ by a sign and
    reading it the wrong way inverts every MM force.

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
        select, cartesian, gaussian = _gaussian_box(
            position, inverse, dimensions, cell, sigma, prefactor, offsets,
        )
        # THREE sign flips, and dropping any one of them inverts every
        # MM force:
        #   grad_r g(r - r_j) = -(r - r_j)/sigma**2 * g
        #   d/dr_j carries the opposite sign of grad_r, giving
        #       dg/dr_j = +(r - r_j)/sigma**2 * g
        #   F_j = -dE/dr_j puts the leading minus back.
        # Net: F_j = -q_j/sigma**2 * integral( phi (r - r_j) g ).
        weight = gaussian * field[np.ix_(*select)]
        forces[index] = -charge * d_volume * np.einsum(
            "ijk,ijkc->c", weight, cartesian,
        ) / sigma**2
    return forces


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
