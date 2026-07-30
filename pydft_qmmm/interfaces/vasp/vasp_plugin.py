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
