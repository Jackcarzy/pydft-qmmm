"""Spectral nuclear terms on the FFT mesh, batched over atoms (CuPy when present).

The periodic PySCF interface couples QM nuclei to the external potential
through its Fourier interpolant:

    V(R)      = Re sum_G  Vhat(G) exp(i G.R)
    grad V(R) = Re sum_G  i G Vhat(G) exp(i G.R)

and deposits nuclear charges with the transposed sum, so that
sum_g q_g V_g = sum_A Z_A V(R_A). grid_potential.py evaluates both one atom
at a time with full-mesh einsums on the host, which costs ~N_atoms * N_grid
complex operations in NumPy. The phase factor separates by axis,
exp(i G.R) = prod_a exp(i m_a p_a), so both sums are one matrix product over
the last axis followed by two small contractions over the other two. The
result is the same Fourier sum, evaluated in a different order.

grid_potential.py stays pure NumPy for the interpreter embedded in VASP; this
module is used only by the PySCF-PBC path.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _xp() -> Any:
    """CuPy when a GPU is usable, else NumPy."""
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:  # noqa: BLE001 - no CuPy or no device: host path
        pass
    return np


def _host(array: Any) -> np.ndarray:
    get = getattr(array, "get", None)
    return np.asarray(get() if callable(get) else array)


def _axis_phases(xp: Any, shape: tuple[int, int, int], cell: np.ndarray,
                 points: np.ndarray, sign: float):
    """exp(sign * i * m_a * p_a) per axis, the integer Miller indices, and G basis."""
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=float)).T
    projection = np.asarray(points, dtype=float) @ reciprocal.T     # (n, 3)
    miller = [np.fft.fftfreq(n) * n for n in shape]
    phases = [
        xp.exp(sign * 1j * xp.asarray(projection[:, a])[:, None]
               * xp.asarray(miller[a])[None, :])
        for a in range(3)
    ]
    return phases, [xp.asarray(m) for m in miller], reciprocal


def spectral_value_and_gradient(field: Any, cell: np.ndarray, points: np.ndarray):
    """Fourier-interpolated value and Cartesian gradient of a periodic grid field.

    Same result as grid_potential.spectral_value_and_gradient (row lattice
    vectors in cell, Å; points Nx3 in Å). Returns host arrays (N,), (N, 3).
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if not len(points):
        return np.zeros(0), np.zeros((0, 3))
    xp = _xp()
    field = xp.asarray(field, dtype=float)
    nx, ny, nz = field.shape
    n = len(points)
    coef = xp.fft.fftn(field) / field.size
    (p0, p1, p2), (m0, m1, m2), reciprocal = _axis_phases(
        xp, field.shape, cell, points, 1.0)
    c2 = coef.reshape(nx * ny, nz)
    t = (c2 @ p2.T).reshape(nx, ny, n)                    # sum over k
    tm = (c2 @ (p2 * m2[None, :]).T).reshape(nx, ny, n)   # ... with m_z
    u = xp.einsum("ijn,nj->in", t, p1)                    # sum over j
    u1 = xp.einsum("ijn,nj->in", t, p1 * m1[None, :])
    u2 = xp.einsum("ijn,nj->in", tm, p1)
    value = xp.einsum("in,ni->n", u, p0)                  # sum over i
    d0 = xp.einsum("in,ni->n", u, p0 * m0[None, :])
    d1 = xp.einsum("in,ni->n", u1, p0)
    d2 = xp.einsum("in,ni->n", u2, p0)
    d = xp.stack([d0, d1, d2], axis=1)                    # per Miller axis
    gradient = (1j * (d @ xp.asarray(reciprocal))).real
    return _host(value.real), _host(gradient)


def spectral_charge_deposition(shape: tuple[int, int, int], cell: np.ndarray,
                               points: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """Grid charges (e per point, C order) whose sum with any grid field V equals
    sum_A Z_A V(R_A) under the Fourier interpolant above. Host array, flattened.
    """
    shape = tuple(int(s) for s in shape)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    xp = _xp()
    nx, ny, nz = shape
    (p0, p1, p2), _, _ = _axis_phases(xp, shape, cell, points, -1.0)
    q = xp.asarray(np.asarray(charges, dtype=float))
    pair = (p0[:, :, None] * p1[:, None, :] * q[:, None, None]).reshape(len(q), nx * ny)
    coef = (pair.T @ p2).reshape(shape)
    return _host(xp.fft.ifftn(coef).real.reshape(-1))
