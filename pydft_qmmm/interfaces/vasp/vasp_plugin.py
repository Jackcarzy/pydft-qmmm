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
