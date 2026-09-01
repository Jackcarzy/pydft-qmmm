"""SCF-bearing tests for the periodic PySCF interface.

Every test here runs an SCF and belongs on a compute node.  The login
node has a 4 GB per-user cap and is heavily throttled.  Run with:

    sbatch ../pbc-runs/integration.slurm
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_embedding_shifts_the_energy(pyscf_pbc_embedded_interface):
    """A charged environment must move the QM energy, and sanely."""
    interface, bare = pyscf_pbc_embedded_interface
    embedded = interface.compute_energy()
    assert embedded != pytest.approx(bare, abs=1e-6)
    # An MM environment of ordinary point charges shifts a small QM
    # region by tens of kJ/mol, not by thousands.  A four-figure shift
    # is the signature of a quadrature failure, so this bound is a
    # guard rather than a curiosity: do not widen it to make it pass.
    assert abs(embedded - bare) < 1000.0


def test_components_sum_to_the_energy(pyscf_pbc_embedded_interface):
    interface, _ = pyscf_pbc_embedded_interface
    components = interface.compute_components()
    assert sum(components.values()) == pytest.approx(
        interface.compute_energy(), rel=1e-9,
    )


def _central_difference(interface, atom, step):
    """The numerical force on one atom, by central differences."""
    system = interface.system
    numeric = np.zeros(3)
    for axis in range(3):
        original = float(system.positions[atom, axis])
        positions = np.asarray(system.positions).copy()
        positions[atom, axis] = original + step
        system.positions = positions
        plus = interface.compute_energy()
        positions = np.asarray(system.positions).copy()
        positions[atom, axis] = original - step
        system.positions = positions
        minus = interface.compute_energy()
        positions = np.asarray(system.positions).copy()
        positions[atom, axis] = original
        system.positions = positions
        numeric[axis] = -(plus - minus) / (2.0 * step)
    return numeric


def test_qm_forces_match_central_differences(pyscf_pbc_embedded_interface):
    """Every QM force channel must differentiate the reported energy."""
    interface, _ = pyscf_pbc_embedded_interface
    atom = sorted(interface.system.select("subsystem I"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 1e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=5e-3, atol=1.0)


def test_pulay_term_is_live(pyscf_pbc_embedded_interface, monkeypatch):
    """Deleting channel 2 must break the finite-difference agreement.

    Without this guard a silently missing Pulay term looks exactly like
    a correct implementation: the patched-hcore gradient still runs and
    still returns plausible numbers.
    """
    from pydft_qmmm.interfaces.pyscf_pbc import pbc_forces
    interface, _ = pyscf_pbc_embedded_interface
    atom = sorted(interface.system.select("subsystem I"))[0]
    numeric = _central_difference(interface, atom, 1e-3)
    monkeypatch.setattr(
        pbc_forces,
        "pulay_forces",
        lambda backend, state, natoms: np.zeros((natoms, 3)),
    )
    crippled = interface.compute_forces()[atom]
    assert not np.allclose(crippled, numeric, rtol=5e-3, atol=1.0)


def test_mm_forces_match_central_differences(pyscf_pbc_embedded_interface):
    """The reaction on a static MM charge must differentiate the energy."""
    interface, _ = pyscf_pbc_embedded_interface
    atom = sorted(interface.system.select("subsystem II"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 2e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1.0)


def test_embedding_channels_conserve_momentum(pyscf_pbc_embedded_interface):
    """The embedding this interface adds must exert no net force.

    Channels 2, 3 and 4 are the QM-MM interaction: the Pulay term, the
    nuclear term, and the reaction on the static MM charges.  They are
    internal forces, so they must cancel to machine-ish precision by
    Newton's third law, and a double count in the exclusion sets would
    show up here as a large residual.

    Channel 1, PySCF's own periodic gradient, is deliberately excluded.
    It carries a net force of its own that this interface neither
    creates nor can remove -- see test_total_force_is_pyscf_s_residual.
    """
    from pydft_qmmm.interfaces.pyscf_pbc import pbc_forces
    from pydft_qmmm.interfaces.pyscf.pyscf_backend import load_backend
    from pydft_qmmm.utils import BOHR_PER_ANGSTROM, KJMOL_PER_EH
    interface, _ = pyscf_pbc_embedded_interface
    state = interface._scf_state()
    backend = load_backend(interface.device)
    natoms = len(interface.system.positions)
    scale = KJMOL_PER_EH * BOHR_PER_ANGSTROM
    embedding = (
        pbc_forces.pulay_forces(backend, state, natoms)
        + pbc_forces.nuclear_forces(state, interface.system.box, natoms)
        + pbc_forces.mm_forces(
            backend, state, interface.system,
            interface.embedding_sigma, natoms,
        )
    ) * scale
    np.testing.assert_allclose(
        embedding.sum(axis=0), np.zeros(3), atol=0.5,
    )


def test_total_force_is_pyscf_s_residual(pyscf_pbc_embedded_interface):
    """The reported net force is the bare periodic gradient's.

    PySCF's gamma-point periodic gradient is not translationally
    invariant on an under-resolved FFT grid.  On a bare unembedded
    water cell the residual is 203 kJ/mol/A at ke_cutoff=80 and 0.7 at
    200 for gth-dzvp, so it is a grid convergence property rather than
    anything the embedding introduces.  At the fixture's converged
    cutoff what is left must be small.
    """
    interface, _ = pyscf_pbc_embedded_interface
    total = interface.compute_forces().sum(axis=0)
    np.testing.assert_allclose(total, np.zeros(3), atol=5.0)


def test_cpu_and_gpu_agree(pyscf_pbc_embedded_factory):
    """Both backends must produce the same physics, not merely run.

    Tolerances come from the Task 1 probe, which measured gpu4pyscf
    against PySCF on a bare single-k-point KRKS cell: 8.8e-5 Eh
    (0.23 kJ/mol) on the energy and 1.6e-5 Eh/a0 (0.08 kJ/mol/A) on the
    gradient.  That gap is gpu4pyscf's grid and screening thresholds
    rather than roundoff, so it does not shrink with conv_tol.
    """
    # Not importorskip: modern pytest re-raises an ImportError that is
    # not ModuleNotFoundError, and on a CPU node gpu4pyscf fails with
    # "libcusolver.so.11: cannot open shared object file" -- the package
    # is present, the CUDA runtime is not.  That is a skip, not a
    # failure; this test belongs to the gpu-v100 job.
    try:
        import gpu4pyscf                                     # noqa: F401
    except ImportError as error:
        pytest.skip(f"GPU4PySCF unavailable on this node: {error}")
    cpu = pyscf_pbc_embedded_factory("cpu")
    gpu = pyscf_pbc_embedded_factory("gpu")
    assert gpu.compute_energy() == pytest.approx(
        cpu.compute_energy(), abs=2.0,
    )
    np.testing.assert_allclose(
        gpu.compute_forces(), cpu.compute_forces(), rtol=1e-2, atol=1.0,
    )


def test_reciprocal_reaction_on_subsystem_iii_is_exact(
        pyscf_pbc_pme_interface,
):
    """Moving a subsystem III atom must reproduce its analytic force.

    This is the reciprocal half of the MM reaction channel, which the
    non-PME fixtures never exercise because they have no subsystem III.
    It must source helPME from the WHOLE QM charge distribution --
    electrons on the grid and valence nuclei as points -- and write only
    to MM rows: compute_source_forces returns
    ``-grad(phi) * system.charges`` for every atom, and the QM atoms
    still carry force-field charges that conservative coupling zeroes
    inside OpenMM without touching System.charges.

    Sourcing from the electrons alone, or letting the QM rows through,
    both break this.
    """
    interface = pyscf_pbc_pme_interface
    assert interface.potentials, "fixture must register a PME potential"
    system = interface.system
    atom = sorted(system.select("subsystem III"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 2e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-3)


@pytest.mark.xfail(
    reason=(
        "KNOWN GAP, not yet root-caused.  With PME active the reported "
        "QM force is not the derivative of the reported energy: measured "
        "11.97 kJ/mol/A on a QM atom at the full-calculator level, with a "
        "residual net force of about 10.  The MM side is exact (see "
        "test_reciprocal_reaction_on_subsystem_iii_is_exact) and the "
        "non-PME path is conservative to well under 1, so the missing "
        "term is on the QM side and involves the QM atoms' own "
        "force-field charges entering V_rec: PMEExcludedPotential "
        "supplies that correction for a MOLECULAR QM region, and whether "
        "it is correct for a PERIODIC one has not been established."
    ),
    strict=True,
)
def test_pme_qm_force_matches_central_differences(pyscf_pbc_pme_interface):
    """The QM force must differentiate the energy when PME is active."""
    interface = pyscf_pbc_pme_interface
    atom = sorted(interface.system.select("subsystem I"))[0]
    analytic = interface.compute_forces()[atom]
    numeric = _central_difference(interface, atom, 2e-3)
    np.testing.assert_allclose(analytic, numeric, rtol=5e-3, atol=1.0)
