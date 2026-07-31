"""Unit tests for independent QM/MM Hamiltonian configuration."""
from __future__ import annotations

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.utils import Subsystem
from pydft_qmmm.utils import TheoryLevel


class TestIndependentConfiguration:

    def test_force_matrices_are_not_shared(self):
        mechanical = QMMMHamiltonian(
            "mechanical", "mechanical", partition=None,
        )
        electrostatic = QMMMHamiltonian(
            "electrostatic", "cutoff", partition=None,
        )

        assert (
            mechanical.force_matrix[Subsystem.I][Subsystem.II]
            == TheoryLevel.MM
        )
        assert (
            mechanical.force_matrix[Subsystem.I][Subsystem.III]
            == TheoryLevel.MM
        )
        assert (
            electrostatic.force_matrix[Subsystem.I][Subsystem.II]
            == TheoryLevel.QM
        )
        assert (
            electrostatic.force_matrix[Subsystem.I][Subsystem.III]
            == TheoryLevel.NO
        )

    def test_default_partitions_are_not_shared(self):
        first = QMMMHamiltonian()
        second = QMMMHamiltonian()

        assert first.partition is not second.partition
        first.partition.cutoff = 9.0
        assert second.partition.cutoff == 14.0

    def test_none_still_disables_partitioning(self):
        coupling = QMMMHamiltonian(partition=None)

        assert coupling.partition is None


class TestDirectPMEForcePartition:
    """The I-III force levels of direct QM/MM/PME.

    John et al., J. Chem. Phys. 161, 034103 (2024), Table I and Eq. 10
    classify QM/MM methods by two independent choices: X, the level of
    theory for the force on subsystem I from the I-III interaction, and
    Y, the level for the force on subsystem III from that same
    interaction.  Direct QM/MM/PME is asymmetric -- X = QM, Y = MM --
    and that asymmetry is deliberate, not an approximation to be fixed
    later.  Y = QM would be QM/MM/SC-PME, the empty row of Table I.
    """

    def test_direct_pme_uses_qm_force_on_i_and_mm_force_on_iii(self):
        coupling = QMMMHamiltonian(
            "electrostatic", "electrostatic", partition=None,
        )

        assert (
            coupling.force_matrix[Subsystem.I][Subsystem.III]
            == TheoryLevel.QM
        )
        assert (
            coupling.force_matrix[Subsystem.III][Subsystem.I]
            == TheoryLevel.MM
        )

    def test_close_range_i_ii_is_qm_in_both_directions(self):
        # Unlike I-III, the I-II interaction is evaluated at the QM level
        # from both sides, which is why Newton's third law is required to
        # hold there and only there.
        coupling = QMMMHamiltonian(
            "electrostatic", "electrostatic", partition=None,
        )

        assert (
            coupling.force_matrix[Subsystem.I][Subsystem.II]
            == TheoryLevel.QM
        )
        assert (
            coupling.force_matrix[Subsystem.II][Subsystem.I]
            == TheoryLevel.QM
        )
