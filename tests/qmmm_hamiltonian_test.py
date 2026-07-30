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
