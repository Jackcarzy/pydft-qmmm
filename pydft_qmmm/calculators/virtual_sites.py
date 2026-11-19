"""Dependent three-parent sites and their conservative force transformation."""
from dataclasses import dataclass

import numpy as np

from .calculator import Calculator, Results
from pydft_qmmm.utils import pluggable_method


from pydft_qmmm.utils.virtual_sites import VirtualSite, VirtualSiteHelper


class CentroidSites(VirtualSiteHelper):
    """Convenience mapping for periodic Au hollow sites, using the shared helper."""

    def __init__(self, indices, parents):
        if len(indices) != len(parents):
            raise ValueError('Expected one parent triplet per site')
        definitions = [VirtualSite(int(i), 'three_average', tuple(map(int, p)),
                                   (1/3, 1/3, 1/3), periodic=True)
                       for i, p in zip(indices, parents)]
        super().__init__(virtual_sites=definitions)

    positions = VirtualSiteHelper.compute_positions
    velocities = VirtualSiteHelper.compute_velocities

    def forces(self, forces):
        return self.distribute_forces(forces, clear_sites=True)


@dataclass(frozen=True)
class VirtualSiteCalculator(Calculator):
    """Wrap the complete calculator once, retaining all site energy terms."""
    calculator: Calculator
    sites: CentroidSites

    def __post_init__(self):
        if isinstance(self.calculator, VirtualSiteCalculator):
            raise ValueError('Virtual-site forces must be redistributed only once')
        if self.calculator.system is not self.system:
            raise ValueError('Calculator and sites must use the same system')

    @property
    def name(self):
        return self.calculator.name

    @pluggable_method
    def calculate(self, return_forces=True, return_components=True):
        self.sites.update(self.system)
        result = self.calculator.calculate(return_forces, return_components)
        return Results(result.energy,
                       self.sites.distribute_forces(result.forces, clear_sites=True) if return_forces else result.forces,
                       result.components)

    def calculation_run(self, evaluations):
        return self.calculator.calculation_run(evaluations)

    def close(self):
        self.calculator.close()
