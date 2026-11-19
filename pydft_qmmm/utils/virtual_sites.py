"""Weighted virtual sites, adapted from virtual-sites.patch for periodic slabs.

OpenMM-native sites use direct averages. Periodic hollow sites use parent
minimum images. Forces must be distributed exactly once by their owner.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ['VirtualSite', 'VirtualSiteHelper', 'extract_virtual_sites']


@dataclass(frozen=True)
class VirtualSite:
    """Two- or three-parent weighted site; indices refer to existing particles."""
    index: int
    site_type: str
    parents: tuple[int, ...]
    parent_weights: tuple[float, ...]
    periodic: bool = False


def extract_virtual_sites(system):
    """Read supported native OpenMM definitions without changing the system."""
    import openmm
    sites = []
    for index in range(system.getNumParticles()):
        if not system.isVirtualSite(index):
            continue
        site = system.getVirtualSite(index)
        parents = tuple(int(site.getParticle(i)) for i in range(site.getNumParticles()))
        if any(system.isVirtualSite(parent) for parent in parents):
            raise NotImplementedError('Nested virtual sites are not supported')
        if isinstance(site, openmm.TwoParticleAverageSite):
            kind = 'two_average'
        elif isinstance(site, openmm.ThreeParticleAverageSite):
            kind = 'three_average'
        else:
            raise TypeError(f'Unsupported virtual site: {type(site).__name__}')
        sites.append(VirtualSite(index, kind, parents,
                                 tuple(float(site.getWeight(i)) for i in range(len(parents)))))
    return tuple(sites)


class VirtualSiteHelper:
    """Construct sites and apply the weighted force chain rule.

    Accept a System, as in the supplied patch, or standalone definitions.
    No GPU import is needed for these small arrays. Metadata is immutable
    and rebuilt explicitly, avoiding stale cached parent/weight arrays.
    """

    def __init__(self, system=None, *, virtual_sites=()):
        self.system = system
        self.definitions = tuple(virtual_sites)
        self._validate(self.definitions)

    @staticmethod
    def _validate(sites):
        indices = [s.index for s in sites]
        if len(set(indices)) != len(indices) or any(i < 0 for i in indices):
            raise ValueError('Site indices must be unique and nonnegative')
        for site in sites:
            count = {'two_average': 2, 'three_average': 3}.get(site.site_type)
            if count is None:
                raise NotImplementedError(f'Unsupported site type: {site.site_type}')
            if len(site.parents) != count or len(site.parent_weights) != count:
                raise ValueError('Parent/weight count does not match site type')
            if len(set(site.parents)) != count or any(i < 0 or i in indices for i in site.parents):
                raise ValueError('Parents must be distinct real particles')
            if not np.isfinite(site.parent_weights).all() or not np.isclose(sum(site.parent_weights), 1.):
                raise ValueError('Finite weights must sum to one')

    def _sites(self):
        return self.system._virtual_sites if self.system is not None else self.definitions

    @property
    def indices(self):
        return np.array([s.index for s in self._sites()], dtype=int)

    @property
    def parents(self):
        return np.array([s.parents for s in self._sites()], dtype=int)

    def _get_batches(self):
        sites = self._sites()
        batches = []
        for kind in ('two_average', 'three_average'):
            for periodic in (False, True):
                group = [s for s in sites if s.site_type == kind and s.periodic == periodic]
                if group:
                    batches.append((np.array([s.index for s in group], dtype=int),
                                    np.array([s.parents for s in group], dtype=int),
                                    np.array([s.parent_weights for s in group]), periodic))
        return batches

    def compute_positions(self, positions, box=None):
        result = np.array(positions, dtype=float, copy=True)
        if box is None and self.system is not None:
            box = self.system.box
        for indices, parents, weights, periodic in self._get_batches():
            triangle = result[parents]
            if periodic and box is not None and np.any(box):
                delta = triangle - triangle[:, :1]
                frac = delta @ np.linalg.inv(box)
                triangle = triangle[:, :1] + (frac - np.rint(frac)) @ box
            result[indices] = np.sum(triangle * weights[:, :, None], axis=1)
        return result

    def compute_velocities(self, velocities):
        result = np.array(velocities, dtype=float, copy=True)
        for indices, parents, weights, _ in self._get_batches():
            result[indices] = np.sum(result[parents] * weights[:, :, None], axis=1)
        return result

    def distribute_forces(self, forces, *, clear_sites=False):
        """Do not apply to native OpenMM forces already transferred to parents.

        Default retains site entries, matching the supplied patch. A complete
        calculator wrapper uses clear_sites=True after its single transfer.
        """
        result = np.array(forces, dtype=float, copy=True)
        for indices, parents, weights, _ in self._get_batches():
            contributions = (result[indices, None, :] * weights[:, :, None]).reshape(-1, 3)
            np.add.at(result, parents.ravel(), contributions)
            if clear_sites:
                result[indices] = 0
        return result

    def update(self, system):
        system.positions = self.compute_positions(system.positions, system.box)
        system.velocities = self.compute_velocities(system.velocities)
