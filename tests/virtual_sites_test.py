"""Conservative dependent sites, including triangles crossing periodic edges."""
from dataclasses import dataclass
import numpy as np
import pytest
from pydft_qmmm import System,Atom,Simulation,VerletIntegrator
from pydft_qmmm.calculators import Calculator,Results
from pydft_qmmm.calculators.virtual_sites import CentroidSites,VirtualSiteCalculator


def fixture():
 system=System([Atom(position=np.array(p,dtype=float),mass=10.,element='Au') for p in [[9.8,4,4],[.2,4,4],[0,4.4,4],[0,0,0],[0,0,0]]],box=np.eye(3)*10)
 return system,CentroidSites([3,4],[[0,1,2],[0,1,2]])


@dataclass(frozen=True)
class Harmonic(Calculator):
 @property
 def name(self):return 'Harmonic'
 def calculate(self,return_forces=True,return_components=True):
  x=np.array(self.system.positions)
  forces=np.zeros_like(x)
  forces[3:]=-2*(x[3:]-[10,4,4])
  return Results(float(np.sum((x[3:]-[10,4,4])**2)),forces,{})


def test_periodic_shared_parent_finite_difference():
 system,sites=fixture();calc=VirtualSiteCalculator(system,Harmonic(system),sites)
 reference=np.array(system.positions);result=calc.calculate()
 assert np.allclose(system.positions[3:],[10,4+0.4/3,4])
 for atom in range(3):
  for axis in range(3):
   energies=[]
   for sign in (1,-1):
    xyz=reference.copy();xyz[atom,axis]+=sign*1e-5;system.positions=xyz
    energies.append(calc.calculate().energy)
   assert result.forces[atom,axis]==pytest.approx(-(energies[0]-energies[1])/2e-5,abs=1e-8)
 assert np.all(result.forces[3:]==0)
 assert np.allclose(result.forces.sum(axis=0),[0,-4*.4/3,0])


def test_dynamics_sites_and_kinetic_energy(tmp_path):
 system,sites=fixture()
 system.velocities[:3]=[.001,.002,.003]
 with Simulation(system,VerletIntegrator(.1),calculator=Harmonic(system),virtual_sites=sites,center=False,wrap=False,output_directory=str(tmp_path)) as sim:
  sim.run_dynamics(3)
  assert np.allclose(system.positions,sites.positions(system.positions,system.box))
  assert np.allclose(system.velocities[3:],np.array(system.velocities)[:3].mean(axis=0))
  v=np.array(system.velocities[:3])+.05*np.array(system.forces[:3])*1e-4/10
  assert sim.energy['.']['Kinetic Energy']==pytest.approx(.5e4*10*np.sum(v*v))


def test_translation_and_velocity():
 system,sites=fixture()
 p=sites.positions(system.positions,system.box)
 shift=np.array([2.,-3.,1.])
 assert np.allclose(sites.positions(np.array(system.positions)+shift,system.box),p+shift)
 v=np.arange(15).reshape(5,3).astype(float)
 assert np.allclose(sites.velocities(v)[3:],v[:3].mean(axis=0))


def test_invalid_mapping_and_double_wrapping():
 with pytest.raises(ValueError):CentroidSites([3],[[0,0,1]])
 with pytest.raises(ValueError):CentroidSites([3],[[0,1,3]])
 system,sites=fixture()
 calc=VirtualSiteCalculator(system,Harmonic(system),sites)
 with pytest.raises(ValueError):VirtualSiteCalculator(system,calc,sites)
 with pytest.raises(ValueError):Simulation(system,VerletIntegrator(1),calculator=Harmonic(system),virtual_sites=sites)


def test_optimization_updates_sites_before_evaluation(tmp_path):
 from pydft_qmmm import Optimization
 system,sites=fixture()
 opt=Optimization('atom 0 1 2',system,calculator=Harmonic(system),virtual_sites=sites,center=False,wrap=False,output_directory=str(tmp_path))
 system.positions[0,2]+=.3
 opt.calculate_energy_forces()
 assert np.allclose(system.positions,sites.positions(system.positions,system.box))
 assert np.allclose(system.forces[:3,2],-2*.1*2/3)
 for handler in opt.handlers:handler.close()


def test_restart_matches_uninterrupted_verlet(tmp_path):
 def advance(system,sites,steps,name):
  with Simulation(system,VerletIntegrator(.1),calculator=Harmonic(system),virtual_sites=sites,center=False,wrap=False,output_directory=str(tmp_path/name)) as simulation:
   simulation.run_dynamics(steps)
 full,sites=fixture();split,_=fixture()
 advance(full,sites,6,'full');advance(split,sites,3,'first')
 resumed,_=fixture();resumed.positions=np.array(split.positions);resumed.velocities=np.array(split.velocities)
 advance(resumed,sites,3,'second')
 assert np.allclose(full.positions,resumed.positions,atol=1e-14,rtol=0)
 assert np.allclose(full.velocities,resumed.velocities,atol=1e-14,rtol=0)


def test_patch_api_mixed_sites_and_position_notifier():
 from pydft_qmmm.utils import VirtualSite
 system,_=fixture()
 system.build_virtual_sites([
  VirtualSite(3,'two_average',(0,1),(.25,.75),periodic=True),
  VirtualSite(4,'three_average',(0,1,2),(1/3,1/3,1/3),periodic=True)])
 assert np.allclose(system.positions[3],[10.1,4,4])
 system.positions[0,2]+=.4
 assert system.positions[3,2]==pytest.approx(4.1)
 assert system.positions[4,2]==pytest.approx(4+.4/3)
 assert np.all(system.masses[3:]==0)
 forces=np.zeros((5,3));forces[3:,2]=[4,3]
 distributed=system.vsite_helper.distribute_forces(forces,clear_sites=True)
 assert np.allclose(distributed[:3,2],[2,4,1])
 assert np.all(distributed[3:]==0)


def test_patch_zero_mass_integrators():
 from pydft_qmmm import LangevinIntegrator
 system,sites=fixture();system.build_virtual_sites(sites.definitions)
 system.forces[:]=1
 for integrator in [VerletIntegrator(1),LangevinIntegrator(1,300,.005)]:
  xyz,velocity=integrator.integrate(system)
  assert np.isfinite(xyz).all() and np.isfinite(velocity).all()
  assert np.all(velocity[sites.indices]==0)
  assert np.isfinite(integrator.compute_kinetic_energy(system))


def test_patch_helper_matches_native_openmm():
 import openmm as mm
 from openmm import unit
 from pydft_qmmm.utils import extract_virtual_sites,VirtualSiteHelper
 model=mm.System()
 for mass in [10,10,10,0]:model.addParticle(mass)
 model.setVirtualSite(3,mm.ThreeParticleAverageSite(0,1,2,.2,.3,.5))
 force=mm.CustomExternalForce('x*x+y*y+z*z');force.addParticle(3,[]);model.addForce(force)
 integrator=mm.VerletIntegrator(.001)
 context=mm.Context(model,integrator,mm.Platform.getPlatformByName('Reference'))
 xyz=np.array([[1.,0,0],[0,1,0],[0,0,1],[0,0,0]])
 context.setPositions(xyz);context.computeVirtualSites()
 state=context.getState(getPositions=True,getForces=True)
 helper=VirtualSiteHelper(virtual_sites=extract_virtual_sites(model))
 expected=helper.compute_positions(xyz)
 assert np.allclose(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),expected)
 raw=np.zeros_like(xyz);raw[3]=-2*expected[3]
 native=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
 assert np.allclose(helper.distribute_forces(raw)[:3],native[:3])
 del context,integrator
