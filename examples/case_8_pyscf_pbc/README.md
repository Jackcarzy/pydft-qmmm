Example Case 8 (periodic PySCF)
===============================

Summary
-------
This case serves as an example of using a **periodic** PySCF
wavefunction as the QM engine, through the ``pyscf_pbc`` interface.
Unlike :example:`case 5 <5_pyscf>`, the QM region is not a molecule in a
periodic environment: the wavefunction itself is periodic, and the QM
cell is the simulation box.  This is the same physical model the VASP
interface implements, run in-process with Gaussian basis functions
rather than through a separate binary and file exchange.

The system is one QM water in a small box of MM water modelled by the
SPC/E forcefield.  Waters whose centroid lies within 4.0 Angstroms of
the QM region form subsystem II and reach the QM electrons as a
Gaussian-smeared near field on the solver's own FFT grid; everything
beyond reaches them through the particle-mesh Ewald reciprocal sum.  A
single energy and set of forces is computed rather than a trajectory.

Running this case requires PySCF, which is installed with the ``pyscf``
extra:

```bash
python -m pip install 'pydft_qmmm[pyscf,qmmm-pme]'
```

How to Run
----------
```bash
python case_8_pbc.py
```

Why this case ships its own box
-------------------------------
Cases 0 and 5 share a 29.9 Angstrom box of SPC/E water.  This case uses
a 12 Angstrom box of 27 waters instead, because **the QM cell is the
box**: the box size sets the size of the QM calculation, not just the
size of the MM environment.  At a force-converged cutoff the 29.9
Angstrom box needs an FFT mesh several hundred points on a side.  The
VASP interface has exactly the same constraint for exactly the same
reason.

Choosing ``ke_cutoff``
----------------------
``ke_cutoff`` must be converged against a **force**, not an energy.  The
energy settles at a cutoff where the gradient is still badly
under-resolved.  Measured on a bare, unembedded water cell, the net
force on the whole cell -- which translational invariance requires to
vanish -- is:

| basis | ``ke_cutoff=80`` | ``ke_cutoff=200`` |
|---|---|---|
| gth-szv | 202.6 | -5.25 |
| gth-dzvp | 92.6 | -0.748 |
| gth-tzvp | 104.1 | -0.128 |

in kJ/mol/Angstrom.  This case uses 200 with ``gth-dzvp`` for that
reason.  A residual of this kind belongs to PySCF's own periodic
gradient rather than to the QM/MM embedding: the embedding's own force
channels cancel to well under 1 kJ/mol/Angstrom.

Other differences from the molecular interface
----------------------------------------------
- ``pseudo`` is **mandatory** and ``ecp`` is rejected.  The periodic
  gradient code refuses all-electron cells, so a GTH pseudopotential and
  a matching GTH basis are required.
- ``ke_cutoff`` or ``mesh`` replaces ``grid_level``.  The embedding is
  integrated on the solver's uniform FFT grid, which is legitimate here
  only because the pseudopotential removes the nuclear cusps; the
  molecular interface needs an atom-centred grid for the opposite
  reason.
- Brillouin-zone sampling is a single k-point.  There are no k-meshes,
  and no stress tensor, in this release.

Running on a GPU
----------------
Uncomment ``device="gpu"`` in the script.  That needs GPU4PySCF and a
CUDA runtime; the periodic gradients it produces agree with PySCF's to
about 1e-5 Eh/a0, which is the backends' own reproducibility rather than
roundoff.
