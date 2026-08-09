"""Single-point Psi4 PBE/def2-QZVP gradient on QM water at frame-0 geometry.

For mechanical embedding the QM forces depend only on the QM region's
own geometry (no point charges, no MM coupling at the QM level), so a
plain Psi4 single point at the initial water geometry reproduces the
forces PyDFT-QMMM passed to the integrator at frame 0.

Compares against tight SPARC forces extracted from
``../case_3_sparc2/sparc_workdir/SPARC.static_01`` (h=0.20 Bohr,
tol_scf=1e-6, kpts=2x2x2).
"""
from __future__ import annotations

import numpy as np
import psi4

BOHR_TO_ANG = 0.529177210903
HA_BOHR_TO_KJ_MOL_NM = 4961.475093  # Ha/Bohr -> kJ/(mol*nm)

# QM water geometry (Angstroms, from water4.pdb, atoms 0-2: O, H1, H2).
geom_ang = np.array([
    [6.000, 6.000, 6.000],   # O
    [6.586, 6.760, 6.000],   # H1
    [6.586, 5.240, 6.000],   # H2
])

mol_str = "0 1\n"
labels = ["O", "H", "H"]
for lbl, xyz in zip(labels, geom_ang):
    mol_str += f"{lbl} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n"
mol_str += "symmetry c1\nno_reorient\nno_com\n"

psi4.core.be_quiet()
psi4.set_memory("2 GB")
psi4.set_options({
    "basis": "def2-QZVP",
    "scf_type": "df",
    "dft_spherical_points": 302,
    "dft_radial_points": 75,
})

mol = psi4.geometry(mol_str)
e, wfn = psi4.gradient("PBE", molecule=mol, return_wfn=True)
grad = np.asarray(wfn.gradient())   # Ha/Bohr, same atom order as input
forces_psi4 = -grad                 # F = -dE/dr

# Tight SPARC forces from ../case_3_sparc2/sparc_workdir/SPARC.static_01.
# SPARC orders atoms by element type: H, H, O.  Reorder to O, H1, H2 to
# match the Psi4/PDB order.
forces_sparc_HHO = np.array([
    [ 4.7761379667e-03,  3.5085583573e-03, -1.1653141718e-09],
    [ 4.7761396170e-03, -3.5085563692e-03, -1.5398233942e-10],
    [-9.5522775837e-03, -1.9881544776e-09,  1.3192965112e-09],
])
forces_sparc = forces_sparc_HHO[[2, 0, 1]]   # -> O, H1, H2

print("Forces in Ha/Bohr at frame-0 geometry (atom order: O, H1, H2)\n")
print(f"{'atom':<6}{'engine':<8}{'Fx':>14}{'Fy':>14}{'Fz':>14}{'|F|':>14}")
for i, lbl in enumerate(labels):
    f_p = forces_psi4[i]
    f_s = forces_sparc[i]
    print(f"{lbl:<6}{'Psi4':<8}{f_p[0]:14.6e}{f_p[1]:14.6e}{f_p[2]:14.6e}"
          f"{np.linalg.norm(f_p):14.6e}")
    print(f"{'':<6}{'SPARC':<8}{f_s[0]:14.6e}{f_s[1]:14.6e}{f_s[2]:14.6e}"
          f"{np.linalg.norm(f_s):14.6e}")
    diff = f_p - f_s
    print(f"{'':<6}{'diff':<8}{diff[0]:14.6e}{diff[1]:14.6e}{diff[2]:14.6e}"
          f"{np.linalg.norm(diff):14.6e}")
    print()

rms_psi4  = np.sqrt(np.mean(forces_psi4**2))
rms_sparc = np.sqrt(np.mean(forces_sparc**2))
rms_diff  = np.sqrt(np.mean((forces_psi4 - forces_sparc)**2))
max_diff  = np.max(np.abs(forces_psi4 - forces_sparc))

print(f"RMS |F| Psi4  : {rms_psi4:.4e} Ha/Bohr "
      f"= {rms_psi4*HA_BOHR_TO_KJ_MOL_NM:.2f} kJ/mol/nm")
print(f"RMS |F| SPARC : {rms_sparc:.4e} Ha/Bohr "
      f"= {rms_sparc*HA_BOHR_TO_KJ_MOL_NM:.2f} kJ/mol/nm")
print(f"RMS diff      : {rms_diff:.4e} Ha/Bohr "
      f"= {rms_diff*HA_BOHR_TO_KJ_MOL_NM:.2f} kJ/mol/nm")
print(f"Max |diff|    : {max_diff:.4e} Ha/Bohr "
      f"= {max_diff*HA_BOHR_TO_KJ_MOL_NM:.2f} kJ/mol/nm")
print(f"Relative RMS  : {rms_diff/rms_sparc*100:.2f} %")
