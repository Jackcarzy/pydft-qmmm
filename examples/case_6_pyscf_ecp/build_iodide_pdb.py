"""Write the iodide-in-SPC/E-water PDB used by case_6_ecp.py.

The first water of ``spce.pdb`` is replaced by a single iodide ion at
its oxygen position, giving a solvated heavy-atom QM region whose def2
basis carries an effective core potential.
"""
from __future__ import annotations

import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "spce.pdb"
TARGET = HERE / "iodide_spce.pdb"


def main() -> None:
    out = []
    serial = 0
    dropped = 0
    for line in SOURCE.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            out.append(line)
            continue
        name = line[12:16].strip()
        if line[22:26].strip() == "1":
            if name != "O":
                # The two hydrogens make way for the monatomic ion.
                dropped += 1
                continue
            serial += 1
            out.append(
                f"HETATM{serial:5d}  I   IOD A   1    "
                + line[30:54]
                + "  1.00  0.00           I",
            )
            continue
        serial += 1
        out.append(f"HETATM{serial:5d}" + line[11:])
    TARGET.write_text("\n".join(out) + "\n")
    print(f"wrote {TARGET.name} ({serial} atoms, dropped {dropped})")


if __name__ == "__main__":
    main()
