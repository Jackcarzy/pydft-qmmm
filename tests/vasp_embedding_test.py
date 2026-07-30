"""Interface-level tests for VASP electrostatic embedding.

The wiring tests here need no VASP binary.  The Tier 0 smoke and Tier 2
gradient tests do, and are skipped unless PYDFT_QMMM_VASP_COMMAND is set.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory


class TestInterfaceWiring:

    def test_incar_gains_plugin_tags_when_embedding(self, vasp_embedded):
        vasp_embedded._write_input()
        with open(os.path.join(vasp_embedded.directory, "INCAR")) as fh:
            incar = fh.read()
        assert "PLUGINS/LOCAL_POTENTIAL = T" in incar
        assert "PLUGINS/FORCE_AND_STRESS = T" in incar

    def test_incar_has_no_plugin_tags_without_embedding(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library,
    ):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
        )
        potential._write_input()
        with open(os.path.join(potential.directory, "INCAR")) as fh:
            incar = fh.read()
        assert "PLUGINS/" not in incar

    def test_plugin_is_copied_into_the_run_directory(self, vasp_embedded):
        vasp_embedded._write_input()
        assert os.path.isfile(
            os.path.join(vasp_embedded.directory, "vasp_plugin.py"),
        )

    def test_mm_charges_are_written(self, vasp_embedded):
        count = vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        positions, charges, _, sigma = vasp_plugin.read_mm_charges(path)
        assert count == len(charges) > 0
        assert sigma == pytest.approx(vasp_embedded.embedding_sigma)

    def test_written_charges_match_subsystem_ii(self, vasp_embedded):
        vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        positions, charges, _, _ = vasp_plugin.read_mm_charges(path)
        indices = sorted(vasp_embedded.system.select("subsystem II"))
        assert len(charges) == len(indices)
        assert positions == pytest.approx(
            np.asarray(vasp_embedded.system.positions)[indices], abs=1e-9,
        )
        assert charges == pytest.approx(
            np.asarray(vasp_embedded.system.charges)[indices], abs=1e-9,
        )

    def test_configured_sigma_reaches_the_plugin(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library,
    ):
        # Guards the process boundary: a module-level SIGMA in the
        # plugin would silently ignore the user's setting.
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            embedding_sigma=0.55,
        )
        potential._write_mm_charges()
        _, _, _, sigma = vasp_plugin.read_mm_charges(
            os.path.join(potential.directory, "MM_CHARGES"),
        )
        assert sigma == pytest.approx(0.55)

    def test_rewrite_replaces_rather_than_appends(self, vasp_embedded):
        vasp_embedded._write_mm_charges()
        first = vasp_embedded._write_mm_charges()
        path = os.path.join(vasp_embedded.directory, "MM_CHARGES")
        _, charges, _, _ = vasp_plugin.read_mm_charges(path)
        assert len(charges) == first

    def test_missing_sentinel_raises(self, vasp_embedded):
        with pytest.raises(Exception, match="sentinel|PLUGINS"):
            vasp_embedded._check_plugin_fired()

    def test_present_sentinel_passes(self, vasp_embedded):
        os.makedirs(vasp_embedded.directory, exist_ok=True)
        with open(
            os.path.join(vasp_embedded.directory, vasp_plugin.SENTINEL), "w",
        ) as fh:
            fh.write("local_potential callback executed\n")
        vasp_embedded._check_plugin_fired()


class TestFactory:

    def test_embedding_defaults_off(self, vasp_qmmm_system, tmp_path, vasp_pp_library):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
        )
        assert potential.embedding is False
        assert potential.embedding_sigma == 0.3

    def test_sigma_is_configurable(self, vasp_qmmm_system, tmp_path, vasp_pp_library):
        potential = vasp_interface_factory(
            vasp_qmmm_system,
            directory=str(tmp_path / "vasp"),
            pp_path=vasp_pp_library,
            embedding=True,
            embedding_sigma=0.5,
        )
        assert potential.embedding_sigma == 0.5

    @pytest.mark.parametrize("bad", [-1.0, 0.0])
    def test_non_positive_sigma_is_rejected(
            self, vasp_qmmm_system, tmp_path, vasp_pp_library, bad,
    ):
        with pytest.raises(ValueError, match="sigma"):
            vasp_interface_factory(
                vasp_qmmm_system,
                directory=str(tmp_path / "vasp"),
                pp_path=vasp_pp_library,
                embedding=True,
                embedding_sigma=bad,
            )
