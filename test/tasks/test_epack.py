"""Tests for epack.py — outfile catalog gating and validation."""

import pytest

from pyfm.domain import MassDict, Outfile
from pyfm.tasks.hadrons.epack import (
    EpackConfig,
    build_input_params,
    create_outfile_catalog,
    validate_config,
)
from pyfm.tasks.hadrons.types import LanczosParams

MASS = MassDict.from_dict({"l": 0.002426})
LANCZOS = LanczosParams(alpha=0.009, beta=24, npoly=11, nstop=10, nk=12, nm=16)


def make_epack(**overrides):
    kwargs = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MASS,
        eigs=4,
        eig=Outfile(filestem="eig/eig", ext=".bin", good_size=1),
        eigdir=Outfile(filestem="eig/eig_{eig_index}", ext=".bin", good_size=1),
        eval=Outfile(filestem="eval/eval", ext=".h5", good_size=1),
    )
    kwargs.update(overrides)
    return EpackConfig(**kwargs)


def catalog_paths(config):
    df = create_outfile_catalog(config)
    return [] if df.empty else sorted(df["filepath"])


class TestCatalogGating:
    def test_defaults_catalog_eval_only(self):
        # save_eigs=False, save_evals=True: the eval file is written, so it
        # must be catalogued (was gated on save_eigs).
        paths = catalog_paths(make_epack())
        assert len(paths) == 1
        assert "eval/eval" in paths[0]

    def test_save_evals_false_omits_eval(self):
        assert catalog_paths(make_epack(save_evals=False)) == []

    def test_save_eigs_single_file(self):
        paths = catalog_paths(make_epack(save_eigs=True, save_evals=False))
        assert len(paths) == 1
        assert "eig/eig" in paths[0]

    def test_save_eigs_multifile_enumerates_indices(self):
        paths = catalog_paths(
            make_epack(save_eigs=True, save_evals=False, multifile=True)
        )
        assert len(paths) == 4

    def test_eval_catalog_matches_eval_save_module(self):
        for save_evals in (True, False):
            config = make_epack(save_evals=save_evals)
            emitted = "eval_save" in build_input_params(config).modules
            catalogued = any("eval/eval" in p for p in catalog_paths(config))
            assert emitted == catalogued


class TestValidate:
    def test_load_needs_nothing(self):
        validate_config(make_epack(load=True))

    def test_generate_requires_lanczos(self):
        with pytest.raises(ValueError, match="Lanczos"):
            validate_config(make_epack(load=False, action_name="stag_mass_{mass}"))

    def test_generate_requires_action_name(self):
        with pytest.raises(ValueError, match="action_name"):
            validate_config(make_epack(load=False, lanczos=LANCZOS))

    def test_generate_complete_passes(self):
        validate_config(
            make_epack(load=False, lanczos=LANCZOS, action_name="stag_mass_{mass}")
        )
