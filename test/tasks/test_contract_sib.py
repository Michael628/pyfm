"""Tests for the contract_sib task module (registration + validation).

Mirrors test_sib_mf.py's registration/validation layers; the golden
end-to-end input-generation test lives with the fixture (Slice 3,
test_generate_input.py's contract section).
"""
import importlib

import numpy as np
import pytest
import yaml

from pyfm.domain import task_registry, build_hooks
from pyfm.tasks.register import _config_to_task_key


@pytest.fixture()
def reset_registries():
    saved_handlers = dict(task_registry._handlers)
    saved_hooks = dict(build_hooks._registry)
    saved_config_to_task_key = dict(_config_to_task_key)
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    yield
    task_registry.clear()
    build_hooks.clear()
    _config_to_task_key.clear()
    task_registry._handlers.update(saved_handlers)
    build_hooks._registry.update(saved_hooks)
    _config_to_task_key.update(saved_config_to_task_key)


def _reload_sib_modules():
    import pyfm.tasks.contract.sib as _s

    importlib.reload(_s)
    return _s


@pytest.mark.usefixtures("reset_registries")
class TestRegistration:
    def test_contract_sib_registered_via_nanny_keys(self):
        _reload_sib_modules()
        handler = task_registry.get("nanny_contract_sib")
        assert handler is not None
        assert handler.config_type.__name__ == "SIBContractConfig"

    def test_diagram_registered_as_subconfig(self):
        _reload_sib_modules()
        handler = task_registry.get("nanny_contract_sib_diagram")
        assert handler is not None
        assert handler.config_type.__name__ == "SIBDiagramConfig"
        # Sub-config-only registration: the handler carries no task
        # callables (the hadrons_sib_mf_batch precedent — hooks only).
        with pytest.raises(AttributeError, match="build_input_params"):
            _ = handler.build_input_params

    def test_reverse_map_keys_new_classes(self):
        _reload_sib_modules()
        from pyfm.a2a.types import SIBContractConfig

        assert _config_to_task_key[SIBContractConfig] == "contract_sib"

    def test_existing_contract_keys_untouched(self):
        import pyfm.tasks.contract.mesonloader as _m
        import pyfm.tasks.contract.diagram as _d
        import pyfm.tasks.contract.contraction as _c

        importlib.reload(_m)
        importlib.reload(_d)
        importlib.reload(_c)
        _reload_sib_modules()
        for key in ("nanny_contract", "nanny_contract_diagram",
                    "nanny_contract_mesonloader"):
            assert task_registry.get(key) is not None
        from pyfm.a2a.types import ContractConfig, DiagramConfig

        assert _config_to_task_key[ContractConfig] == "contract"
        assert _config_to_task_key[DiagramConfig] == "contract_diagram"


@pytest.mark.usefixtures("reset_registries")
class TestNpoint:
    def test_sib_npoint_is_three(self):
        from pyfm.a2a.types import ContractType

        assert ContractType.SIB.npoint == 3
        assert ContractType.TWOPOINT.npoint == 2


def _diagram_kwargs(**overrides):
    from pyfm.a2a.types import ContractType
    from pyfm.domain import MassDict, OpList, Outfile

    base = dict(
        formatting={},
        logging_level="INFO",
        runid="t",
        contraction_type=ContractType.SIB,
        operations=OpList.from_dict({"gamma": ["vec_local"], "mass": ["l"]}),
        mass=MassDict.from_dict({"l": 0.01, "u": 0.02}),
        blocks=Outfile(
            filestem="sib/{leg_pair}{mass}_a{n_index}{hp_index}",
            ext=".20/{gamma}_0_0_0.h5",
            good_size=1,
        ),
        tab=Outfile(
            filestem="sib/tab_a{n_index}", ext=".20/G1_G1_0_0_0.h5", good_size=1
        ),
        evalfile=Outfile(filestem="eval", ext=".20.h5", good_size=1),
        outfile=Outfile(
            filestem="corr/m{mass}/{gamma}/c3_{gamma}_m{mass}_{series}",
            ext=".20.h5",
            good_size=1,
        ),
        noise=2,
    )
    if "operations" in overrides:
        overrides["operations"] = OpList.from_dict(overrides["operations"])
    if "contraction_type" in overrides and isinstance(
        overrides["contraction_type"], str
    ):
        overrides["contraction_type"] = ContractType[overrides["contraction_type"]]
    return base | overrides


def _build_diagram(**overrides):
    from pyfm.a2a.types import SIBDiagramConfig

    return SIBDiagramConfig(**_diagram_kwargs(**overrides))


@pytest.mark.usefixtures("reset_registries")
class TestDiagramValidation:
    def test_valid_diagram_passes(self):
        _reload_sib_modules()
        from pyfm.tasks.contract.sib import validate_sib_diagram

        validate_sib_diagram(_build_diagram())

    @pytest.mark.parametrize(
        "field,value,match",
        [
            ("contraction_type", "TWOPOINT", "only SIB"),
            ("operations", {"gamma": ["pion_local"], "mass": ["l"]}, "SIB family"),
            ("operations", {"gamma": ["vec_local"], "mass": []}, "no mass"),
            ("noise", 1, "noise >= 2"),
        ],
    )
    def test_rejections(self, field, value, match):
        _reload_sib_modules()
        from pyfm.tasks.contract.sib import validate_sib_diagram

        with pytest.raises(ValueError, match=match):
            validate_sib_diagram(_build_diagram(**{field: value}))

    def test_unknown_mass_label_rejected(self):
        _reload_sib_modules()
        from pyfm.tasks.contract.sib import validate_sib_diagram

        with pytest.raises(ValueError, match="mass label"):
            validate_sib_diagram(
                _build_diagram(operations={"gamma": ["vec_local"], "mass": ["x"]})
            )

    def test_unknown_defl_mass_rejected(self):
        _reload_sib_modules()
        from pyfm.tasks.contract.sib import validate_sib_diagram

        with pytest.raises(ValueError, match="mass label"):
            validate_sib_diagram(_build_diagram(defl_mass="x"))

    def test_shared_mode_blocks_rejected(self):
        _reload_sib_modules()
        from pyfm.domain import Outfile
        from pyfm.tasks.contract.sib import validate_sib_diagram

        shared = Outfile(
            filestem="sib/{leg_pair}{mass}_a",  # no world tokens
            ext=".20/{gamma}_0_0_0.h5",
            good_size=1,
        )
        with pytest.raises(ValueError, match="split-noise"):
            validate_sib_diagram(_build_diagram(blocks=shared))

    def test_tab_without_world_token_rejected(self):
        _reload_sib_modules()
        from pyfm.domain import Outfile
        from pyfm.tasks.contract.sib import validate_sib_diagram

        tab = Outfile(
            filestem="sib/tab_a", ext=".20/G1_G1_0_0_0.h5", good_size=1
        )
        with pytest.raises(ValueError, match=r"\{n_index\}"):
            validate_sib_diagram(_build_diagram(tab=tab))

    def test_outfile_missing_tokens_rejected(self):
        _reload_sib_modules()
        from pyfm.domain import Outfile
        from pyfm.tasks.contract.sib import validate_sib_diagram

        bad = Outfile(filestem="corr/c3_{series}", ext=".20.h5", good_size=1)
        with pytest.raises(ValueError, match=r"\{mass\}"):
            validate_sib_diagram(_build_diagram(outfile=bad))

    def test_blocks_missing_gamma_ext_rejected(self):
        _reload_sib_modules()
        from pyfm.domain import Outfile
        from pyfm.tasks.contract.sib import validate_sib_diagram

        bad = Outfile(
            filestem="sib/{leg_pair}{mass}_a{n_index}{hp_index}",
            ext=".20.h5",
            good_size=1,
        )
        with pytest.raises(ValueError, match=r"\{gamma\}"):
            validate_sib_diagram(_build_diagram(blocks=bad))


@pytest.mark.usefixtures("reset_registries")
class TestCompositeValidation:
    def test_empty_diagrams_rejected(self):
        _reload_sib_modules()
        from pyfm.a2a.types import SIBContractConfig
        from pyfm.tasks.contract.sib import validate_config

        with pytest.raises(ValueError, match="must not be empty"):
            validate_config(
                SIBContractConfig(
                    formatting={}, logging_level="INFO", runid="t",
                    diagrams={}, time=4,
                )
            )



class TestGoldenInput:
    def test_generate_contract_sib_input(
        self, tmp_path, monkeypatch, tasks_data_dir, contract_params
    ):
        """End-to-end input generation against the golden YAML (the
        test_generate_contract_input sibling). Runs WITHOUT the registry
        reset: the production import-time wiring (including JobConfig's
        normalize_resources hook) must be intact for write_input_file."""
        from pyfm.nanny import write_input_file

        monkeypatch.chdir(tmp_path)
        write_input_file("contract_sib", contract_params, "a", "20")

        actual = yaml.safe_load(
            (tmp_path / "in" / "contract-sib-a.20.yaml").read_text()
        )
        expected = yaml.safe_load(
            (tasks_data_dir / "in" / "test-contract-sib-a.20.yaml").read_text()
        )
        assert actual == expected, "contract_sib YAML output mismatch"


@pytest.mark.usefixtures("reset_registries")
class TestTermNormalize:
    def _frame(self, term_in_index: bool):
        import pandas as pd

        rng = np.random.default_rng(5)
        base = pd.DataFrame(
            {
                "corr": [complex(r, r) for r in rng.normal(size=8)],
                "term": ["lll", "nll", "lnl", "lln", "nnl", "nln", "lnn", "nnn"],
                "t1": [0] * 8,
                "t2": [0] * 8,
                "t3": [0] * 8,
            }
        )
        if term_in_index:
            base = base.set_index("term", append=True)
        return base

    def test_factors_applied_index_and_column(self):
        from pyfm.dataio.processor import term_normalize

        factors = {
            "lll": 1.0,
            "nll": 0.5,
            "lnl": 0.5,
            "lln": 0.5,
            "nnl": 0.25,
            "nln": 0.25,
            "lnn": 0.25,
            "nnn": 0.0,
        }
        for in_index in (True, False):
            df = self._frame(in_index)
            out = term_normalize(df, "corr", **factors)
            raw = df.reset_index("term") if in_index else df
            got = out.reset_index("term") if in_index else out
            for _, row in got.iterrows():
                assert row["corr"] == pytest.approx(
                    raw.loc[raw["term"] == row["term"], "corr"].iloc[0]
                    * factors[row["term"]]
                )

    def test_missing_factor_raises_loudly(self):
        from pyfm.dataio.processor import term_normalize

        df = self._frame(False)
        with pytest.raises(ValueError, match="missing for term labels"):
            term_normalize(df, "corr", lll=1.0)

    def test_missing_term_axis_raises(self):
        import pandas as pd

        from pyfm.dataio.processor import term_normalize

        df = pd.DataFrame({"corr": [1.0]})
        with pytest.raises(ValueError, match="no 'term' level"):
            term_normalize(df, "corr", lll=1.0)

    def test_normalize_then_sum_composes(self):
        """The aggregation pipeline: term_normalize -> sum reproduces the
        weighted C3 for every world-factor scheme."""
        from pyfm.dataio.processor import execute as proc_execute
        from pyfm.tasks.contract.sib import _sib_term_factors

        factors = _sib_term_factors(2)
        df = self._frame(False)
        raw = df.set_index(["term"])["corr"]
        out = proc_execute(
            df.set_index(["term"]),
            {"term_normalize": dict(factors), "term_sum": ["term"]},
        )
        expected = sum(raw[t] * factors[t] for t in factors)
        assert out["corr"].iloc[0] == pytest.approx(expected)

    def test_sum_action_mean_semantics_not_used_for_terms(self):
        """The legacy `sum` action means over the group; C3 needs a true
        sum — term_sum provides it and the aggregator uses it."""
        import numpy as np
        import pandas as pd

        from pyfm.dataio.processor import term_sum

        df = pd.DataFrame(
            {
                "corr": np.ones(4, dtype=complex),
                "term": list("abcd"),
                "gamma": ["GX_GX"] * 4,
            }
        ).set_index(["term"])
        out = term_sum(df, "corr", "term")
        assert out["corr"].iloc[0] == pytest.approx(4.0)

    def test_action_order_registered_before_sum(self):
        from pyfm.dataio.processor import ACTION_ORDER

        assert "term_normalize" in ACTION_ORDER
        assert ACTION_ORDER.index("term_normalize") < ACTION_ORDER.index(
            "term_sum"
        ) < ACTION_ORDER.index("sum")

    def test_aggregation_load_and_c3(self, tmp_path):
        """End-to-end aggregation: synthetic per-term files in the outfile
        grammar load through the emitted load_files (wildcard-recovered
        gamma) and the actions produce the weighted C3 — the load-level
        test the slice-verifier demanded."""
        import glob
        import os

        import numpy as np
        import pandas as pd

        from pyfm.a2a.types import SIBContractConfig, SIB_TERM_LABELS
        from pyfm.dataio import data_to_frame, write_files
        from pyfm.domain import LoadDictConfig, Outfile
        from pyfm.nanny.aggregator import load_data, process_data
        from pyfm.tasks.contract.sib import (
            _sib_term_factors,
            build_aggregator_params,
        )

        nt = 2
        # The stem must contain "correlators/" — get_processed_filename
        # reroutes writes to processed/ from there (production grammar).
        outfile = Outfile(
            filestem=str(tmp_path / "e100n2" / "correlators" / "sib3pt")
            + "/m{mass}/{gamma}/c3_{gamma}_m{mass}_{series}",
            ext=".{cfg}.h5",
            good_size=1,
        )
        base = _diagram_kwargs(outfile=outfile)
        from pyfm.a2a.types import SIBDiagramConfig

        diagram = SIBDiagramConfig(**base)
        config = SIBContractConfig(
            formatting={},
            logging_level="INFO",
            runid="t",
            diagrams={"hvp": diagram},
            time=nt,
        )

        rng = np.random.default_rng(9)
        terms = {
            term: rng.normal(size=(nt, nt, nt))
            + 1j * rng.normal(size=(nt, nt, nt))
            for term in SIB_TERM_LABELS
        }
        data_config = LoadDictConfig.create(
            dict_labels=["term"],
            array_order=["t1", "t2", "t3"],
            array_labels={t: f"0..{nt - 1}" for t in ("t1", "t2", "t3")},
        )
        fname = outfile.filename.format(
            gamma="GX_GX",
            mass=diagram.mass.to_string("l", True),
            series="a",
            cfg="20",
        )
        df = data_to_frame(terms, data_config)
        write_files(df, fname, format="hdf5")
        assert os.path.exists(fname)

        agg = build_aggregator_params(config, average=False)
        result = load_data(agg, skip_existing=False, format="hdf5")
        assert not result["hvp"].empty
        process_data(result, agg, format="hdf5")

        factors = _sib_term_factors(diagram.noise)
        expected = sum(terms[t] * factors[t] for t in SIB_TERM_LABELS)

        out_glob = (
            agg["hvp"]["out_files"]["filestem"]
            .replace("{format}", "hdf5")
            .replace("{gamma}", "*")
            .replace("{mass}", "*")
            + ".h5"
        )
        written = sorted(glob.glob(out_glob))
        assert written, out_glob
        got = pd.read_hdf(written[0])
        # t1/t2/t3 columns fold the array; compare flattened C3 values.
        corr = got["corr"].to_numpy()
        assert np.allclose(corr, np.asarray(expected).flatten(), atol=1e-12)


class TestBuildHelpers:
    def test_term_factors(self):
        _reload_sib_modules()
        from pyfm.tasks.contract.sib import _sib_term_factors

        f = _sib_term_factors(3)
        assert f["lll"] == 1.0
        assert f["nll"] == pytest.approx(1 / 3)
        assert f["nnl"] == pytest.approx(1 / 6)
        assert f["nnn"] == pytest.approx(1 / 6)

    def test_term_factors_noise_two_clamps_empty_selection(self):
        """noise=2 is valid (cross-noise pairs exist) but nnn has no
        pairwise-distinct triple — its factor must be 0, not a
        ZeroDivisionError."""
        _reload_sib_modules()
        from pyfm.tasks.contract.sib import _sib_term_factors

        f = _sib_term_factors(2)
        assert f["lll"] == 1.0
        assert f["nll"] == pytest.approx(0.5)
        assert f["nnl"] == pytest.approx(0.5)
        assert f["nnn"] == 0.0

    def test_aggregator_params_with_default_noise(self):
        """The class-default noise=2 config builds aggregator params without
        crashing (the regression the nnn factor clamp fixes)."""
        _reload_sib_modules()
        from pyfm.a2a.types import SIBContractConfig
        from pyfm.tasks.contract.sib import build_aggregator_params

        config = SIBContractConfig(
            formatting={},
            logging_level="INFO",
            runid="t",
            diagrams={"hvp": _build_diagram()},
            time=4,
        )
        agg = build_aggregator_params(config, average=False)
        assert agg["run"] == ["hvp"]
        actions = agg["hvp"]["actions"]
        assert actions["term_normalize"]["nnn"] == 0.0
        assert actions["term_sum"] == ["term"]
        assert agg["hvp"]["load_files"]["array_order"] == ["t1", "t2", "t3"]

    def test_term_labels_complete(self):
        from pyfm.a2a.types import SIB_TERM_LABELS

        assert len(SIB_TERM_LABELS) == 8
        assert len(set(SIB_TERM_LABELS)) == 8

    def test_diagram_input_round_trips_operations(self):
        from pyfm.tasks.contract.sib import _diagram_input

        diagram = _build_diagram(
            operations={
                "vec_local": {"mass": ["l", "u"]},
                "vec_onelink": {"mass": ["l"]},
            }
        )
        data = _diagram_input(diagram)
        assert data["operations"] == {
            "vec_local": {"mass": ["l", "u"]},
            "vec_onelink": {"mass": ["l"]},
        }
        assert data["contraction_type"] == "SIB"
        assert data["mass"] == {"l": 0.01, "u": 0.02, "zero": 0.0}
