"""Tests for the contract_qed task module.

Mirrors test_contract_sib.py: registration (incl. sibling isolation),
diagram/composite validation, the seed-expansion normalize, golden
input YAML, term factors, aggregator params, and an end-to-end
aggregation over synthetic raw outputs (write → load → process →
folded C4). Kernel math lives in test/a2a/test_qed_conn_4pt.py;
accessor math in test/a2a/test_qed_derive.py.
"""

import glob
import importlib
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from pyfm import utils
from pyfm.dataio import data_to_frame, write_files
from pyfm.domain import (
    LoadDictConfig,
    MassDict,
    OpList,
    Outfile,
    build_hooks,
    task_registry,
)
from pyfm.nanny import write_input_file
from pyfm.nanny.aggregator import load_data, process_data
from pyfm.nanny.taskbuilder import create_task
from pyfm.tasks.register import _config_to_task_key

import pyfm.tasks.contract.qed  # noqa: F401 — registration side effect
from pyfm.tasks.contract import qed as qed_task


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


def _reload_qed_modules():
    import pyfm.tasks.contract.qed as _q

    importlib.reload(_q)
    return _q


def _diagram_kwargs(**overrides):
    from pyfm.a2a.types import ContractType

    base = dict(
        formatting={},
        logging_level="INFO",
        runid="t",
        contraction_type=ContractType.QED,
        ordering=("vec", "em", "vec", "em"),
        em=("ja", "ja"),
        operations=OpList.from_dict({"gamma": ["vec_local"], "mass": ["l"]}),
        mass=MassDict.from_dict({"l": 0.01, "u": 0.02}),
        blocks=Outfile(
            filestem="qed/{family}_{leg_pair}{mass}_a{em_index}",
            ext=".20/{file_key}.h5",
            good_size=1,
        ),
        tab=Outfile(
            filestem="qed/tab_a", ext=".20/{gamma}_0_0_0.h5", good_size=1
        ),
        evalfile=Outfile(filestem="eval", ext=".20.h5", good_size=1),
        outfile=Outfile(
            filestem=(
                "e100n2/correlators/qed4pt/m{mass}/{gamma}/"
                "c4_{gamma}_m{mass}_P{momentum}_em{em}_{series}"
            ),
            ext=".{cfg}.h5",
            good_size=1,
        ),
        noise=2,
        n_em=2,
        em_seeds=("s_n0", "s_n1"),
        momentum=((1, 0, 0),),
        defl_mass="l",
    )
    base.update(overrides)
    return base


@pytest.mark.usefixtures("reset_registries")
class TestRegistration:
    def test_contract_qed_registered_via_nanny_keys(self):
        _reload_qed_modules()
        handler = task_registry.get("nanny_contract_qed")
        assert handler is not None
        assert handler.config_type.__name__ == "QEDContractConfig"

    def test_diagram_registered_as_subconfig(self):
        _reload_qed_modules()
        handler = task_registry.get("nanny_contract_qed_diagram")
        assert handler is not None
        assert handler.config_type.__name__ == "QEDDiagramConfig"
        with pytest.raises(AttributeError, match="build_input_params"):
            _ = handler.build_input_params

    def test_reverse_map_keys_new_classes(self):
        _reload_qed_modules()
        from pyfm.a2a.types import QEDContractConfig

        assert _config_to_task_key[QEDContractConfig] == "contract_qed"

    def test_existing_contract_keys_untouched(self):
        import pyfm.tasks.contract.mesonloader as _m
        import pyfm.tasks.contract.diagram as _d
        import pyfm.tasks.contract.contraction as _c
        import pyfm.tasks.contract.sib as _s

        importlib.reload(_m)
        importlib.reload(_d)
        importlib.reload(_c)
        importlib.reload(_s)
        _reload_qed_modules()
        for key in (
            "nanny_contract",
            "nanny_contract_diagram",
            "nanny_contract_sib",
        ):
            assert task_registry.get(key) is not None
        from pyfm.a2a.types import ContractConfig, DiagramConfig

        assert _config_to_task_key[ContractConfig] == "contract"
        assert _config_to_task_key[DiagramConfig] == "contract_diagram"


class TestNormalizeSeeds:
    def _broad(self):
        return {
            "series": "a",
            "cfg": "20",
            "n_em": 2,
            "em_seed": "qed_{series}_{cfg}",
            "noise": 2,
            "diagrams": ["qed_ja_ja"],
            "diagram_params": {
                "qed_ja_ja": {
                    "contraction_type": "qed",
                    "ordering": ["vec", "em", "vec", "em"],
                }
            },
        }

    def test_expansion(self):
        out = qed_task.normalize_qed_params(self._broad())
        assert out["diagrams"]["qed_ja_ja"]["em_seeds"] == [
            "qed_a_20_n0",
            "qed_a_20_n1",
        ]

    def test_no_mutation_of_shared_tree(self):
        params = self._broad()
        qed_task.normalize_qed_params(params)
        qed_task.normalize_qed_params(params)  # second cfgno call
        assert "em_seeds" not in params["diagram_params"]["qed_ja_ja"]
        # The shared YAML tree never gains em_seeds; the em_seed template
        # lives in shared params (job_setup params), untouched here.
        assert params["diagram_params"]["qed_ja_ja"]["contraction_type"] == "qed"

    def test_expansion_per_cfg(self):
        out20 = qed_task.normalize_qed_params(self._broad())
        params21 = self._broad() | {"cfg": "21"}
        out21 = qed_task.normalize_qed_params(params21)
        assert out21["diagrams"]["qed_ja_ja"]["em_seeds"] == [
            "qed_a_21_n0",
            "qed_a_21_n1",
        ]
        assert (
            out20["diagrams"]["qed_ja_ja"]["em_seeds"]
            != out21["diagrams"]["qed_ja_ja"]["em_seeds"]
        )

    def test_missing_series_cfg_raises(self):
        params = self._broad() | {"series": None}
        with pytest.raises(ValueError, match="series/cfg"):
            qed_task.normalize_qed_params(params)

    def test_preexpanded_passthrough(self):
        params = self._broad()
        params["diagram_params"]["qed_ja_ja"]["em_seeds"] = ["x_n0", "x_n1"]
        out = qed_task.normalize_qed_params(params)
        assert out["diagrams"]["qed_ja_ja"]["em_seeds"] == ["x_n0", "x_n1"]


@pytest.mark.usefixtures("reset_registries")
class TestValidation:
    def _validate(self, **overrides):
        from pyfm.a2a.types import QEDDiagramConfig
        from pyfm.tasks.contract.qed import validate_qed_diagram

        validate_qed_diagram(QEDDiagramConfig(**_diagram_kwargs(**overrides)))

    def test_valid_baseline_passes(self):
        self._validate()  # no exception

    def test_rejects_wrong_type(self):
        from pyfm.a2a.types import ContractType

        with pytest.raises(ValueError, match="contraction_type"):
            self._validate(contraction_type=ContractType.SIB)

    def test_rejects_bad_ordering(self):
        with pytest.raises(ValueError, match="ordering"):
            self._validate(ordering=("vec", "em", "vec", "em", "em"))
        with pytest.raises(ValueError, match="ordering"):
            self._validate(ordering=("em", "vec", "em", "vec"))
        with pytest.raises(ValueError, match="ordering"):
            self._validate(ordering=("vec", "vec", "vec", "em"))

    def test_rejects_bad_em_family(self):
        with pytest.raises(ValueError, match="EM family"):
            self._validate(em=("ja", "photon"))

    def test_rejects_noise_below_two(self):
        with pytest.raises(ValueError, match="noise"):
            self._validate(noise=1)

    def test_rejects_seed_count_mismatch(self):
        with pytest.raises(ValueError, match="em_seeds"):
            self._validate(em_seeds=("s_n0",))

    def test_rejects_missing_block_tokens(self):
        with pytest.raises(ValueError, match="blocks filestem"):
            self._validate(
                blocks=Outfile(
                    filestem="qed/{family}_{leg_pair}_a",
                    ext=".20/{file_key}.h5",
                    good_size=1,
                )
            )
        with pytest.raises(ValueError, match="file_key"):
            self._validate(
                blocks=Outfile(
                    filestem=(
                        "qed/{family}_{leg_pair}{mass}_a{em_index}"
                    ),
                    ext=".20/x.h5",
                    good_size=1,
                )
            )

    def test_rejects_missing_outfile_tokens(self):
        with pytest.raises(ValueError, match="outfile filestem"):
            self._validate(
                outfile=Outfile(
                    filestem="corr/c4_{gamma}_m{mass}_em{em}",
                    ext=".{cfg}.h5",
                    good_size=1,
                )
            )
        with pytest.raises(ValueError, match="_em"):
            self._validate(
                outfile=Outfile(
                    filestem=(
                        "corr/m{mass}/{gamma}/"
                        "c4_{gamma}_m{mass}_P{momentum}_{em}_{series}"
                    ),
                    ext=".{cfg}.h5",
                    good_size=1,
                )
            )

    def test_rejects_empty_composite(self):
        from pyfm.a2a.types import QEDContractConfig
        from pyfm.tasks.contract.qed import validate_config

        with pytest.raises(ValueError, match="must not be empty"):
            validate_config(
                QEDContractConfig(
                    formatting={},
                    logging_level="INFO",
                    runid="t",
                    diagrams={},
                    time=4,
                )
            )


class TestGoldenInput:
    def test_generate_contract_qed_input(
        self, tmp_path, monkeypatch, tasks_data_dir, contract_params
    ):
        """Runs WITHOUT reset_registries — the production import-time
        wiring must be intact for write_input_file (the sib golden
        precedent)."""
        monkeypatch.chdir(tmp_path)
        write_input_file("contract_qed", contract_params, "a", "20")
        actual = yaml.safe_load(
            (tmp_path / "in" / "contract-qed-a.20.yaml").read_text()
        )
        expected = yaml.safe_load(
            (tasks_data_dir / "in" / "test-contract-qed-a.20.yaml").read_text()
        )
        assert actual == expected, "contract_qed YAML output mismatch"


class TestAggregation:
    def test_term_factors(self):
        factors = qed_task._qed_term_factors(noise=2, n_em=4)
        assert factors["llll"] == pytest.approx(1 / 4)
        assert factors["nlll"] == pytest.approx(1 / (2 * 4))
        assert factors["nnll"] == pytest.approx(1 / (2 * 1 * 4))
        assert factors["nnnl"] == 0.0
        assert factors["nnnn"] == 0.0

    def test_aggregator_params_shape(self, contract_params):
        task = create_task("contract_qed", contract_params, "a", "20")
        agg = task.handler.build_aggregator_params(task.config, False)
        assert agg["run"] == ["qed_ja_ja", "qed_selfen"]
        d = agg["qed_ja_ja"]
        assert d["actions"]["term_sum"] == ["em", "term"]
        assert d["load_files"]["array_order"] == ["t1", "t2", "t3", "t4"]
        assert d["load_files"]["dict_labels"] == ["term", "momentum", "em"]
        assert "_em{em}" not in d["out_files"]["filestem"]
        assert "{momentum}" in d["out_files"]["filestem"]
        assert "{series}" not in d["out_files"]["filestem"]

    def test_average_rejected(self, contract_params):
        task = create_task("contract_qed", contract_params, "a", "20")
        with pytest.raises(ValueError, match="average"):
            task.handler.build_aggregator_params(task.config, True)

    def test_end_to_end_fold(self, tmp_path):
        """Write synthetic raw per-term files in the outfile grammar, run
        the aggregator load+process pipeline, and check the folded C4:
        em0 contributes ×1 and em1 ×2 (sum ×3); every term folds with
        its factor; the term axis disappears."""
        from pyfm.a2a.types import (
            QEDContractConfig,
            QEDDiagramConfig,
            QED_TERM_LABELS,
        )

        nt = 2
        outfile = Outfile(
            filestem=str(tmp_path / "e100n2" / "correlators" / "qed4pt")
            + "/m{mass}/{gamma}/c4_{gamma}_m{mass}_P{momentum}_em{em}_{series}",
            ext=".{cfg}.h5",
            good_size=1,
        )
        diagram = QEDDiagramConfig(
            **_diagram_kwargs(outfile=outfile)
        )
        config = QEDContractConfig(
            formatting={},
            logging_level="INFO",
            runid="t",
            diagrams={"qed": diagram},
            time=nt,
        )

        rng = np.random.default_rng(9)
        terms = {
            term: rng.normal(size=(nt,) * 4)
            + 1j * rng.normal(size=(nt,) * 4)
            for term in QED_TERM_LABELS
        }
        data_config = LoadDictConfig.create(
            dict_labels=["term"],
            array_order=["t1", "t2", "t3", "t4"],
            array_labels={
                t: f"0..{nt - 1}" for t in ("t1", "t2", "t3", "t4")
            },
        )
        for em, scale in (("0", 1.0), ("1", 2.0)):
            fname = outfile.filename.format(
                gamma="GX_GX",
                mass=diagram.mass.to_string("l", True),
                momentum="1_0_0",
                em=em,
                series="a",
                cfg="20",
            )
            df = data_to_frame(
                {t: arr * scale for t, arr in terms.items()}, data_config
            )
            write_files(df, fname, format="hdf5")
            assert os.path.exists(fname)

        agg = qed_task.build_aggregator_params(config, average=False)
        result = load_data(agg, skip_existing=False, format="hdf5")
        assert not result["qed"].empty
        process_data(result, agg, format="hdf5")

        factors = qed_task._qed_term_factors(diagram.noise, diagram.n_em)
        expected = sum(terms[t] * factors[t] * 3.0 for t in QED_TERM_LABELS)

        out_glob = (
            agg["qed"]["out_files"]["filestem"]
            .replace("{format}", "hdf5")
            .replace("{gamma}", "*")
            .replace("{mass}", "*")
            .replace("{momentum}", "*")
            + ".h5"
        )
        written = sorted(glob.glob(out_glob))
        assert written, out_glob
        got = pd.read_hdf(written[0])
        corr = got["corr"].to_numpy()
        assert np.allclose(
            corr, np.asarray(expected).flatten(), atol=1e-12
        )


# --- Step-5 triage additions (rows 1 and 3) --------------------------------


class TestOutfileRouting:
    """The contract_qed label must route cfg_h5.

    Outfile.from_param's name_keys walk checks "qed" before "contract",
    and the label contains both — the explicit earlier "contract_qed"
    entry added to pyfm/domain/outfiles.py in §6 pins the correlator
    outfile to the plain per-cfg ext. Pure qed_* labels keep the block
    token ext.
    """

    def test_contract_qed_label_routes_cfg_h5(self):
        o = Outfile.from_param(
            "contract_qed", "/tmp", {"filestem": "x", "good_size": 1}
        )
        assert o.ext == ".{cfg}.h5"

    def test_pure_qed_labels_still_route_cfg_qed_h5(self):
        o = Outfile.from_param(
            "qed_blocks", "/tmp", {"filestem": "x", "good_size": 1}
        )
        assert o.ext == ".{cfg}/{file_key}.h5"


class TestCatalogUniqueness:
    """Emitted-name uniqueness per diagram (the 70d8419/6252189 lessons).

    Scoped PER DIAGRAM: the design's D4 outfile grammar carries no diagram
    axis, so multiple diagrams sharing one outfile label legitimately
    produce overlapping filepaths in the concatenated catalog (flagged in
    Developer Context). Within one diagram's emission, filepaths must be
    unique — a double-suffixed or token-colliding stem fails here.
    """

    def test_diagram_catalog_emits_unique_filepaths(self, contract_params):
        from pyfm.tasks.contract.qed import _diagram_catalog

        task = create_task("contract_qed", contract_params, "a", "20")
        for diagram in task.config.diagrams.values():
            catalog = _diagram_catalog(diagram)
            assert catalog["filepath"].is_unique, (
                "diagram catalog emitted duplicate filepaths"
            )
