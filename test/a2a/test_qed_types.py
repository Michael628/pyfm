"""Tests for the QED contraction-type surface (types layer).

Covers the ContractType cleanup (PHOTEX/SELFEN deleted, QED added with
npoint=4), the sixteen-term junction label constant, and the
QEDDiagramConfig/QEDContractConfig dataclass surface (momentum pairing,
output tokens, EM positions). Validation hooks live in the task module
(``pyfm/tasks/contract/qed.py``) and are tested there.
"""

import pytest

from pyfm.a2a.types import (
    ContractType,
    QEDContractConfig,
    QEDDiagramConfig,
    QED_TERM_LABELS,
)
from pyfm.domain import MassDict, OpList, Outfile


def _outfile(stem: str, ext: str) -> Outfile:
    return Outfile(stem, ext, good_size=1)


def _diagram(**overrides) -> QEDDiagramConfig:
    base = dict(
        formatting={},
        logging_level="INFO",
        runid="test",
        contraction_type=ContractType.QED,
        ordering=("vec", "em", "vec", "em"),
        em=("ja", "ja"),
        operations=OpList.from_dict({"vec_local": {"mass": ["l", "u"]}}),
        mass=MassDict.from_dict({"l": 0.002426, "u": 0.001524}),
        blocks=_outfile(
            "qed/{family}_{leg_pair}{mass}_{series}{em_index}",
            ".{cfg}/{file_key}.h5",
        ),
        tab=_outfile("qed/tab_{series}", ".{cfg}/{gamma}_0_0_0.h5"),
        evalfile=_outfile("eigen/eval/eval4444nv100", ".{cfg}.h5"),
        outfile=_outfile(
            "qed4pt/m{mass}/{gamma}/c4_{gamma}_m{mass}_P{momentum}_em{em}",
            ".{cfg}.h5",
        ),
        noise=2,
        n_em=2,
        em_seeds=("qed_a_20_n0", "qed_a_20_n1"),
        momentum=((1, 0, 0),),
    )
    base.update(overrides)
    return QEDDiagramConfig(**base)


class TestContractType:
    def test_qed_npoint_is_four(self):
        assert ContractType.QED.npoint == 4

    def test_dead_members_deleted(self):
        names = [m.name for m in ContractType]
        assert names == ["TWOPOINT", "SIB", "QED"]

    def test_member_round_trip(self):
        assert ContractType.from_dict("qed") is ContractType.QED

    def test_deleted_members_rejected(self):
        with pytest.raises(ValueError):
            ContractType.from_dict("photex")
        with pytest.raises(ValueError):
            ContractType.from_dict("selfen")


class TestTermLabels:
    def test_sixteen_unique_terms(self):
        assert len(QED_TERM_LABELS) == 16
        assert len(set(QED_TERM_LABELS)) == 16

    def test_junction_letter_shape(self):
        assert QED_TERM_LABELS[0] == "llll"
        assert QED_TERM_LABELS[-1] == "nnnn"
        for label in QED_TERM_LABELS:
            assert len(label) == 4
            assert set(label) <= {"l", "n"}

    def test_noise_junction_counts(self):
        counts = [label.count("n") for label in QED_TERM_LABELS]
        assert counts.count(0) == 1
        assert counts.count(1) == 4
        assert counts.count(2) == 6
        assert counts.count(3) == 4
        assert counts.count(4) == 1


class TestQEDDiagramConfig:
    def test_npoint(self):
        assert _diagram().npoint == 4

    def test_mom_pairs_conjugate(self):
        assert _diagram().mom_pairs == [("1 0 0", "-1 0 0")]

    def test_mom_pairs_zero_momentum(self):
        d = _diagram(momentum=())
        assert d.mom_pairs == [("0 0 0", "0 0 0")]
        assert d.output_momenta == ["0_0_0"]

    def test_output_momenta_tokens(self):
        assert _diagram(
            momentum=((1, 0, 0), (0, 2, 0))
        ).output_momenta == ["1_0_0", "0_2_0"]

    def test_em_positions_photex(self):
        assert _diagram().em_positions == (1, 3)

    def test_em_positions_selfen(self):
        assert _diagram(
            ordering=("vec", "em", "em", "vec")
        ).em_positions == (1, 2)

    def test_correlator_masses_first_occurrence(self):
        assert _diagram().correlator_masses == ["l", "u"]

    def test_defaults(self):
        d = _diagram()
        assert d.defl_mass == "l"
        assert d.symmetric is False


class TestQEDContractConfig:
    def test_composite_construction(self):
        config = QEDContractConfig(
            formatting={},
            logging_level="INFO",
            runid="test",
            diagrams={"qed_ja_ja": _diagram()},
            time=4,
        )
        assert config.diagrams["qed_ja_ja"].npoint == 4
        assert config.time == 4
        assert config.overwrite is True
        assert config.comm_size == 1  # no MPI fabric in tests
        assert config.rank == 0
