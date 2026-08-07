from pyfm.domain import Gamma, MassDict, OpList, Outfile
from pyfm.tasks.hadrons.highmode.twopoint import contraction_gen, quark_gen
from pyfm.tasks.hadrons.types import HighModeConfig


def make_config(gamma: Gamma) -> HighModeConfig:
    return HighModeConfig(
        formatting={},
        logging_level="INFO",
        runid="test",
        mass=MassDict.from_dict({"l": 0.01}),
        action_name="action_{mass}",
        solver_name="solver_{solver}_{mass}",
        low_modes_name="low_modes",
        operations=OpList([OpList.Op(gamma=gamma, mass=("l",))]),
        high_modes=Outfile(
            filestem="corr/{mass}/{dset}/{gamma_label}/corr_{tsource}",
            ext=".h5",
            good_size=1,
        ),
        tstart=0,
        tstop=0,
        dt=1,
        noise=1,
        time=64,
        skip_cg=True,
        shift_gauge_name="shift_gauge",
    )


def test_quark_gen_handles_fourvec_ops():
    assert {op.gamma for op in quark_gen(make_config(Gamma.FOURVEC_LOCAL))} == {
        Gamma.PION_LOCAL,
        Gamma.FOURVEC_LOCAL,
    }
    assert {op.gamma for op in quark_gen(make_config(Gamma.FOURVEC_ONELINK))} == {
        Gamma.PION_LOCAL,
        Gamma.FOURVEC_ONELINK,
    }
    assert {op.gamma for op in quark_gen(make_config(Gamma.AXIAL_FOURVEC_LOCAL))} == {
        Gamma.IDENTITY,
        Gamma.FOURVEC_LOCAL,
    }
    assert {op.gamma for op in quark_gen(make_config(Gamma.AXIAL_FOURVEC_ONELINK))} == {
        Gamma.IDENTITY,
        Gamma.FOURVEC_ONELINK,
    }


def test_contraction_gen_handles_fourvec_ops():
    expected = {
        Gamma.FOURVEC_LOCAL: (Gamma.FOURVEC_LOCAL, Gamma.PION_LOCAL),
        Gamma.FOURVEC_ONELINK: (Gamma.FOURVEC_ONELINK, Gamma.PION_LOCAL),
        Gamma.AXIAL_FOURVEC_LOCAL: (Gamma.FOURVEC_LOCAL, Gamma.IDENTITY),
        Gamma.AXIAL_FOURVEC_ONELINK: (Gamma.FOURVEC_ONELINK, Gamma.IDENTITY),
    }

    for gamma, (quark_gamma, antiquark_gamma) in expected.items():
        op, con = next(contraction_gen(make_config(gamma)))
        assert op.gamma == gamma
        assert con.quark.gamma == quark_gamma
        assert con.antiquark.gamma == antiquark_gamma
        assert con.sink.gamma == gamma
