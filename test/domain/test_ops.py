import pytest

from pyfm.domain.ops import Gamma, OpList


@pytest.mark.parametrize(
    ("gamma", "gamma_list", "gamma_string", "local", "shift"),
    [
        (
            Gamma.FOURVEC_ONELINK,
            ["GX_G1", "GY_G1", "GZ_G1", "GT_G1"],
            "(GX G1) (GY G1) (GZ G1) (GT G1)",
            False,
            1,
        ),
        (
            Gamma.FOURVEC_LOCAL,
            ["GX_GX", "GY_GY", "GZ_GZ", "GT_GT"],
            "(GX GX) (GY GY) (GZ GZ) (GT GT)",
            True,
            0,
        ),
        (
            Gamma.AXIAL_FOURVEC_ONELINK,
            ["G5X_G5", "G5Y_G5", "G5Z_G5", "G5T_G5"],
            "(G5X G5) (G5Y G5) (G5Z G5) (G5T G5)",
            False,
            1,
        ),
        (
            Gamma.AXIAL_FOURVEC_LOCAL,
            ["G5X_G5X", "G5Y_G5Y", "G5Z_G5Z", "G5T_G5T"],
            "(G5X G5X) (G5Y G5Y) (G5Z G5Z) (G5T G5T)",
            True,
            0,
        ),
    ],
)
def test_fourvec_gamma_metadata(gamma, gamma_list, gamma_string, local, shift):
    assert gamma.gamma_list == gamma_list
    assert gamma.gamma_string == gamma_string
    assert gamma.local is local
    assert gamma.shift == shift


def test_op_list_parses_fourvec_gamma_names():
    op_list = OpList.from_dict(
        {
            "gamma": [
                "fourvec_onelink",
                "fourvec_local",
                "axial_fourvec_onelink",
                "axial_fourvec_local",
            ],
            "mass": "l",
        }
    )

    assert [op.gamma for op in op_list] == [
        Gamma.FOURVEC_ONELINK,
        Gamma.FOURVEC_LOCAL,
        Gamma.AXIAL_FOURVEC_ONELINK,
        Gamma.AXIAL_FOURVEC_LOCAL,
    ]
