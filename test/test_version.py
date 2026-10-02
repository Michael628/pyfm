import subprocess
import tomllib
from importlib import metadata
from pathlib import Path

import pytest
from click.testing import CliRunner

import pyfm
from pyfm import version
from pyfm.cli import cli

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"


@pytest.fixture(autouse=True)
def clear_sha_cache():
    version.git_sha.cache_clear()
    yield
    version.git_sha.cache_clear()


def _pyproject_version():
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


def test_compat_matches_pyproject_major_minor():
    assert version.major_minor(_pyproject_version()) == version.HADRONS_MILC_COMPAT


def test_installed_version_matches_pyproject():
    assert pyfm.__version__ == _pyproject_version()


def test_dist_version_falls_back_without_metadata(monkeypatch):
    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(version.metadata, "version", missing)
    assert version._dist_version() == "0+unknown"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.2", "0.2"),
        ("0.2.0", "0.2"),
        ("v0.2.3", "0.2"),
        (" 1.10.4 ", "1.10"),
        ("", None),
        ("abc", None),
    ],
)
def test_major_minor(raw, expected):
    assert version.major_minor(raw) == expected


def test_git_sha_unknown_without_dot_git(monkeypatch, tmp_path):
    monkeypatch.setattr(version, "_SOURCE_ROOT", tmp_path)
    assert version.git_sha() == version.UNKNOWN


def test_git_sha_unknown_when_git_fails(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(version, "_SOURCE_ROOT", tmp_path)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(version.subprocess, "run", fail)
    assert version.git_sha() == version.UNKNOWN


def test_git_sha_unknown_when_git_missing(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(version, "_SOURCE_ROOT", tmp_path)

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(version.subprocess, "run", missing)
    assert version.git_sha() == version.UNKNOWN


def test_build_provenance_shape(monkeypatch):
    monkeypatch.setattr(version, "git_sha", lambda: "abc123")
    prov = version.build_provenance()
    assert set(prov) == {"pyfmVersion", "pyfmSha", "hadronsMilcCompat", "generated"}
    assert all(isinstance(v, str) for v in prov.values())
    assert prov["pyfmSha"] == "abc123"
    assert prov["hadronsMilcCompat"] == version.HADRONS_MILC_COMPAT
    assert prov["generated"].endswith("Z")


PREFIX = "Hadrons : Message : 0.012345 s : "
FULL_LOG = "\n".join(
    [
        "Grid : Message : Current Grid git commit hash=7f82b1ee: (HEAD -> feature/LMI-master) clean",
        PREFIX + "HadronsMILC version=0.2.1 git=v0.2.1",
        PREFIX + "Grid git=7f82b1ee: (HEAD -> feature/LMI-master) clean",
        PREFIX + "Hadrons git=feature/LMI-develop 3a098015 (configure-time)",
        PREFIX + "GridMilc version=GridMilc 0.1.0",
        PREFIX + "Dependency pins=match",
        PREFIX
        + "Provenance pyfmVersion=0.2.0 pyfmSha=abc123-dirty "
        + "hadronsMilcCompat=0.2 generated=2026-10-02T04:00:00Z",
        PREFIX + "HadronsMILC version=9.9.9 git=later-duplicate",
    ]
)


def test_parse_run_log_full_banner():
    info = version.parse_run_log(FULL_LOG)
    assert info.hadrons_milc_version == "0.2.1"
    assert info.hadrons_milc_git == "v0.2.1"
    assert info.grid_git == "7f82b1ee: (HEAD -> feature/LMI-master) clean"
    assert info.hadrons_git == "feature/LMI-develop 3a098015 (configure-time)"
    assert info.grid_milc_version == "GridMilc 0.1.0"
    assert info.dependency_pins == "match"
    assert info.provenance == {
        "pyfmVersion": "0.2.0",
        "pyfmSha": "abc123-dirty",
        "hadronsMilcCompat": "0.2",
        "generated": "2026-10-02T04:00:00Z",
    }
    assert info.mismatch_warning is False
    assert info.compat_status() == "match"


def test_parse_run_log_empty_provenance_values_fall_back():
    log = "\n".join(
        [
            PREFIX + "HadronsMILC version=0.2.0 git=v0.2.0-3-gabc",
            PREFIX
            + "Provenance pyfmVersion= pyfmSha= hadronsMilcCompat= generated= ",
        ]
    )
    info = version.parse_run_log(log)
    assert info.provenance == dict.fromkeys(version.PROVENANCE_KEYS, "")
    assert info.expected_compat() == version.HADRONS_MILC_COMPAT
    assert info.compat_status() == "match"


def test_parse_run_log_without_banner_is_unknown():
    info = version.parse_run_log(
        "Grid : Message : Current Grid git commit hash=149dbd82: clean\n"
    )
    assert info.hadrons_milc_version is None
    assert info.grid_git is None
    assert info.compat_status() == "unknown"


def test_parse_run_log_mismatch():
    log = "\n".join(
        [
            PREFIX + "HadronsMILC version=0.3.0 git=v0.3.0",
            "Hadrons : Warning : 0.02 s : Parameter file targets HadronsMILC 0.2 "
            "but this binary is 0.3.0 (MAJOR.MINOR mismatch)",
        ]
    )
    info = version.parse_run_log(log)
    assert info.mismatch_warning is True
    assert info.compat_status() == "mismatch"


def test_cli_version_option():
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == (
        f"pyfm, version {version.__version__} "
        f"(HadronsMILC {version.HADRONS_MILC_COMPAT})"
    )
