"""Unit tests for pyfm/utils/io.py helpers."""
import pytest

from pyfm.utils import io


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("csv", ".csv"),
        ("hdf5", ".h5"),
        ("parquet", ".parquet"),
        ("dict", ".npy"),
    ],
)
def test_get_file_ext_from_format(fmt, expected):
    assert io.get_file_ext_from_format(fmt) == expected


def test_get_file_ext_from_format_rejects_unknown():
    with pytest.raises(ValueError, match="Invalid format option"):
        io.get_file_ext_from_format("xml")


def test_process_files_ignores_replacements_not_in_filestem(tmp_path):
    """Non-matching replacement keys are dropped, not crashed on (brace-free stem).

    Regression guard for the --skip-existing re-read path: load_data unions
    {"format": fmt} (plus the load replacements) into the output-filestem load;
    when the output stem has no template keys the pre-fix code crashed with
    ValueError from zip(*) on an empty product.
    """
    target = tmp_path / "out" / "corr.csv"
    target.parent.mkdir()
    target.write_text("x")
    out = io.process_files(
        str(target),
        lambda f, r: (f, dict(r)),
        replacements={"series": ["a"], "format": "csv"},
    )
    assert out == [(str(target), {})]


def test_process_files_templated_stem_still_expands(tmp_path):
    """Templated stems keep exact expansion semantics (fix changes nothing here)."""
    p = tmp_path / "csv" / "pion.csv"
    p.parent.mkdir()
    p.write_text("x")
    out = io.process_files(
        str(tmp_path / "{format}" / "pion.csv"),
        lambda f, r: (f, dict(r)),
        replacements={"format": "csv"},
    )
    assert out == [(str(p), {"format": "csv"})]
