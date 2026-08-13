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
