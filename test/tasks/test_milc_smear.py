"""Tests for the milc_smear task create_outfile_catalog migration to catalog_files."""

from pyfm.domain import Outfile
from pyfm.tasks.milc.smear import (
    SmearConfig,
    create_outfile_catalog,
)


def _outfile(stem: str, good_size: int = 100) -> Outfile:
    return Outfile(filestem=stem, ext="", good_size=good_size)


def _config() -> SmearConfig:
    return SmearConfig(
        formatting={},
        logging_level="info",
        runid="test",
        time=32,
        space=32,
        node_geometry="1 1 1 1",
        ildg_links=_outfile("out/gauge.20"),
        long_links=_outfile("out/long.20"),
        fat_links=_outfile("out/fat.20"),
        v5_links=_outfile("in/milc.20", good_size=0),
    )


class TestCreateOutfileCatalog:
    def test_covers_three_ildg_outputs(self):
        cat = create_outfile_catalog(_config())
        assert sorted(cat["filepath"]) == [
            "out/fat.20",
            "out/gauge.20",
            "out/long.20",
        ]

    def test_excludes_v5_input(self):
        cat = create_outfile_catalog(_config())
        assert all("in/milc" not in p for p in cat["filepath"])

    def test_carries_good_size(self):
        cat = create_outfile_catalog(_config())
        assert list(cat["good_size"]) == [100, 100, 100]

    def test_has_four_base_columns(self):
        cat = create_outfile_catalog(_config())
        assert set(cat.columns) == {"filepath", "good_size", "exists", "file_size"}
