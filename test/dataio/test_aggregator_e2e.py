"""End-to-end --skip-existing round trip through load_data/process_data.

Closes the prior plan's verification gap: the manifest write -> read -> skip ->
rewrite cycle was claimed "verified in Phase 6's tests" but no test invoked
``load_data(skip_existing=True)`` end-to-end. Everything here is real files on
disk (raw Grid HDF5 inputs, unmocked dataio) via the shared factory fixtures.

The round trip requires ``utils.io.process_files`` to tolerate replacement
keys that do not occur in a brace-free output filestem (fixed alongside this
test in ``pyfm/utils/io.py``): ``load_data``'s skip-existing re-read unions
``{"format": fmt}`` plus the load replacements into the output-stem load.
"""
import logging
import os

import h5py
import numpy as np
import pandas as pd

from pyfm.nanny.aggregator import _read_skip_manifest, load_data, process_data


def _write_one_raw_grid(base: str, cfg, n_t: int = 24, n_meson: int = 2) -> str:
    """One raw Grid corr.h5, mirroring the conftest factory's file body."""
    p = os.path.join(base, "a", str(cfg), "corr.h5")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with h5py.File(p, "w") as f:
        for i in range(n_meson):
            g = f.create_group(f"meson/meson_{i}")
            ds = g.create_dataset("corr", data=np.arange(n_t, dtype=np.complex128) + i)
            ds.attrs["sourceGamma"] = "GX_GX"
    return p


def _agg_params(tmpdir: str, h5kwargs: dict) -> dict:
    """Hand-rolled agg_params (shape precedents: scripts/aggregate_hadrons_contract_data.py:60-75,
    test/nanny/test_aggregator.py:61-70)."""
    return {
        "run": ["r1"],
        "r1": {
            "load_files": {
                "filestem": os.path.join(tmpdir, "{series}", "{cfg}", "corr.h5"),
                "replacements": {"series": ["a"]},
                "regex": {"cfg": "[0-9]+"},
                **h5kwargs,
            },
            "out_files": {"filestem": os.path.join(tmpdir, "out", "corr")},
            "actions": {},
        },
    }


def test_load_data_skip_existing_round_trip(raw_grid_h5_factory):
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=6)
    params = _agg_params(tmpdir, h5kwargs)
    out_stem = os.path.join(tmpdir, "out", "corr")
    out_csv = out_stem + ".csv"

    # First run: aggregate all 6 inputs; process_data writes the CSV agg and
    # its manifest sidecar (unmocked write path).
    process_data(load_data(params, skip_existing=False, format="csv"), params, "csv")

    assert os.path.exists(out_csv)
    assert len(pd.read_csv(out_csv)) == 6 * 48  # 6 cfg × 24 t × 2 gamma

    skip_set, manifest_exists = _read_skip_manifest(out_stem, format="csv")
    assert manifest_exists
    assert len(skip_set) == 6
    assert {os.path.join(tmpdir, "a", str(i), "corr.h5") for i in range(6)} <= skip_set

    # A new input appears (cfg=6) after the first aggregation.
    _write_one_raw_grid(tmpdir, 6)

    # Second run with skip_existing: only the new file is loaded; old rows are
    # carried from the existing agg exactly once (no duplication).
    process_data(load_data(params, skip_existing=True, format="csv"), params, "csv")

    final = pd.read_csv(out_csv)
    assert len(final) == 7 * 48
    assert final["cfg"].nunique() == 7

    skip_set2, _ = _read_skip_manifest(out_stem, format="csv")
    assert len(skip_set2) == 7


def test_load_data_skip_existing_missing_manifest_warns_and_aggregates_all(
    raw_grid_h5_factory, caplog
):
    """Design Q4→C: a missing manifest warns and aggregates ALL inputs (the
    manifest is rewritten by process_data afterwards). With an existing agg on
    disk that means old rows + full reload — pinned here as the documented
    contract, not silent dedup."""
    tmpdir, h5kwargs = raw_grid_h5_factory(n_cfg=3)
    params = _agg_params(tmpdir, h5kwargs)

    process_data(load_data(params, skip_existing=False, format="csv"), params, "csv")

    os.remove(os.path.join(tmpdir, "out", "corr.manifest.json"))

    with caplog.at_level(logging.WARNING):
        result = load_data(params, skip_existing=True, format="csv")

    assert "No manifest found" in caplog.text
    # old agg (3×48) + full reload of all 3 inputs (3×48)
    assert len(result["r1"]) == 2 * 3 * 48
