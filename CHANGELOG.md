# Changelog

pyfm and [HadronsMILC](https://github.com/milc-qcd/HadronsMILC) share
MAJOR.MINOR: pyfm `X.Y.*` emits XML for HadronsMILC `X.Y.*`. Each repository
bumps PATCH independently. The MINOR is bumped when the set of emitted module
types or their option keys (`pyfm/tasks/hadrons/modules.py`) changes.

`pyfm.version.HADRONS_MILC_COMPAT` records the targeted MAJOR.MINOR and is
written into every Hadrons input file as `<grid><provenance>`. After a run,
`pyfm audit version <log>` checks it against the binary's version banner.

## Release process

1. Merge `develop` into `main` with `git merge --no-ff develop`, run on `main`,
   so that `main` stays the first parent. Never fast-forward `main`.
2. On `main`, commit the release-only changes: `version` in `pyproject.toml`,
   `HADRONS_MILC_COMPAT` in `pyfm/version.py` when the MINOR changes, and this
   file's section for the release.
3. Tag that commit `vX.Y.Z` (annotated). Release tags must be on
   `git log --first-parent main`. To check:
   `git rev-list --first-parent main | grep -q "$(git rev-list -n1 vX.Y.Z)"`.
4. Merge `main` back into `develop`.

`v0.1.0` (`1c561dd`) is the only exception to the first-parent rule. It was
tagged retroactively on a commit reached through `develop`.

## [0.2.0]

Pairs with HadronsMILC `v0.2.0` (`d6503a6`), which pins Grid `7f82b1ee` and
Hadrons `3a098015` (both tagged `hadronsmilc/v0.2.0`).

### Interface
- Requires HadronsMILC 0.2. New module types emitted: `MGauge::HISQSmear`,
  `MIO::LoadMilc`, `MIO::SaveIldg`.
- `MAction::ImprovedStaggeredMILC` emits only `mass`, `gaugefat` and
  `gaugelong`. The removed `c1`/`tad`/`twist` parameters are no longer written.
- Split-grid runs emit `<parameters><split><mpiSplit>` and per-module
  `<subgrid>`. These are parsed by the pinned Hadrons (`feature/split-grid-integration`).
- Never released: `MAction::HighlyImprovedStaggeredMILC` (pyfm
  `20dbe52`..`918a129`), which paired only with transient HadronsMILC develop commits.

### Added
- `pyfm.__version__` and the `pyfm.version` module (`HADRONS_MILC_COMPAT`,
  `git_sha`, `build_provenance`, `parse_run_log`).
- `<grid><provenance>` in every Hadrons input file, with leaves `pyfmVersion`,
  `pyfmSha` (`git describe --always --dirty`, or `unknown`), `hadronsMilcCompat`
  and `generated` (UTC). Binaries that predate HadronsMILC's version checks
  ignore it.
- `pyfm --version`.
- `pyfm audit version LOG`, which reports the HadronsMILC/Grid/Hadrons version
  lines and the provenance from a run log. It exits 1 on a MAJOR.MINOR mismatch
  and reports `unknown` for binaries without a version banner.
- `bias_config` on LMI tasks (`tasks.bias` YAML block): `nbias` seeded random
  wall sources as a second high-mode configuration. Uses existing module types
  only; output is unchanged when `bias` is unset.
- `pyfm export convert --average`.
- `pyfm task aggregate --generate-manifest`, which writes manifest sidecars
  from existing processed files without re-aggregating.

### Changed
- Data I/O refactor: the chunked HDF5 loader uses a fork-pinned process pool,
  and HDF5 files are opened with `locking=False` for concurrent loads.
- Requires `h5py>=3.5` (was `>=2.10.0`).

### Behavior
- HadronsMILC 0.2 clamps the `MSolver::StagMixedPrecisionCG` inner tolerance to
  `max(residual, 1e-7)`. pyfm's default `residual` of `1e-8`
  (`pyfm/tasks/hadrons/types.py`) is below that floor, so default runs are affected.

### Removed
- `setup.py` and `requirements.txt`. `pyproject.toml` is the only metadata
  source, and `Requires-Python` is now `>=3.12` (setup.py had leaked `>=3.11`).

### Known issues
- `em_field()` emits `MGauge::StochEm`, which neither pinned Hadrons registers.
  It has no callers.

## [0.1.0]

`1c561dd`, retroactive baseline. Pairs with HadronsMILC `v0.1.0` (`40877b8`),
which pins Grid `149dbd82` and Hadrons `9371ba59`. It has no runtime version
metadata, and `pyproject.toml` declared `0.1.0`.
