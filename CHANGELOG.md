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

## [Unreleased]

Tracks HadronsMILC `feature/opt-a2a-general-spin-taste` @ `273c6d9` (ahead
of develop `cceb0cd`; Grid pin `5325da90`) — the QED module surface is
feature-branch-only until the branch merges; update as integration moves.

### Interface
- New task `hadrons_qed_mf`: perturbative-QED HVP diagram components —
  momentum-projected vec meson fields (±p conjugate pairs on the external
  legs), stochastic-photon `StagA2AASlashMesonField` (local J·A) and
  `StagA2AASlashOneLinkMesonField` (per-component J_kA_k) blocks in
  per-sample split_em worlds sharing one `MGauge::StochEmFunc` and the
  sib_mf deflation chain.
- New module types emitted: `MGauge::StochEmFunc`,
  `MContraction::StagA2AASlashMesonField`,
  `MContraction::StagA2AASlashOneLinkMesonField` (all feature-branch-only).
- `meson_field_v2` gains a `mom` kwarg (general momentum;
  feature-branch-only). Removed dead emitters: `qed_meson_field`, `em_fn`,
  `em_field`, `seq_aslash`, `a2a_vector`, `load_vectors` (zero callers
  since the initial import).

## [0.3.0] - 2026-10-10

Pairs with HadronsMILC `v0.3.0` (`4a19a15`), which pins Grid `d67dd20` and
Hadrons `3a09801` (both tagged `hadronsmilc/v0.3.0`).

### Interface
- Requires HadronsMILC 0.3. New module types emitted: `MFermion::SpinTaste`,
  `MUtilities::EigenPackCBPairs`, `MUtilities::GammaMapElement`,
  `MIO::LoadMesonField`, `MFermion::StagLMAMesonFieldProp`,
  `MFermion::StagGaugePropLegacy` and `MContraction::StagMesonLegacy`.
- New task `hadrons_sib_mf`: SIB meson fields with three-point contraction
  support (connected HVP `V.S.V`), `split_noise` mode and full-volume noise
  batch addressing.
- New task `hadrons_lma_new`: SpinTaste-map-driven `meson_v2`/`highmode_v2`
  LMA flow with file-driven low modes (`low_mode_method=load`) and
  `build_lh_cache` cache synthesis (including a `cache_only` pathway with
  `skip_high_modes`).
- `bias_config` is replaced by a multi-entry `high_modes` list; `high_modes`
  operations must be nested under `operations:`.
- `cross_terms` is split into `mass_cross_terms` and `solve_cross_terms`,
  with a `chain_cg_solves` toggle for independent high-mode CG solves.

### Fixed
- Sink suffix for multi-highmode entries no longer collapses distinct
  high-mode entries.
- Default high-modes label and CB-pair prefixing no longer collide on module
  names; loader mass substitution uses the prefix-removed value in
  `highmode_v2`.
- Cross-mass labels are emitted in canonical raw-key order; solve-cross
  semantics hardened after the cross-terms split.

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
