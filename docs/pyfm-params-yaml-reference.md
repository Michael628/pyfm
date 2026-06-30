# PyFM `params.yaml` Parameter Reference

> A section-by-section walkthrough of a PyFM parameter file, using
> `example/params_files/params_l3248.yaml` as the worked example.
> Separated by `---` so each section can become its own slide.

## How to read this document

Every value in `params.yaml` ends up in one of two places:

1. **String template / literal text** — the value is a string (often with `{key}` placeholders) that gets pattern-substituted at runtime. Unmatched `{key}` placeholders are left as literals (`PartialFormatter`).
2. **Config object input** — the value is handed to a Pydantic config object in `pyfm/tasks/` (sometimes after a `preprocess`/`normalize` step) and becomes a typed property, e.g. `high_mode_config.tstart`.

Two substitution mechanisms appear throughout:
- `PYFM_WORKSPACE_*` — environment placeholders, replaced by `pyfm workspace setup`/`env` (e.g. `PYFM_WORKSPACE_TOPDIR`, `PYFM_WORKSPACE_SCHEDULER`).
- `{ens}`, `{series}`, `{cfg}`, `{eigs}`, `{mass}`, ... — runtime keys filled from `shared_params`, `<job-type>_params`, and the current series/config.

**Merge hierarchy**:
> how `params.yaml` sections combine to build config objects 
> (from `pyfm/nanny/core.py`)

`shared_params`  
- defaults injected EVERYWHERE (nanny config, every job_type, e.g. `hadrons`, and every task_type, e.g. `lmi`)
\+ `nanny` + `submit` + `files`         → NannyConfig
\+ `job_setup.<step>` + `submit.layout` → JobConfig
\+ `<job_type>_params` (e.g. `hadrons_params`, `contract_params`) + `job_setup.<step>.params` → task global params
\+ `job_setup.<step>.tasks`           → the task's nested config structure

---

## `shared_params` — global defaults

```yaml
shared_params:
  home: PYFM_WORKSPACE_TOPDIR/l{ens}
  mass:
    u: 0.001524
    l: 0.002426
  ens: '3248f211b580m002426m06730m8447'
  space: 32
  time: 48
  eigs: 1000
  dt: 1
  noise: 1
  lattice: [32, 32, 32, 48]
  runid: "LMI-RW-series-{series}-{eigs}-eigs-{noise}-noise"
  logging_level: DEBUG
```

Merged into the nanny config **and** every job/task. These are the values most `{placeholders}` resolve against.

| Key | Role | Consumed as |
|-----|------|-------------|
| `home` | Work directory root; `PYFM_WORKSPACE_TOPDIR/l{ens}` | String template (env + `{ens}`) |
| `mass` | Mass label → value map (`u`, `l`, `d`, ...) | Config input → `MassDict` |
| `logging_level` | Verbosity (`DEBUG`/`INFO`/...) | Config property |
| `ens` | Ensemble tag, fills `{ens}` everywhere | Template key |
| `space` / `time` | Spatial / temporal lattice extent | Config input (e.g. `SmearConfig`, `*.time`) |
| `runid` | Run identifier string with `{series}/{eigs}/{noise}` | String template |
| `dt` | Source-time-slice spacing, fills `{dt}` | Config input (`HighModeConfig.dt`) + template key |
| `noise` | Noise count (keep `1`), fills `{noise}` | Config input + template key |
| `eigs` | Number of eigenvalues used, fills `{eigs}` | Config input (`EpackConfig.eigs`) + template key |
| `lattice` | `[nx, ny, nz, nt]` geometry | Config input (`JobConfig.lattice`) |

**Tip for slides:** `mass` is special — many configs take a `MassDict` and look up labels (`"l"`, `"u"`) rather than raw numbers.

---

## `contract_params` — A2A contraction settings

```yaml
contract_params:
  hardware: "cpu"
  diagram_params:
    pion_local_u:
      ...
    vec_local_u:
      ...
```

Becomes the `contract_params` global layer for `contract` jobs and is built into `ContractConfig` (`pyfm/a2a/types.py`).

| Key | Role | Consumed as |
|-----|------|-------------|
| `hardware` | `"cpu"` or `"gpu"` backend selection | `ContractConfig.hardware` |
| `diagram_params` | Map of diagram-name → diagram definition | `ContractConfig.diagrams` (each → `DiagramConfig`) |

`overwrite` (default `True`) and `time` also live on `ContractConfig` (time pulled from `shared_params`).

---

## `contract_params.diagram_params.<name>` — one diagram → `DiagramConfig`

```yaml
contract_params:
  diagram_params:
    pion_local_u:           # diagram label
      gamma_label: pion_local
      symmetric: True
      contraction_type: two_point
      mesons:               # one entry or a list
        - file: "meson"
          mass_original: "l"
          mass_updated: "u"
          evalfile: "eval"
      gammas:
        - "G5_G5"
      eig_range:
        min: 0
        max: 2000
      outfile: contract
```

Each entry builds a `DiagramConfig` (a `CompositeConfig`). Note the `normalize`/`route` preprocessing: the `mesons` block is canonicalized (`mass`→`mass_original`, `new_mass`→`mass_updated`) and routed to child `MesonLoaderConfig` objects.

| Key | Role | Consumed as |
|-----|------|-------------|
| `gamma_label` | Label used in output filenames (`{gamma_label}`) | Template key |
| `symmetric` | Use symmetric contraction | `DiagramConfig.symmetric` |
| `contraction_type` | `two_point` (→ npoint=2), `sib`, ... | `DiagramConfig.contraction_type` (`ContractType`) |
| `mesons` | One or a list of meson-field sources | `DiagramConfig.mesons` → `List[MesonLoaderConfig]` |
| `gammas` | Gamma-structure pairs, e.g. `"G5_G5"` | `DiagramConfig.gammas` |
| `eig_range` | `{min, max}` low-mode (eigenvector) index window | `DiagramConfig.eig_range` (`MesonIndex`) |
| `stoch_range` | `{min, max}` stochastic high-mode window | `DiagramConfig.stoch_range` (requires `stoch_seed_indices`) |
| `outfile` | **Label** into the `files:` section for output | Resolved to `Outfile` |
| `perms` *(opt)* | Explicit permutation list | `DiagramConfig.perms` |

Validation: must provide `eig_range` or `stoch_range`; `mesons` must be non-empty.

---

## `diagram_params.<name>.mesons[]` — meson source → `MesonLoaderConfig`

```yaml
# Mass-shifted (u from l eigenvectors):
mesons:
  - file: "meson"
    mass_original: "l"
    mass_updated: "u"
    evalfile: "eval"

# No shift (shorthand):
mesons:
  file: "meson"
  mass: "l"
```

Each meson entry describes which meson-field file to load and any mass shift applied during contraction.

| Key | Role | Consumed as |
|-----|------|-------------|
| `file` | **Label** into `files:` for the meson field | Resolved to `Outfile` |
| `mass` *(shorthand)* | Original mass label (no shift) | → `mass_shift.original` |
| `mass_original` | Mass used when the meson file was created | `MassShift.original` |
| `mass_updated` | Desired (shifted) mass label | `MassShift.updated` |
| `evalfile` | **Label** into `files:` for eigenvalue file | `MesonLoaderConfig.evalfile` (required if shifting) |

Rule: if `mass_updated` is set, an `evalfile` **must** be provided (you can't shift a mass without eigenvalues).

---

## `hadrons_params` — Grid/Hadrons LMI settings

```yaml
hadrons_params:
  tstart: 0
  tstop: 47          # time - 1
  sourceeigs: 1000
  residual: 1e-8
  blocksize: 500
  overwrite: false
  lanczos:
    ...
```

The `hadrons_params` global layer for `hadrons` jobs; feeds the LMI subconfigs (`EpackConfig`, `MesonConfig`, `HighModeConfig`).

| Key | Role | Consumed as |
|-----|------|-------------|
| `tstart` / `tstop` | First / last source time slice (`tstop = time-1`) | `HighModeConfig.tstart/tstop` |
| `sourceeigs` | Eigenvalues present in the source file | task input |
| `lanczos` | IRL eigensolver block (see below) | `EpackConfig.lanczos` → `LanczosParams` |
| `residual` | CG/solver residual target | `HighModeConfig.residual` |
| `blocksize` | Fields contracted per meson-field kernel call | `MesonConfig.blocksize` |
| `overwrite` | Skip solves/gammas if output already exists | `*.overwrite` |

---

## `hadrons_params.lanczos` — `LanczosParams`

```yaml
hadrons_params:
  lanczos:
    alpha: 0.009
    beta: 24
    npoly: 81
    nstop: 1000
    nk: 1030
    nm: 1600
    residual: 1e-8
```

Direct inputs to the Lanczos/IRL eigensolver (`pyfm/tasks/hadrons/types.py`). All are required except `residual`.

| Key | Role |
|-----|------|
| `alpha` / `beta` | Chebyshev acceleration window bounds |
| `npoly` | Chebyshev polynomial order |
| `nstop` | Converged-eigenvalue stop count |
| `nk` / `nm` | Krylov subspace sizes (working / max) |
| `residual` | Convergence residual (default `1e-8`) |

---

## `nanny` — todo-file / loop control

```yaml
nanny:
  todo_file: todo
  max_cases: 1
  max_queue: 200
  wait: 5
  check_interval: 30
```

Merged into `NannyConfig` (`pyfm/nanny/core.py`). Controls the automated submission loop, not the physics.

| Key | Role |
|-----|------|
| `todo_file` | Filename listing jobs to run per configuration |
| `max_cases` | Bundle up to N jobs together (`>1` = bundling) |
| `max_queue` | Don't submit if this many jobs already queued |
| `wait` | Seconds between submissions |
| `check_interval` | Seconds between completion checks |

---

## `submit` — scheduler & process layout

```yaml
submit:
  scheduler: PYFM_WORKSPACE_SCHEDULER
  job_name_pfx: LMI
  layout:
    ppn: 4             # default processes per node
    hadrons:
      nodes: 1
      geom: [1, 1, 1, 2]
    smear:
      ppn: 1
      nodes: 1
      geom: [1, 1, 1, 1]
    contract:
      ppn: 1
      nodes: 1
      geom: [1, 1, 1, 1]
```

Also merged into `NannyConfig`; its `layout` sub-block is merged into each `JobConfig`.

| Key | Role | Consumed as |
|-----|------|-------------|
| `scheduler` | `SLURM`/`PBS`/`LSF`/`INTERACTIVE`/`COBALT` (or `PYFM_WORKSPACE_SCHEDULER`) | `NannyConfig.scheduler` (`Scheduler` enum) |
| `job_name_pfx` | Prefix for submitted job names | `NannyConfig.job_name_pfx` |
| `layout.ppn` | Default processes per node | `JobConfig.ppn` |
| `layout.<step>.nodes` | Node count for that step | `JobConfig.nodes` |
| `layout.<step>.geom` | MPI geometry split of the lattice; product must equal `nodes*ppn` | `JobConfig.geom` |
| `layout.<step>.ppn` | Per-step override of `ppn` | `JobConfig.ppn` |

---

## `job_setup` — the job steps

```yaml
job_setup:
  smear:
    run: milc.PYFM_WORKSPACE_SCHEDULER
    job_type: smear
    io: "smear"
    wall_time: "0:30:00"
    tasks: ...
  contract:
    run: contract.PYFM_WORKSPACE_SCHEDULER
    job_type: contract
    io: "contract-e{eigs}-n{noise}"
    wall_time: "0:30:00"
    tasks: ...
  hadrons:
    run: grid.PYFM_WORKSPACE_SCHEDULER
    job_type: hadrons
    task_type: lmi
    io: "lma-e{eigs}-n{noise}"
    wall_time: "1:00:00"
    tasks: ...
```

A map of step-name → job definition. Each entry merges with `shared_params` + `submit.layout` to build a `JobConfig`. The `job_type`/`task_type` pair selects the registered task handler.

| Key | Role | Consumed as |
|-----|------|-------------|
| `run` | Scheduler script filename (often `*.PYFM_WORKSPACE_SCHEDULER`) | `JobConfig.run` |
| `job_type` | Handler family: `smear`, `contract`, `hadrons` | `JobConfig.job_type` |
| `task_type` | Sub-handler, e.g. `lmi` (hadrons) | `JobConfig.task_type` |
| `io` | Output/input filename stem template (`{eigs}`, `{noise}`) | `JobConfig.io` (string template) |
| `wall_time` | Scheduler wall-clock request | `JobConfig.wall_time` |
| `tasks` | Nested per-task config (handed to the task handler) | `JobConfig.tasks` |

The registered handler key is `"{job_type}_{task_type}"` (e.g. `hadrons_lmi`, `contract_diagram`).

---

## `job_setup.smear.tasks` — MILC smearing → `SmearConfig`

```yaml
job_setup:
  smear:
    tasks:
      unsmeared_file: "lat/v5/l{ens}{series}.{cfg}"
      node_geometry: "1 1 1 1"
```

| Key | Role | Consumed as |
|-----|------|-------------|
| `unsmeared_file` | Path to the input unsmeared lattice (`{ens}{series}.{cfg}`) | `SmearConfig.unsmeared_file` (template) |
| `node_geometry` | MILC node geometry string | `SmearConfig.node_geometry` |

`SmearConfig` also pulls `time`, `space`, and the `gauge/long/fat_links` `Outfile`s from `shared_params` + `files`.

---

## `job_setup.contract.tasks` — contraction job

```yaml
job_setup:
  contract:
    tasks:
      diagrams:
        - pion_local_l
        - vec_local_l
        - pion_local_u
        - vec_local_u
        - pion_local_d
        - vec_local_d
```

| Key | Role |
|-----|------|
| `diagrams` | List of diagram names (keys into `contract_params.diagram_params`) to compute in this job |

---

## `job_setup.hadrons.tasks` — LMI task → `LMIConfig`

```yaml
job_setup:
  hadrons:
    tasks:
      epack:
        load: false
        save_evals: true
        save_eigs: true
      meson:
        gamma: [vec_onelink, local]
        mass: ["l"]
      high_modes:
        vec_local:
          mass: ["l"]
        pion_local:
          mass: ["l"]
        vec_onelink:
          mass: ["l"]
```

This is the richest task. `LMIConfig` is composite: each `tasks` sub-block routes to a child config, and **omitting a sub-block sets the corresponding `skip_*` flag** (`normalize_params`).

| Sub-block | Builds | Notable keys |
|-----------|--------|--------------|
| `epack` | `EpackConfig` | `load` (load vs. IRL-solve), `save_evals`, `save_eigs` |
| `meson` | `MesonConfig` | `gamma` (list of gamma structures), `mass` (label list) |
| `high_modes` | `HighModeConfig` | per-operator (`vec_local`, `pion_local`, ...) → `mass` list (CG solves + deflation) |

Validation: if `epack` is skipped, `meson` must be skipped too (no eigenvectors → no meson fields).

---

## `job_setup.hadrons.tasks.epack` — `EpackConfig`

```yaml
job_setup:
  hadrons:
    tasks:
      epack:
        load: false      # true = load from file; false = run IRL solve
        save_evals: true
        save_eigs: true
```

| Key | Role |
|-----|------|
| `load` | `true` → load eigenvectors from the `eig` file; `false` → run IRL solve |
| `save_evals` | Save eigenvalues to the `eval` file |
| `save_eigs` | Save eigenvectors to the `eig` file |

`eigs`, `lanczos`, and the `eig`/`eigdir`/`eval` `Outfile`s come from `shared_params`/`hadrons_params`/`files`.

---

## `job_setup.hadrons.tasks.gauge` — `GaugeConfig` *(optional)*

The `gauge` sub-block is **optional**. When omitted, `GaugeConfig` defaults to
`action_type: load` — the thin gauge and pre-smeared `fat_links`/`long_links`
are loaded from disk (the behavior in the worked example). Add a `gauge` block
only when you want to change how the fat/long links are produced.

```yaml
job_setup:
  hadrons:
    tasks:
      gauge:
        action_type: smear   # load (default) | smear | free
        save_smear: true      # write on-the-fly links to fat/long_links paths
      epack:
        load: false
        save_evals: true
        save_eigs: true
      # ... meson, high_modes
```

`action_type` selects how the thin gauge and fat/long links are produced (they
are published under the same field names regardless):

- **`load`** *(default)* — thin gauge loaded from `gauge_links`; pre-smeared
  `fat_links`/`long_links` loaded from disk.
- **`smear`** — thin gauge loaded from `gauge_links`; fat/long links derived on
  the fly via `MGauge::HISQSmear`.
- **`free`** — unit thin gauge, smeared on the fly via `MGauge::HISQSmear`
  (no gauge file read).

```yaml
# HISQ-smear a loaded thin gauge on the fly, without persisting the links:
gauge:
  action_type: smear

# Free-field (unit gauge) test, smeared on the fly and saved:
gauge:
  action_type: free
  save_smear: true
```

| Key | Role | Consumed as |
|-----|------|-------------|
| `action_type` | `load` (default) / `smear` / `free`; how thin gauge + fat/long links are produced | `GaugeConfig.action_type` (`ActionType`) |
| `save_smear` | When `smear`/`free`, also write the on-the-fly fat/long links to the `fat_links`/`long_links` paths via `MIO::SaveIldg` | `GaugeConfig.save_smear` |
| `free` *(legacy)* | Legacy boolean; `true`→`action_type: free`, `false`→`load`. An explicit `action_type` always wins | Normalized to `action_type` |

`action_name`, `mass`, and the `gauge_links`/`fat_links`/`long_links` `Outfile`s
are supplied by the LMI parent + `files` (not set in this block). `save_smear`
only has an effect for `smear`/`free` (under `load` the links are already on
disk).

---

## `files` — file label catalog

```yaml
files:
  fat_links:
    filestem: lat/scidac/fat{ens}{series}.ildg
    good_size: 905900000
  eig:
    filestem: eigen/eig{ens}nv{eigs}{series}
    good_size: 37751000000
  eval:
    filestem: eigen/eval/eval{ens}nv{eigs}{series}
    good_size: 11000
  meson:
    filestem: e{eigs}n{noise}dt{dt}/mesons/m{mass}/mf_{series}
    good_size: 1536000000
  contract:
    filestem: e{eigs}n{noise}dt{dt}/correlators/m{mass}/{gamma_label}/a2aLL/corr_{gamma_label}_a2aLL_m{mass}_{series}
    good_size: 400
  # ... long_links, gauge_links, eigdir, high_modes follow the same pattern
```

The single source of truth for every input/output path. Each label becomes an `Outfile` object with a `filestem` (template) and a `good_size` (the byte threshold validation uses to judge completeness). Other sections refer to these by **label** (e.g. `outfile: contract`, `file: meson`).

| Label | Role |
|-------|------|
| `fat_links` / `long_links` / `gauge_links` | SciDAC/ILDG gauge field files |
| `eig` / `eigdir` | Eigenvector file / directory |
| `eval` | Eigenvalue file |
| `high_modes` | High-mode correlator output |
| `meson` | Meson-field output |
| `contract` | A2A correlator output |

Each entry:

| Sub-key | Role | Consumed as |
|---------|------|-------------|
| `filestem` | Path template with `{ens}`, `{series}`, `{eigs}`, `{mass}`, `{gamma_label}`, `{dset}`, `{tsource}` | `Outfile.filestem` (string template) |
| `good_size` | Minimum valid file size in bytes | `Outfile.good_size` → used by validator (`file_size >= good_size`) |

---

## Suggestions / Notes

- **Two-mental-models slide:** lead with the "template vs. config object" distinction — it's the key to reading every other section.
- **Merge-order slide:** the `shared_params → ... → tasks` cascade explains why a value set once at the top appears everywhere; worth a dedicated diagram.
- **Labels vs. paths:** emphasize that `outfile:`, `file:`, `evalfile:` are *labels* into `files:`, not paths. This trips people up.
- **Skip-by-omission:** the LMI `skip_*` behavior (omit a `tasks` sub-block to skip that stage) is non-obvious and deserves its own callout.
- **`good_size` is validation, not allocation:** clarify it's the completeness threshold the nanny uses to decide a file is "done", not a size hint.
- **Worth confirming:** I documented behavior from the configs in `pyfm/tasks/` and `pyfm/a2a/types.py`. The `sib`/`photex`/`selfen` contraction types and `cross_terms` exist in code but aren't exercised in this example file — flag whether your audience needs those covered.
