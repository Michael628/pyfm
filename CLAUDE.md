# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pip install -e .          # Editable install
python -m pytest          # Run all tests
python -m pytest test/path/to/test_file.py::test_function  # Single test
python -m pytest test/nanny/  # Test directory
```

## Architecture

PyFM is a lattice QCD workflow toolkit with three main subsystems:

1. **Task Handler** (`pyfm/tasks/`) - Generates input files for external programs (MILC, Hadrons), validates task completion, and aggregates output
2. **A2A Contraction Engine** (`pyfm/a2a/`) - Performs all-to-all contractions (2pt, 3pt, 4pt) with CPU/GPU support
3. **Nanny Job Manager** (`pyfm/nanny/`) - Manages automated HPC job submission (SLURM, PBS, LSF) via todo files

### Plugin-based Handler Registry

Tasks register themselves at module import time into a global registry (`pyfm/domain/registry.py`). Handler keys follow the format `"{module_scope}_{config_key}"` (e.g., `"hadrons_lmi"`, `"contract_diagram"`).

```python
# Registration pattern in task modules:
register_task(MyConfig, build_input_params, create_outfile_catalog, build_aggregator_params)
```

### Configuration System

All configs are **frozen Pydantic dataclasses** (`pyfm/domain/conftypes.py`):
- `SimpleConfig` - flat configuration
- `CompositeConfig` - contains nested subconfigs, introspected via `get_subconfigs()`

Config construction in `pyfm/core/builder.py` follows four ordered phases:
1. **Preprocess** - `preprocess_params()` transforms YAML dict before construction
2. **Construct** - `ConfigBuilder` recursively builds subconfigs, then top-level; `__post_init__` runs here (use only for computed properties, not validation)
3. **Postprocess** - `postprocess_config()` modifies the frozen config via `dataclasses.replace()`
4. **Validate** - `validate_config()` checks the final, complete config state

Use `validate_config` (not `__post_init__`) for all validation logic, since it runs after postprocessing on the complete config.

#### `_preprocessor` - parent-to-child context injection

`_preprocessor` is a routing table that a parent's `preprocess` hook injects into `params` to pass context down to child configs during recursive construction. It is **never** a field on any config class -- it exists only during the build traversal.

**The contract:** the parent curates what each child receives; the child's own `preprocess` hook decides how to apply it. The default behaviour is to merge `_preprocessor` contents directly into the child's params, but a child hook may handle conflicts differently. **Never bypass the child's preprocess hook by injecting data directly into its params** -- that removes the child's ability to control the merge.

Structure by container type:
- **SIMPLE:** `_preprocessor[field_name]` is a dict passed as the child's `_preprocessor`
- **LIST:** `_preprocessor[field_name]` is a list; each element is passed as the corresponding child's `_preprocessor`
- **DICT:** `params[field_name]` maps keys to per-child param dicts; `_preprocessor[field_name]` maps those same keys to per-child preprocessor dicts

See `build_composite_config()` in `pyfm/core/builder.py` and `lmi.py`'s `preprocess_params` for the canonical example.

### Adding a New Task

See `pyfm/tasks/hadrons/lmi.py` as the exemplary implementation. The minimal pattern:

```python
@dataclass(frozen=True)
class MyTaskConfig(CompositeConfig):
    key: t.ClassVar[str] = "hadrons_mytask"
    some_param: str

def build_input_params(config: MyTaskConfig): ...
def create_outfile_catalog(config: MyTaskConfig) -> pd.DataFrame: ...
def build_aggregator_params(config: MyTaskConfig, average: bool) -> Dict: ...

register_task(MyTaskConfig, build_input_params, create_outfile_catalog, build_aggregator_params,
              preprocess_params=...,   # optional
              postprocess_config=...,  # optional
              validate=validate_config) # optional
```

Import and register in the module's `__init__.py` so it's picked up at import time.

### Data Processing Pipeline

`pyfm/dataio/processor.py` transforms DataFrames through ordered actions defined in `ACTION_ORDER`. To add a new action: implement a function with signature `(df, data_col, *args, **kwargs) -> pd.DataFrame`, add it to `ACTION_ORDER`, and invoke it via YAML `actions: {my_action: params}`.

### Key Patterns

```python
# Config modification (configs are frozen)
from dataclasses import replace
new_config = replace(old_config, field=new_value)

# Handler retrieval
from pyfm.tasks.register import get_task_handler
handler = get_task_handler(job_type="hadrons", task_type="lmi")

# Config building from YAML
from pyfm.core.builder import build_config
config = build_config(ConfigType, yaml_params, file_params)

# String template formatting (unmatched keys left as literals)
path = config.format_string("/path/{series}/{cfg}")
```

## Coding Conventions

- Python 3.12+
- Imports: stdlib → third-party → local
- PascalCase for classes, snake_case for functions/variables, `_name` for private
- Pydantic dataclasses with `frozen=True` for all config classes
- Handler methods automatically receive `config` as first param when their signature includes it

## Common Gotchas

- Registry is **global** - registration happens at import; missing imports = missing handlers
- `CompositeConfig` subconfigs are built **recursively** - field order matters
- `PartialFormatter` leaves unresolved `{key}` as **literals** (no KeyError)
- Adding a new task to `pyfm/tasks/hadrons/` also requires adding it to `pyproject.toml` packages list if it's a new subdirectory
