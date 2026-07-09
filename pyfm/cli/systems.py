import os
import re
import shlex
import subprocess
from pathlib import Path

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

import click

_SYSTEMS_DIR = Path(__file__).parent.parent.parent / "systems"


@click.group()
def build():
    """Compile lattice QCD software components (Grid, Hadrons, LMA)."""
    pass


@build.command(name="run")
@click.option("--gmp", is_flag=True, default=False, help="Build GMP arbitrary-precision library.")
@click.option("--mpfr", is_flag=True, default=False, help="Build MPFR floating-point library.")
@click.option("--lime", is_flag=True, default=False, help="Build LIME I/O library.")
@click.option("--hdf5", is_flag=True, default=False, help="Build HDF5 data format library.")
@click.option("--ssl", is_flag=True, default=False, help="Build OpenSSL library.")
@click.option("--qmp", is_flag=True, default=False, help="Build QMP message-passing library.")
@click.option("--qio", is_flag=True, default=False, help="Build QIO lattice I/O library.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug build flags.")
@click.option("--mpi-reduction", is_flag=True, default=False, help="Enable MPI reduction optimizations.")
@click.option("--force", is_flag=True, default=False, help="Force rebuild even if already built.")
@click.option("--skip-make", is_flag=True, default=False, help="Skip the make step (configure only).")
@click.option("--grid", is_flag=True, default=False, help="Build the Grid library.")
@click.option("--hadrons", is_flag=True, default=False, help="Build the Hadrons framework.")
@click.option("--hlma", is_flag=True, default=False, help="Build the HLMA (hadronic LMA) module.")
@click.option("--glma", is_flag=True, default=False, help="Build the GLMA (grid LMA) module.")
@click.option("--all", "all_components", is_flag=True, default=False, help="Build all components.")
@click.option("--system", type=str, default="scalar", help="Target system profile name (default: scalar).")
@click.option("--ext", type=str, default=None, help="Extension/variant tag for the system profile.")
@click.option("--threads", type=int, default=4, help="Number of parallel make threads (default: 4).")
@click.option("--old-rng", is_flag=True, default=False, help="Use Grid's old rng population algorithm.")
def build_run(
    gmp,
    mpfr,
    lime,
    hdf5,
    ssl,
    qmp,
    qio,
    debug,
    mpi_reduction,
    force,
    skip_make,
    grid,
    hadrons,
    hlma,
    glma,
    all_components,
    system,
    ext,
    threads,
    old_rng,
):
    """Build selected software components for the given system profile."""
    script = _SYSTEMS_DIR / "build.sh"
    args = [str(script), "--system", system, "--threads", str(threads)]
    if gmp:
        args.append("--gmp")
    if mpfr:
        args.append("--mpfr")
    if lime:
        args.append("--lime")
    if hdf5:
        args.append("--hdf5")
    if ssl:
        args.append("--ssl")
    if qmp:
        args.append("--qmp")
    if qio:
        args.append("--qio")
    if debug:
        args.append("--debug")
    if mpi_reduction:
        args.append("--mpi-reduction")
    if force:
        args.append("--force")
    if skip_make:
        args.append("--skip-make")
    if grid:
        args.append("--grid")
    if hadrons:
        args.append("--hadrons")
    if hlma:
        args.append("--hlma")
    if glma:
        args.append("--glma")
    if old_rng:
        args.append("--old-rng")
    if all_components:
        args.append("--all")
    if ext is not None:
        args.extend(["--ext", ext])
    subprocess.run(args, check=True)


@click.group()
def workspace():
    """Initialize and configure a PyFM workspace environment."""
    pass


@workspace.command()
@click.option("--workspace", "workspace_dir", type=click.Path(file_okay=False), default=None, help="Workspace root directory path.")
@click.option("--storage", type=click.Path(file_okay=False), default=None, help="Storage root directory for job data.")
@click.option("--scheduler", type=str, default=None, help="HPC scheduler type (slurm, pbs, lsf).")
@click.option("--lattice", type=str, default=None, help="Lattice geometry descriptor.")
@click.option("--system", type=str, default=None, help="System profile name to configure.")
def setup(workspace_dir, storage, scheduler, lattice, system):
    """Set up a new PyFM workspace with directory structure and config files."""
    script = _SYSTEMS_DIR / "setup-workspace.sh"
    args = [str(script)]
    if workspace_dir is not None:
        p = Path(workspace_dir)
        if not p.is_absolute():
            if not p.exists():
                raise click.BadParameter(f"Directory does not exist: {workspace_dir}", param_hint="'--workspace'")
            workspace_dir = str(p.resolve())
        args.extend(["--workspace", workspace_dir])
    if storage is not None:
        p = Path(storage)
        if not p.is_absolute():
            if not p.exists():
                raise click.BadParameter(f"Directory does not exist: {storage}", param_hint="'--storage'")
            storage = str(p.resolve())
        args.extend(["--storage", storage])
    if scheduler is not None:
        args.extend(["--scheduler", scheduler])
    if lattice is not None:
        args.extend(["--lattice", lattice])
    if system is not None:
        args.extend(["--system", system])
    result = subprocess.run(args)
    if result.returncode != 0:
        raise click.exceptions.Exit(result.returncode)


@workspace.command()
@click.option("--system", required=True, type=str, help="System profile name to source (required).")
@click.option("--ext", type=str, default=None, help="Extension/variant tag for the system profile.")
@click.option("--runtime-env", "runtime_env", type=str, default="true", help="Include runtime environment variables (default: true).")
def env(system, ext, runtime_env):
    """Print shell export statements for the system environment.

    Intended to be eval'd in the shell: eval $(pyfm workspace env --system <name>)
    """
    script = _SYSTEMS_DIR / "source-system-env.sh"
    source_cmd = f"source {script} --system {system}"
    if ext is not None:
        source_cmd += f" --ext {ext}"
    source_cmd += f" --runtime-env {runtime_env}"
    result = subprocess.run(
        ["bash", "-c", f"{{ {source_cmd}; }} 1>&2 && env -0"],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    current = os.environ
    for item in result.stdout.split("\0"):
        if "=" not in item:
            continue
        key, _, value = item.partition("=")
        if not _IDENTIFIER.match(key):
            continue
        if current.get(key) == value:
            continue
        click.echo(f"export {key}={shlex.quote(value)}")
