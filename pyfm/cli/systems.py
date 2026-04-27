import os
import subprocess
from pathlib import Path

import click

_SYSTEMS_DIR = Path(__file__).parent.parent.parent / "systems"


@click.group()
def build():
    pass


@build.command(name="run")
@click.option("--gmp", is_flag=True, default=False)
@click.option("--mpfr", is_flag=True, default=False)
@click.option("--lime", is_flag=True, default=False)
@click.option("--hdf5", is_flag=True, default=False)
@click.option("--ssl", is_flag=True, default=False)
@click.option("--qmp", is_flag=True, default=False)
@click.option("--qio", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--mpi-reduction", is_flag=True, default=False)
@click.option("--force", is_flag=True, default=False)
@click.option("--skip-make", is_flag=True, default=False)
@click.option("--grid", is_flag=True, default=False)
@click.option("--hadrons", is_flag=True, default=False)
@click.option("--hlma", is_flag=True, default=False)
@click.option("--glma", is_flag=True, default=False)
@click.option("--all", "all_components", is_flag=True, default=False)
@click.option("--system", type=str, default="scalar")
@click.option("--ext", type=str, default=None)
@click.option("--threads", type=int, default=4)
def build_run(
    gmp, mpfr, lime, hdf5, ssl, qmp, qio, debug, mpi_reduction, force,
    skip_make, grid, hadrons, hlma, glma, all_components, system, ext, threads,
):
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
    if all_components:
        args.append("--all")
    if ext is not None:
        args.extend(["--ext", ext])
    subprocess.run(args, check=True)


@click.group()
def workspace():
    pass


@workspace.command()
@click.option("--workspace", "workspace_dir", type=str, default=None)
@click.option("--storage", type=str, default=None)
@click.option("--scheduler", type=str, default=None)
@click.option("--lattice", type=str, default=None)
@click.option("--system", type=str, default=None)
def setup(workspace_dir, storage, scheduler, lattice, system):
    script = _SYSTEMS_DIR / "setup-workspace.sh"
    args = [str(script)]
    if workspace_dir is not None:
        args.extend(["--workspace", workspace_dir])
    if storage is not None:
        args.extend(["--storage", storage])
    if scheduler is not None:
        args.extend(["--scheduler", scheduler])
    if lattice is not None:
        args.extend(["--lattice", lattice])
    if system is not None:
        args.extend(["--system", system])
    subprocess.run(args, check=True)


@workspace.command()
@click.option("--system", required=True, type=str)
@click.option("--ext", type=str, default=None)
@click.option("--runtime-env", "runtime_env", type=str, default="true")
def env(system, ext, runtime_env):
    script = _SYSTEMS_DIR / "source-system-env.sh"
    source_cmd = f"source {script} --system {system}"
    if ext is not None:
        source_cmd += f" --ext {ext}"
    source_cmd += f" --runtime-env {runtime_env}"
    result = subprocess.run(
        ["bash", "-c", f"{source_cmd} && env"],
        capture_output=True,
        text=True,
        check=True,
    )
    current_env = os.environ
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if "BASH_FUNC" in key or key.startswith("_"):
            continue
        if current_env.get(key) != value:
            click.echo(f"export {key}={value}")
