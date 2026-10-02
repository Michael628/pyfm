import json

import click

from pyfm import utils
from pyfm.performance import analyze_file, benchmark_lmi_performance
from pyfm.nanny.validator import compare_task_outputs
from pyfm.version import parse_run_log
from pyfm.cli._options import job_option, param_file_option


@click.group()
def audit():
    """Audit runtime performance output logs."""
    pass


@audit.command(name="runtime")
@click.argument(
    "output_file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
def runtime(output_file):
    """Print a runtime performance summary for a Hadrons OUTPUT_FILE."""
    try:
        analyze_file(output_file)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@audit.command(name="benchmark")
@job_option(required=True, help="Job step name.")
@click.option(
    "--log",
    "log_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Hadrons/Grid performance log file.",
)
@param_file_option()
def benchmark(job, log_file, param_file):
    """Emit component-first JSON benchmark data for a configured Hadrons/Grid LMI LOG."""
    try:
        params = utils.io.load_param(param_file)
        result = benchmark_lmi_performance(job, log_file, params)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(json.dumps(result, sort_keys=True))


@audit.command(name="output")
@param_file_option()
@click.option(
    "-j",
    "--job",
    "jobs",
    type=str,
    nargs=2,
    required=True,
    help="Two job step names to compare (e.g. -j baseline rerun).",
)
@click.option(
    "-s", "--series", type=str, required=True, help="Gauge field series label."
)
@click.option(
    "-n", "--config", "cfg", type=str, required=True, help="Configuration number."
)
@click.option(
    "--rtol",
    type=float,
    default=1e-9,
    show_default=True,
    help="Relative tolerance (np.allclose-style).",
)
@click.option(
    "--atol",
    type=float,
    default=1e-12,
    show_default=True,
    help="Absolute tolerance (np.allclose-style).",
)
@click.pass_context
def output(ctx, param_file, jobs, series, cfg, rtol, atol):
    """Compare the outputs of two jobs of the same task type.

    Checks that all outputs exist for both jobs, that the jobs share a task
    type, that the task type supports output comparison, then runs
    compare_outputs and reports per-file max abs/rel differences. Exits
    non-zero if any compared file is outside tolerance or a precondition fails.
    """
    job_a, job_b = jobs
    try:
        params = utils.io.load_param(param_file)
        report = compare_task_outputs(
            params, job_a, job_b, series, cfg, rtol=rtol, atol=atol
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    click.echo(report.to_string(index=False))

    compared = report[report["status"] == "compared"]
    if compared.empty:
        click.echo("No files were compared (one or both jobs are missing outputs).")
        ctx.exit(1)

    n_out = int((compared["within_tolerance"] == False).sum())
    if n_out == 0:
        click.echo(f"All {len(compared)} compared file(s) within tolerance.")
    else:
        click.echo(f"{n_out} of {len(compared)} compared file(s) OUTSIDE tolerance.")
        ctx.exit(1)


@audit.command(name="version")
@click.argument(
    "log_file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.pass_context
def version_check(ctx, log_file):
    """Check the HadronsMILC version reported in a Hadrons run LOG_FILE.

    Compares the binary's MAJOR.MINOR with the hadronsMilcCompat recorded in the
    log's Provenance line (falling back to this pyfm's HADRONS_MILC_COMPAT).
    Exits non-zero on a mismatch; logs from binaries without a version banner
    report "unknown" and exit zero.
    """
    with open(log_file, errors="replace") as f:
        info = parse_run_log(f.read())

    fields = [
        ("hadronsMilcVersion", info.hadrons_milc_version),
        ("hadronsMilcGit", info.hadrons_milc_git),
        ("gridGit", info.grid_git),
        ("hadronsGit", info.hadrons_git),
        ("gridMilcVersion", info.grid_milc_version),
        ("dependencyPins", info.dependency_pins),
    ]
    for key, value in fields:
        click.echo(f"{key}: {value or 'unknown'}")
    if info.provenance is not None:
        for key, value in info.provenance.items():
            click.echo(f"provenance.{key}: {value or 'unknown'}")

    status = info.compat_status()
    binary = info.hadrons_milc_version or "unknown"
    click.echo(f"compat: {status} (expected {info.expected_compat()}, binary {binary})")
    if status == "mismatch":
        ctx.exit(1)
