import sys
import os
import subprocess

import typing as t
from pyfm import utils
import pandas as pd

from pyfm.nanny.config import (
    build_job_configs,
    get_nanny_config,
    JobConfig,
    NannyConfig,
    Scheduler,
)
from pyfm.nanny.taskbuilder import Task, create_task
import pyfm.nanny.todo as todo
from pyfm.domain.protocols import OutputComparisonProtocol, OutfileCatalogProtocol
from pyfm.tasks.hadrons import highmode


def get_outfiles(task: Task) -> pd.DataFrame | None:

    if isinstance(task.handler, OutfileCatalogProtocol):
        return task.handler.create_outfile_catalog(task.config)
    else:
        utils.get_logger().debug(
            "create_outfile_catalog not implemented for task. Skipping validation."
        )
        return None


def audit_outfiles(task: Task, verbose: bool = False) -> pd.DataFrame | None:
    logger = utils.get_logger()

    df = get_outfiles(task)
    MAX_FILES = 5

    if df is None:
        logger.warning(f"No output files given for task: {task.key}.")
        return df

    # Classify every row, then order so the interesting rows come first.
    exists = df["exists"].astype(bool)
    complete = exists & (df["file_size"] >= df["good_size"])
    status = pd.Series("complete", index=df.index)
    status[exists & ~complete] = "too small"
    status[~exists] = "missing"

    order = pd.Categorical(status, categories=["missing", "too small", "complete"])
    view = df.assign(status=status).iloc[order.argsort(kind="stable")]

    # verbose -> all rows; otherwise just the first MAX_FILES incomplete ones.
    if not verbose:
        view = view[view["status"] != "complete"]
    view = view.head(None if verbose else MAX_FILES)

    counts = status.value_counts().to_dict()
    logger.info(
        f"Outfile audit for {task.key}: "
        f"{counts.get('missing', 0)} missing, "
        f"{counts.get('too small', 0)} too small, "
        f"{counts.get('complete', 0)} complete"
    )
    if not view.empty:
        logger.info("\n" + view[["status", "filepath"]].to_string(index=False))
    if not verbose and (status != "complete").sum() > MAX_FILES:
        logger.info("...")

    return df


def compare_task_outputs(
    yaml_params: t.Dict,
    job_a: str,
    job_b: str,
    series: str,
    cfg: str,
    *,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> pd.DataFrame:
    """Compare the outputs of two jobs of the same task type.

    Four-step flow (mirrors the ``pyfm audit output`` contract):
      1. Verify the compared (high_modes) outputs exist (and meet good_size)
         for both jobs.
      2. Verify both jobs resolve to the same task type.
      3. Verify the task type has a comparison protocol registered.
      4. Run ``compare_outputs(config_a, config_b)`` and return the report.
    """
    task_a = create_task(job_a, yaml_params, series, cfg)
    task_b = create_task(job_b, yaml_params, series, cfg)

    # Step 1: the compared outputs (high_modes correlators) exist & are complete
    # for both jobs. Scoped to high_modes — the subconfig step 4 actually
    # compares — so unrelated epack/meson outputs don't gate the comparison.
    # (v1 coupling: hadrons_lmi exposes .high_modes_config; this generalizes
    # when other task types register compare_outputs.)
    for task, job in ((task_a, job_a), (task_b, job_b)):
        df = highmode.create_outfile_catalog(task.config.high_modes_config)
        if df is None or df.empty:
            raise ValueError(
                f"Job {job!r} ({task.key}) has no high_modes outfile catalog; cannot compare."
            )
        bad = df[
            (df["exists"] == False) | (df["file_size"].fillna(0) < df["good_size"])
        ]
        if not bad.empty:
            raise ValueError(
                f"Job {job!r} ({task.key}) is missing or has incomplete high_modes "
                f"outputs ({len(bad)} file(s) below threshold)."
            )

    # Step 2: same task type.
    if task_a.key != task_b.key:
        raise ValueError(
            f"Cannot compare jobs of different task types: "
            f"{job_a!r} -> {task_a.key!r} vs {job_b!r} -> {task_b.key!r}."
        )

    # Step 3: comparison protocol registered for this task type.
    handler = task_a.handler
    if not isinstance(handler, OutputComparisonProtocol):
        raise ValueError(
            f"Task type {task_a.key!r} has no comparison protocol registered "
            f"(compare_outputs not implemented for this task type)."
        )

    # Step 4: run the comparison.
    return handler.compare_outputs(task_a.config, task_b.config, rtol=rtol, atol=atol)


### Residual old code from Carleton
######################################################################
def job_still_queued(nanny_config: NannyConfig, job_id):
    """Get the status of the queued job"""
    # This code is locale dependent

    scheduler = nanny_config.scheduler

    user = os.environ["USER"]
    match scheduler:
        case Scheduler.LSF:
            cmd = " ".join(["bjobs", "-u", user, "|", "grep -w", job_id])
        case Scheduler.PBS:
            cmd = " ".join(["qstat", "-u", user, "|", "grep -w", job_id])
        case Scheduler.SLURM | Scheduler.INTERACTIVE:
            cmd = " ".join(["squeue", "-u", user, "|", "grep -w", job_id])
        case Scheduler.COBALT:
            cmd = " ".join(["qstat", "-fu", user, "|", "grep -w", job_id])
        case _:
            print("Don't recognize scheduler", scheduler)
            print("Quitting")
            sys.exit(1)

    # print(cmd)
    reply = ""
    try:
        reply = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        status = e.returncode
        # If status is other than 0 or 1, we have an squeue/bjobs problem
        # Treat job as unfinished
        if status != 1:
            print("ERROR", status, "Can't get the job status.  Skipping.")
            return True

    if len(reply) > 0:
        a = reply.decode().split()
        match scheduler:
            case Scheduler.LSF:
                # The start time
                if a[2] == "PEND":
                    time = "TBA"
                else:
                    time = a[5] + " " + a[6] + " " + a[7]
                field = "start"
                jobstat = a[2]
            case Scheduler.PBS:
                time = a[8]
                field = "queue"
                jobstat = a[9]
            case Scheduler.SLURM | Scheduler.INTERACTIVE:
                time = a[5]
                field = "run"
                jobstat = a[4]
            case Scheduler.COBALT:
                time = a[5]
                field = "run"
                jobstat = a[8]
            case _:
                print("Don't recognize scheduler", scheduler)
                print("Quitting")
                sys.exit(1)

        print("Job status", jobstat, field, "time", time)
        # If job is being canceled, jobstat = C (PBS).  Treat as finished.
        if jobstat == "C":
            return False
        else:
            return True

    return False


######################################################################


def next_finished(
    nanny_config: NannyConfig, todo_list, entry_list
) -> t.Tuple[int, str, str] | None:
    """Find the next well-formed entry marked "Q" whose job is no longer
    in the queue
    """
    logger = utils.get_logger()

    a = ()
    nskip = 0
    while len(entry_list) > 0:
        cfgno = entry_list.pop(0)
        a = todo_list[cfgno]
        if n := todo.find_next_queued_task(a):
            index, cfgno, step = n
            step = "_".join(step.split("_")[:-1])
        else:
            logger.debug(f"{cfgno} has no assigned tasks.")
            continue

        print("--------------------------------------------------------------")
        print("Checking cfg", todo_list[cfgno])
        print("--------------------------------------------------------------")

        # Is job still queued?
        job_id = a[index + 1]
        if job_still_queued(nanny_config, job_id):
            index = 0  # To signal no checking
            continue

        return index, cfgno, step

    return None


######################################################################
def has_good_output(task: Task) -> bool:
    df = audit_outfiles(task)
    if df is None or len(df) == 0:
        return False
    bad_file_mask = (df["exists"] == False) | (df["file_size"] < df["good_size"])
    has_good_files = df[bad_file_mask].empty
    if has_good_files:
        return True
    return False


######################################################################
def check_jobs(
    yaml_params: t.Dict,
    *,
    nanny_config: NannyConfig | None = None,
    job_configs: t.Dict[str, JobConfig] | None = None,
):
    """Process all entries marked Q in the todolist.

    ``nanny_config`` and ``job_configs`` may be pre-built by the caller (e.g. the
    nanny loop, which builds them once per iteration) to avoid rebuilds; when
    omitted (standalone callers) they are built from ``yaml_params``.
    """
    logger = utils.get_logger()

    if nanny_config is None:
        nanny_config = get_nanny_config(yaml_params)
    if job_configs is None:
        job_configs = build_job_configs(yaml_params)

    # Read the to-do file
    todo_file = nanny_config.todo_file
    lock_file = todo.lock_file_name(todo_file)

    # First, just get a list of entries
    todo.wait_set_todo_lock(lock_file)
    todo_list = todo.read_todo(todo_file)
    todo.remove_todo_lock(lock_file)
    entry_list = sorted(todo_list, key=todo.key_todo_entries)

    # Run through the entries. The entry_list is static, but the
    # to-do file could be changing due to other proceses
    while len(entry_list) > 0:
        # Reread the to-do file (it might have changed)
        todo.wait_set_todo_lock(lock_file)
        todo_list = todo.read_todo(todo_file)

        n = next_finished(nanny_config, todo_list, entry_list)
        if n is None:
            todo.remove_todo_lock(lock_file)
            continue

        index, cfgno, step = n

        # step = step[:-1]
        # Mark that we are checking this item and rewrite the to-do list
        todo_list[cfgno][index] = step + "_C"
        todo.write_todo(todo_file, todo_list)
        todo.remove_todo_lock(lock_file)

        if step not in yaml_params["job_setup"].keys():
            logger.error("Unrecognized step key", step)
            sys.exit(1)

        # Check that the job completed successfully
        series, cfg = cfgno.split(".")
        task = create_task(step, yaml_params, series, cfg)
        status = has_good_output(task)
        sys.stdout.flush()

        # Update the entry in the to-do file
        todo.wait_set_todo_lock(lock_file)
        todo_list = todo.read_todo(todo_file)
        if status:
            todo_list[cfgno][index] = f"{step}_X"
            logger.info(f"Job step {step} is COMPLETE")
        elif job_configs[step].barrier:
            todo_list[cfgno][index] = f"{step}_XXfix"
            logger.info("Marking todo entry XXfix.  Fix before rerunning.")
        else:
            todo_list[cfgno][index] = f"{step}_XXfixcont"
            logger.info(
                "Marking todo entry XXfixcont.  Non-blocking task failed; "
                "checking and submission will continue past it."
            )
        todo.write_todo(todo_file, todo_list)
        todo.remove_todo_lock(lock_file)

        # Take a cat nap (avoids hammering the login node)
        subprocess.check_call(["sleep", "1"])
    logger.info("Reached end of todo file. Exiting job checker.")
