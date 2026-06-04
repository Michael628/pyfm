import sys
import os
import subprocess

import typing as t
from pyfm import utils
import pandas as pd

from pyfm.nanny.core import create_task, get_nanny_config, NannyConfig, Scheduler, Task
import pyfm.nanny.todo as todo


@t.runtime_checkable
class TaskOutputProtocol(t.Protocol):
    def create_outfile_catalog(self) -> pd.DataFrame:
        """Creates a dataframe of information on all output files for task including
        whether the file exists, the size of the file, and whether it matches the expected size.
        """
        ...


def get_outfiles(task: Task) -> pd.DataFrame | None:

    if isinstance(task.handler, TaskOutputProtocol):
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
    output_count = 0

    if df is not None:
        for i, row in df.iterrows():
            if not row["exists"]:
                logger.info(f"File {row['filepath']} does not exist")
            elif not row["file_size"] >= row["good_size"]:
                logger.info(f"File {row['filepath']} is not complete")
            elif verbose:
                logger.info(f"File {row['filepath']} is complete")
            else:
                continue

            output_count += 1

            if output_count >= MAX_FILES:
                logger.info("...")
                break
    else:
        logger.warn(f"No output files given for task: {task.key}.")

    return df


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
    bad_file_mask = (df["exists"] == False) | (df["file_size"] < df["good_size"])
    has_good_files = df is not None and df[bad_file_mask].empty
    if has_good_files:
        return True
    return False


######################################################################
def check_jobs(yaml_params: t.Dict):
    """Process all entries marked Q in the todolist"""

    logger = utils.get_logger()

    nanny_config = get_nanny_config(yaml_params)

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
        else:
            todo_list[cfgno][index] = f"{step}_XXfix"
            logger.info("Marking todo entry XXfix.  Fix before rerunning.")
        todo.write_todo(todo_file, todo_list)
        todo.remove_todo_lock(lock_file)

        # Take a cat nap (avoids hammering the login node)
        subprocess.check_call(["sleep", "1"])
    logger.info("Reached end of todo file. Exiting job checker.")
