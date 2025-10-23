import sys
import os
import subprocess


import typing as t
from pyfm import utils
import pandas as pd

from pyfm.nanny.setup import create_task


@t.runtime_checkable
class TaskOutputProtocol(t.Protocol):
    def create_outfile_catalog(self) -> pd.DataFrame:
        """Creates a dataframe of information on all output files for task including
        whether the file exists, the size of the file, and whether it matches the expected size.
        """
        ...


def get_outfiles(
    job_step: str, yaml_data: t.Dict, series: str, cfg: str
) -> pd.DataFrame:
    task = create_task(job_step, yaml_data, series, cfg)
    if isinstance(task, TaskOutputProtocol):
        return task.create_outfile_catalog()
    else:
        utils.get_logger().warning(
            "create_outfile_catalog not implemented for task. Skipping validation."
        )
        return pd.DataFrame()


def print_file_audit(
    job_step: str, yaml_data: t.Dict, series: str, cfg: str, verbose: bool = False
) -> pd.DataFrame:
    df = get_outfiles(job_step, yaml_data, series, cfg)

    for _, row in df.iterrows():
        if not row["exists"]:
            print(f"File {row['filepath']} does not exist")
        elif not row["file_size"] >= row["good_size"]:
            print(f"File {row['filepath']} is not complete")
        elif verbose:
            print(f"File {row['filepath']} is complete")

    return df


def get_bad_files(df: pd.DataFrame) -> t.List[str]:
    df = df[df["exists"] != True] or df[df["file_size"] < df["good_size"]]
    return list(df["filepath"])


def has_good_output(df: pd.DataFrame) -> bool:
    return len(get_bad_files(df)) == 0


### Residual old code from Carleton
######################################################################
def job_still_queued(param, job_id):
    """Get the status of the queued job"""
    # This code is locale dependent

    scheduler = param["submit"]["scheduler"]

    user = os.environ["USER"]
    if scheduler == "LSF":
        cmd = " ".join(["bjobs", "-u", user, "|", "grep -w", job_id])
    elif scheduler == "PBS":
        cmd = " ".join(["qstat", "-u", user, "|", "grep -w", job_id])
    elif scheduler == "SLURM":
        cmd = " ".join(["squeue", "-u", user, "|", "grep -w", job_id])
    elif scheduler == "INTERACTIVE":
        cmd = " ".join(["squeue", "-u", user, "|", "grep -w", job_id])
    elif scheduler == "Cobalt":
        cmd = " ".join(["qstat", "-fu", user, "|", "grep -w", job_id])
    else:
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
        if scheduler == "LSF":
            # The start time
            if a[2] == "PEND":
                time = "TBA"
            else:
                time = a[5] + " " + a[6] + " " + a[7]
            field = "start"
            jobstat = a[2]
        elif scheduler == "PBS":
            time = a[8]
            field = "queue"
            jobstat = a[9]
        elif scheduler == "SLURM":
            time = a[5]
            field = "run"
            jobstat = a[4]
        elif scheduler == "INTERACTIVE":
            time = a[5]
            field = "run"
            jobstat = a[4]
        elif scheduler == "Cobalt":
            time = a[5]
            field = "run"
            jobstat = a[8]
        else:
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


def next_finished(param, todo_list, entry_list):
    """Find the next well-formed entry marked "Q" whose job is no longer
    in the queue
    """
    a = ()
    nskip = 0
    while len(entry_list) > 0:
        cfgno = entry_list.pop(0)
        a = todo_list[cfgno]
        index, cfgno, step = utils.todo.find_next_queued_task(a)
        if index == 0:
            continue

        if step == "":
            nskip = 5

        # Skip entries to Avoid collisions with other check-completed processes
        if nskip > 0:
            nskip -= 1  # Count down from nonzero
            a = ()
            continue

        print("--------------------------------------------------------------")
        print("Checking cfg", todo_list[cfgno])
        print("--------------------------------------------------------------")

        # Is job still queued?
        job_id = a[index + 1]
        if job_still_queued(param, job_id):
            index = 0  # To signal no checking
            continue
        break

    return index, cfgno, step


######################################################################
def good_output(step: str, cfgno: str, param: t.Dict) -> bool:
    (series, cfg) = cfgno.split(".")

    df = get_outfiles(step, param, series, cfg)
    # Use the new JobConfig method for cleaner code
    if has_good_output(df):
        return True
    else:
        bad_files = get_bad_files(df)
        if bad_files:
            utils.get_logger().warning(
                f"File `{bad_files[0]}` not found or not of correct file size."
            )
        return False


######################################################################
def check_jobs(YAML):
    """Process all entries marked Q in the todolist"""

    # Read primary parameter file
    param = utils.io.load_param(YAML)

    # Read the to-do file
    todo_file = param["nanny"]["todo_file"]
    lock_file = utils.todo.lock_file_name(todo_file)

    # First, just get a list of entries
    utils.todo.wait_set_todo_lock(lock_file)
    todo_list = utils.todo.read_todo(todo_file)
    utils.todo.remove_todo_lock(lock_file)
    entry_list = sorted(todo_list, key=utils.todo.key_todo_entries)

    # Run through the entries. The entry_list is static, but the
    # to-do file could be changing due to other proceses
    while len(entry_list) > 0:
        # Reread the to-do file (it might have changed)
        utils.todo.wait_set_todo_lock(lock_file)
        todo_list = utils.todo.read_todo(todo_file)

        index, cfgno, step = next_finished(param, todo_list, entry_list)
        if index == 0:
            utils.todo.remove_todo_lock(lock_file)
            continue

        step = step[:-1]
        # Mark that we are checking this item and rewrite the to-do list
        todo_list[cfgno][index] = step + "C"
        utils.todo.write_todo(todo_file, todo_list)
        utils.todo.remove_todo_lock(lock_file)

        if step not in param["job_setup"].keys():
            print("ERROR: unrecognized step key", step)
            sys.exit(1)

        # Check that the job completed successfully
        status = good_output(step, cfgno, param)
        sys.stdout.flush()

        # Update the entry in the to-do file
        utils.todo.wait_set_todo_lock(lock_file)
        todo_list = utils.todo.read_todo(todo_file)
        if status:
            todo_list[cfgno][index] = step + "X"
            print("Job step", step, "is COMPLETE")
        else:
            todo_list[cfgno][index] = step + "XXfix"
            print("Marking todo entry XXfix.  Fix before rerunning.")
        utils.todo.write_todo(todo_file, todo_list)
        utils.todo.remove_todo_lock(lock_file)

        # Take a cat nap (avoids hammering the login node)
        subprocess.check_call(["sleep", "1"])
