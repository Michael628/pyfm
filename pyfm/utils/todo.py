# Scripts supporting job queue management
# spawnjob.py and checkjobs.py

# For Python 3 version

import typing as t
import sys
import os
import subprocess
import time

from pyfm import utils


######################################################################
def lock_file_name(todo_file):
    """Directory entry"""
    return todo_file + ".lock"


######################################################################
def wait_set_todo_lock(lock_file):
    """Set lock file"""

    while os.access(lock_file, os.R_OK):
        utils.get_logger().warn("Lock file present. Sleeping.")
        sys.stdout.flush()
        time.sleep(600)

    subprocess.call(["touch", lock_file])


######################################################################
def remove_todo_lock(lock_file):
    """Remove lock file"""
    subprocess.call(["rm", lock_file])


######################################################################
def read_todo(todo_file):
    """Read the todo file"""

    todo_list = dict()
    try:
        with open(todo_file) as todo:
            todo_lines = todo.readlines()
    except IOError:
        utils.get_logger().error(f"Can't open {todo_file}")
        sys.exit(1)

    for line in todo_lines:
        if len(line) == 1:
            continue
        a = line.split()
        for i in range(len(a)):
            if isinstance(a[i], bytes):
                a[i] = a[i].decode("ASCII")
        key = a[0]
        todo_list[key] = a

    todo.close()
    return todo_list


######################################################################
def key_todo_entries(td):
    """Sort key for todo entries with format x.nnnn"""

    (stream, cfg) = td.split(".")
    return "{0:s}{1:010d}".format(stream, int(cfg))


######################################################################
def write_todo(todo_file, todo_list):
    """Write the todo file"""

    # Back up the files
    subprocess.call(["mv", todo_file, todo_file + ".bak"])

    try:
        todo = open(todo_file, "w")

    except IOError:
        utils.get_logger().error(f"Can't open {todo_file} for writing")
        sys.exit(1)

    for line in sorted(todo_list, key=key_todo_entries):
        print(" ".join(todo_list[line]), file=todo)

    todo.close()


######################################################################
def find_next_task(
    line: list[str], condition_fn: t.Callable[[str], bool]
) -> t.Tuple | None:
    """Examine todo line to see if condition is met"""

    # Format
    # a.1170 SX 0 EX 2147965 LQ 2150955 A 0 H 0

    cfgno = line[0]

    for index, step in ((i, line[i]) for i in range(1, len(line), 2)):

        if condition_fn(step):
            return index, cfgno, step
        elif step.endswith("Q") or step.endswith("XXfix"):
            # Do not search past barrier designations
            return None

    return None


def find_next_unfinished_task(
    line: list[str], require_step: str | None = None
) -> t.Tuple | None:
    """Examine todo line looking for unfinished task that is ready to run."""

    # Format
    # a.1170 SX 0 EX 2147965 LQ 2150955 A 0 H 0

    skip_states = ["X", "XXfix", "Q", "Qcont", "C"]
    cond = lambda x: not any(x.endswith(state) for state in skip_states) and (
        require_step is None or x.startswith(require_step)
    )
    return find_next_task(line, cond)


######################################################################
def find_next_queued_task(a):
    """Examine todo line to see if there are any queued tasks"""

    # Format
    # a.1170 SX 0 EX 2147965 LQ 2150955 A 0 H 0

    cond = lambda x: x.endswith("Q") or x.endswith("Qcont")

    return find_next_task(line, cond)
