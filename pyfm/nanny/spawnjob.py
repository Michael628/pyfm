#! /usr/bin/env python3

# Python 3 version

import yaml
import sys
import os
import subprocess
import typing as t
from pyfm.nanny import config, todo_utils, checkjobs, tasks
from pyfm import utils

from functools import reduce

from dict2xml import dict2xml as dxml

from pyfm.nanny.tasks.contract import SubmitContractConfig, ContractTask
from pyfm.nanny.tasks.hadrons.components import hadmods
from pyfm.nanny.tasks.hadrons import SubmitHadronsConfig

# Nanny script for managing job queues
# C. DeTar 7/11/2022

# Usage

# From the ensemble directory containing the to-do file
# ../scripts/spawnjob.py

# Requires a to-do file with a list of configurations to be processed

# The to-do file contains the list of jobs to be done.
# The line format of the to-do file is
# <cfgno> <task code+flag> <task jobid> <task code+flag> <task jobid> etc.
# Example: a.1170 SX 0 EX 2147965 LQ 2150955 A 0 H 0
# Where cfgno is tne configuration number in the format x.nnn where x is the
# series letter and nnn is the configuration number in the series
# The second letter in the task code and flag is the flag
# The task codes are
# S links smearing job
# E eigenvector generation job
# L LMA job
# M meson job
# A A2A + meson job
# H contraction local job
# I contraction one-link job
# They must be run in this sequence (but L and A can be concurrent)
# If flag is "X", the job has been finished
# If it is "Q", the job was queued with the given <jobid>
# If it is "C", the job is finished and is undergoing checking and tarring
# If the flag letter is empty, the job needs to be run

# Requires Todo_utils.py and params-launch.yaml with definitions
# of variables needed here


######################################################################
def count_queue(scheduler, myjob_name_pfx):
    """Count my jobs in the queue"""

    user = os.environ["USER"]

    if scheduler == "LSF":
        cmd = " ".join(
            ["bjobs -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
        )
    elif scheduler == "PBS":
        cmd = " ".join(
            ["qstat -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
        )
    elif scheduler == "SLURM":
        cmd = " ".join(
            ["squeue -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
        )
    elif scheduler == "INTERACTIVE":
        cmd = " ".join(
            ["squeue -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
        )
    elif scheduler == "Cobalt":
        cmd = " ".join(
            ["qstat -fu", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
        )
    else:
        print("Don't recognize scheduler", scheduler)
        print("Quitting")
        sys.exit(1)

    nqueued = int(subprocess.check_output(cmd, shell=True))

    return nqueued


######################################################################
def next_cfgno_steps(max_cases, todo_list):
    """Get next sets of cfgnos / job steps from the to-do file"""

    # Return a list of cfgnos and indices to be submitted in the next job
    # All subjobs in a single job must do the same step

    step = "none"
    cfgno_steps = []
    for line in sorted(todo_list, key=todo_utils.key_todo_entries):
        a = todo_list[line]
        if len(a) < 2:
            print("ERROR: bad todo line format")
            print(a)
            sys.exit(1)

        index, cfgno, new_step = todo_utils.find_next_unfinished_task(a)
        if index > 0:
            if step == "none":
                step = new_step
            elif step != new_step:
                # Ensure only one step per job
                break
            cfgno_steps.append([cfgno, index])
            # We don't bundle the S (links) or H (contraction) steps
            if step in ["S", "H", "I"]:
                break
        # Stop when we have enough for a bundle
        if len(cfgno_steps) >= max_cases:
            break

    ncases = len(cfgno_steps)

    if ncases > 0:
        print("Found", ncases, "cases...", cfgno_steps)
        sys.stdout.flush()

    return step, cfgno_steps


######################################################################
def make_inputs(param, step, cfgno_steps):
    """Create input XML files for this job"""

    ncases = len(cfgno_steps)
    input_files = []

    for i in range(ncases):
        (cfgno_series, _) = cfgno_steps[i]

        # Extract series and cfgno  a.1428 -> a 1428
        (series, cfgno) = cfgno_series.split(".")

        job_config = config.get_job_config(param, step)
        submit_config = config.get_submit_config(
            param, job_config, series=series, cfg=cfgno
        )

        os.environ["ENS"] = submit_config.ens

        input_file: str = job_config.get_infile(submit_config)

        # TODO: Move input file creation into runio module
        input_params, schedule = config.input_params(job_config, submit_config)

        if job_config.job_type == "contract":
            input_string = yaml.dump(input_params)
        elif job_config.job_type == "hadrons":
            assert isinstance(submit_config, SubmitHadronsConfig)

            if schedule:
                os.makedirs("schedules/", exist_ok=True)
                sched_file = f"schedules/{input_file[: -len('.xml')]}.sched"
                with open(sched_file, "w") as f:
                    f.write(str(len(schedule)) + "\n" + "\n".join(schedule))
            else:
                sched_file = ""

            xml_dict = hadmods.xml_wrapper(
                runid=submit_config.run_id, sched=sched_file, cfg=submit_config.cfg
            )

            xml_dict["grid"]["modules"] = {"module": input_params}
            input_string = dxml(xml_dict)
        else:
            input_string = input_params

        os.makedirs("in/", exist_ok=True)
        with open(f"in/{input_file}", "w") as f:
            f.write(input_string)

        input_files.append(input_file)

    os.environ["INPUTLIST"] = " ".join(input_files)


######################################################################
def submit_job(param, step, cfgno_steps, max_cases):
    """Submit the job"""

    ncases = len(cfgno_steps)

    job_script = param["job_setup"][step]["run"]
    wall_time = param["job_setup"][step]["wall_time"]

    layout = param["submit"]["layout"]
    basenodes = layout[step]["nodes"]
    ppj = reduce((lambda x, y: x * y), layout[step]["geom"])
    ppn = layout["ppn"] if "ppn" not in layout[step].keys() else layout[step]["ppn"]
    jpn = int(ppn / ppj)
    basetasks = basenodes * ppn if basenodes > 1 or jpn <= 1 else ppj
    nodes = (
        basenodes * ncases if jpn <= 1 else int((basenodes * ncases + jpn - 1) / jpn)
    )
    NP = str(nodes * ppn)
    geom = ".".join([str(i) for i in layout[step]["geom"]])

    # Append the number of cases to the step tag, as in A -> A3
    job_name = param["submit"]["job_name_pfx"] + "-" + step + str(ncases)
    os.environ["NP"] = NP
    os.environ["PPN"] = str(ppn)
    os.environ["PPJ"] = str(ppj)
    os.environ["BASETASKS"] = str(basetasks)
    os.environ["BASENODES"] = str(basenodes)
    os.environ["LAYOUT"] = geom

    # Check that the job script exists
    try:
        stat = os.stat(job_script)
    except OSError:
        print("Can't find the job script:", job_script)
        print("Quitting")
        sys.exit(1)

    # Job submission command depends on locale
    scheduler = param["submit"]["scheduler"]
    if scheduler == "LSF":
        cmd = f"bsub -nnodes {str(nodes)} -J {job_name} {job_script}"
    elif scheduler == "PBS":
        cmd = f"qsub -l nodes={str(nodes)} -N {job_name} {job_script}"
    elif scheduler == "SLURM":
        # NEEDS UPDATING
        cmd = (
            f"sbatch -N {str(nodes)} -n {NP} -J {job_name} -t {wall_time} {job_script}"
        )
    elif scheduler == "INTERACTIVE":
        cmd = f"./{job_script}"
    # elif scheduler == 'Cobalt':
    # NEEDS UPDATING IF WE STILL USE Cobalt
    # cmd = (f"qsub -n {str(nodes)} --jobname {job_name} {archflags}"
    #       f"--mode script --env LATS={LATS}:NCASES={NCASES}"
    #       f":NP={NP} {job_script}")
    else:
        print("Don't recognize scheduler", scheduler)
        print("Quitting")
        sys.exit(1)

    # Run the job submission command
    print(cmd)
    reply = ""
    try:
        reply = subprocess.check_output(cmd, shell=True).decode().splitlines()
    except subprocess.CalledProcessError as e:
        print("\n".join(reply))
        print("Job submission error.  Return code", e.returncode)
        print("Quitting")
        sys.exit(1)

    print("\n".join(reply))

    # Get job ID
    if scheduler == "LSF":
        # a.2100 Q Job <99173> is submitted to default queue <batch>
        jobid = reply[0].split()[1].split("<")[1].split(">")[0]
        if isinstance(jobid, bytes):
            jobid = jobid.decode("ASCII")
    elif scheduler == "PBS":
        # 3314170.kaon2.fnal.gov submitted
        jobid = reply[0].split(".")[0]
    elif scheduler == "SLURM":
        # Submitted batch job 10059729
        jobid = reply[len(reply) - 1].split()[3]
    elif scheduler == "INTERACTIVE":
        jobid = os.environ["SLURM_JOBID"]
    elif scheduler == "Cobalt":
        # ** Project 'semileptonic'; job rerouted to queue 'prod-short'
        # ['1607897']
        jobid = reply[-1]
    if isinstance(jobid, bytes):
        jobid = jobid.decode("ASCII")

    cfgnos = ""
    for cfgno, index in cfgno_steps:
        cfgnos = cfgnos + cfgno
    date = subprocess.check_output("date", shell=True).rstrip().decode()
    print(date, "Submitted job", jobid, "for", cfgnos, "step", step)

    return (0, jobid)


######################################################################
def mark_queued_todo_entries(step, cfgno_steps, jobid, todo_list):
    """Update the todo_file, change status to "Q" and mark the job number"""

    for k in range(len(cfgno_steps)):
        c, i = cfgno_steps[k]

        todo_list[c][i] = step + "Q"
        todo_list[c][i + 1] = jobid


######################################################################
def nanny_loop(YAML):
    """Check job periodically and submit to the queue"""

    date = subprocess.check_output("date", shell=True).rstrip().decode()
    hostname = subprocess.check_output("hostname", shell=True).rstrip().decode()
    print(date, "Spawn job process", os.getpid(), "started on", hostname)
    sys.stdout.flush()

    param = utils.load_param(YAML)

    # Keep going until
    #   we see a file called "STOP" OR
    #   we have exhausted the list OR
    #   there are job submission or queue checking errors

    check_count = int(param["nanny"]["check_interval"])
    while True:
        if os.access("STOP", os.R_OK):
            print("Spawn job process stopped because STOP file is present")
            break

        todo_file = param["nanny"]["todo_file"]
        max_cases = param["nanny"]["max_cases"]
        job_name_pfx = param["submit"]["job_name_pfx"]
        scheduler = param["submit"]["scheduler"]

        lock_file = todo_utils.lock_file_name(todo_file)

        # Count queued jobs with our job name
        nqueued = count_queue(scheduler, job_name_pfx)

        # Submit until we have the desired number of jobs in the queue
        if nqueued < param["nanny"]["max_queue"]:
            todo_utils.wait_set_todo_lock(lock_file)
            todo_list = todo_utils.read_todo(todo_file)
            todo_utils.remove_todo_lock(lock_file)

            # List a set of cfgnos
            step, cfgno_steps = next_cfgno_steps(max_cases, todo_list)
            ncases = len(cfgno_steps)

            # Check completion and purge scratch files for complete jobs
            if check_count == 0:
                checkjobs.main()
                check_count = int(param["nanny"]["check_interval"])

            if ncases > 0:
                # Make input
                make_inputs(param, step, cfgno_steps)

                # Submit the job

                status, jobid = submit_job(param, step, cfgno_steps, max_cases)

                # Job submissions succeeded
                # Edit the todo_file, marking the lattice queued and
                # indicating the jobid
                if status == 0:
                    todo_utils.wait_set_todo_lock(lock_file)
                    todo_list = todo_utils.read_todo(todo_file)
                    mark_queued_todo_entries(step, cfgno_steps, jobid, todo_list)
                    todo_utils.write_todo(todo_file, todo_list)
                    todo_utils.remove_todo_lock(lock_file)
                else:
                    # Job submission failed
                    if status == 1:
                        # Fatal error
                        print("Quitting")
                        sys.exit(1)
                    else:
                        print("Will retry submitting", cfgno_steps, "later")

        sys.stdout.flush()

        subprocess.call(["sleep", str(param["nanny"]["wait"])])
        check_count -= 1

        # Reload parameters in case of hot changes
        param = utils.load_param(YAML)


############################################################
def main():
    # Set permissions
    os.system("umask 022")

    YAML = "params.yaml"

    nanny_loop(YAML)


############################################################
if __name__ == "__main__":
    main()
