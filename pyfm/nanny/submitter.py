import sys
import os
import subprocess
import typing as t

from pyfm import utils
from pyfm.nanny.validator import check_jobs
from pyfm.nanny.inputgen import write_input_file
from pyfm.nanny.core import get_nanny_config, NannyConfig, Scheduler
from pyfm.nanny.jobconfig import JobConfig, JobBundle, build_job_configs
import pyfm.nanny.todo as todo

from functools import reduce


######################################################################
def count_jobs_in_queue(scheduler, myjob_name_pfx):
    """Count my jobs in the queue"""

    user = os.environ["USER"]

    match scheduler:
        case Scheduler.LSF:
            cmd = " ".join(
                ["bjobs -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
            )
        case Scheduler.PBS:
            cmd = " ".join(
                ["qstat -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
            )
        case Scheduler.SLURM | Scheduler.INTERACTIVE:
            cmd = " ".join(
                ["squeue -u", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
            )
        case Scheduler.COBALT:
            cmd = " ".join(
                ["qstat -fu", user, "| grep", user, "| grep ", myjob_name_pfx, "| wc -l"]
            )
        case _:
            print("Don't recognize scheduler", scheduler)
            print("Quitting")
            sys.exit(1)

    nqueued = int(subprocess.check_output(cmd, shell=True))

    return nqueued


######################################################################
def plan_submission(
    todo_list,
    job_configs: t.Dict[str, JobConfig],
    step_request: str | None = None,
) -> JobBundle | None:
    """Select the next step from todo state and bundle its cfgnos into a JobBundle.

    Walks the todo in priority order, latches the active step from the first
    unfinished task, resolves its pre-built JobConfig, and bundles contiguous
    same-step cfgnos up to ``job_config.max_cases`` (a config object, not a raw
    int). Preserves the contiguity and ``step_request`` semantics of the old
    step-selection/bundling pass it replaces. Returns None when no tasks are
    ready.
    """
    current_step = None
    job_config = None
    cfgno_steps = []

    for line in sorted(todo_list, key=todo.key_todo_entries):
        a = todo_list[line]
        if len(a) < 2:
            print("ERROR: bad todo line format")
            print(a)
            sys.exit(1)

        if n := todo.find_next_unfinished_task(a, step_request):
            index, cfgno, new_step = n
            if current_step is None:
                current_step = new_step
                if current_step not in job_configs:
                    raise ValueError(
                        f"No `job_setup` parameters provided for `{current_step}`."
                    )
                job_config = job_configs[current_step]
            elif current_step != new_step:
                # Ensure only one step type per bundled job
                break
            cfgno_steps.append([cfgno, index])

        # Stop when we have enough for a bundle
        if job_config is not None and len(cfgno_steps) >= job_config.max_cases:
            break

    ncases = len(cfgno_steps)

    if ncases > 0:
        print("Found", ncases, "cases...", cfgno_steps)
        sys.stdout.flush()

    if current_step is None or job_config is None:
        return None
    return JobBundle(job_config=job_config, cfgno_steps=cfgno_steps)


def make_inputs(param, step, cfgno_steps):
    ncases = len(cfgno_steps)
    input_files = []

    for i in range(ncases):
        cfgno_series, _ = cfgno_steps[i]
        series, cfgno = cfgno_series.split(".")

        infile = write_input_file(step, param, series, cfgno)

        input_files.append(infile)

    # Set environment variable for job scripts
    os.environ["INPUTLIST"] = " ".join(input_files)
    return input_files


def get_submit_command(
    nanny_config: NannyConfig, job_config: JobConfig, job_name: str, nodes: int, np: int
):

    job_script = job_config.run
    wall_time = job_config.wall_time

    # Check that the job script exists
    try:
        stat = os.stat(job_script)
    except OSError:
        print("Can't find the job script:", job_script)
        print("Quitting")
        sys.exit(1)

    # Job submission command depends on locale
    scheduler = nanny_config.scheduler
    match scheduler:
        case Scheduler.LSF:
            cmd = f"bsub -nnodes {str(nodes)} -J {job_name} {job_script}"
        case Scheduler.PBS:
            #cmd = f"qsub -l nodes={str(nodes)} -l walltime={wall_time} -N {job_name} {job_script}"
            cmd = f"qsub -l select={str(nodes)} -l walltime={wall_time} -N {job_name} {job_script}"
        case Scheduler.SLURM:
            cmd = f"sbatch -N {str(nodes)} -n {str(np)} -J {job_name} -t {wall_time} {job_script}"
        case Scheduler.INTERACTIVE:
            cmd = f"./{job_script}"
        case _:
            print("Don't recognize scheduler", scheduler)
            print("Quitting")
            sys.exit(1)

    return cmd


def get_jobid(scheduler: Scheduler, reply: str):
    # Get job ID
    match scheduler:
        case Scheduler.LSF:
            # a.2100 Q Job <99173> is submitted to default queue <batch>
            jobid = reply[0].split()[1].split("<")[1].split(">")[0]
            if isinstance(jobid, bytes):
                jobid = jobid.decode("ASCII")
        case Scheduler.PBS:
            # 3314170.kaon2.fnal.gov submitted
            jobid = reply[0].split(".")[0]
        case Scheduler.SLURM:
            # Submitted batch job 10059729
            jobid = reply[len(reply) - 1].split()[3]
        case Scheduler.INTERACTIVE:
            jobid = "0000"
        case Scheduler.COBALT:
            # ** Project 'semileptonic'; job rerouted to queue 'prod-short'
            # ['1607897']
            jobid = reply[-1]
    if isinstance(jobid, bytes):
        jobid = jobid.decode("ASCII")

    return jobid


######################################################################
def submit_job(nanny_config: NannyConfig, job_config: JobConfig, ncases: int):
    """Submit the job"""

    basenodes = job_config.nodes
    basetasks = reduce((lambda x, y: x * y), job_config.geom)
    ppn = job_config.ppn

    jpn = int(ppn / basetasks)
    NP = str(basetasks * ncases)
    nodes = max(basenodes, int((basetasks * ncases + ppn - 1) / ppn))
    geom = ".".join(map(str, job_config.geom))
    lattice = ".".join(map(str, job_config.lattice))

    # Append the number of cases to the step tag, as in A -> A3
    os.environ["WORKDIR"] = nanny_config.home
    os.environ["NP"] = NP
    os.environ["PPN"] = str(ppn)
    os.environ["PPJ"] = str(basetasks)
    os.environ["BASETASKS"] = str(basetasks)
    os.environ["BASENODES"] = str(basenodes)
    os.environ["LAYOUT"] = geom
    os.environ["LATTICE"] = lattice

    job_name = nanny_config.job_name_pfx + "-" + job_config.step + str(ncases)

    job_nodes = (
        nodes
        if job_config.node_minimum is None
        else max(nodes, job_config.node_minimum)
    )

    cmd = get_submit_command(nanny_config, job_config, job_name, job_nodes, int(NP))

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

    jobid = get_jobid(nanny_config.scheduler, reply)
    return 0, jobid


######################################################################
def mark_queued_todo_entries(step, cfgno_steps, jobid, todo_list, barrier: bool = True):
    """Update the todo_file, change status to "Q" and mark the job number"""

    barrier_mark = "Q" if barrier else "Qcont"
    for k in range(len(cfgno_steps)):
        c, i = cfgno_steps[k]

        todo_list[c][i] = f"{step}_{barrier_mark}"
        todo_list[c][i + 1] = jobid


######################################################################
def nanny_loop(YAML, require_step: str | None = None):
    """Check job periodically and submit to the queue"""

    date = subprocess.check_output("date", shell=True).rstrip().decode()
    try:
        hostname = subprocess.check_output("hostname", shell=True).rstrip().decode()
        print(date, "Spawn job process", os.getpid(), "started on", hostname)
    except subprocess.CalledProcessError:
        print(date, "Spawn job process", os.getpid(), "started on", "localhost")

    sys.stdout.flush()

    yaml_params = utils.io.load_param(YAML)

    # Keep going until
    #   we see a file called "STOP" OR
    #   we have exhausted the list OR
    #   there are job submission or queue checking errors

    check_count = int(yaml_params["nanny"]["check_interval"])
    while True:
        if os.access("STOP", os.R_OK):
            print("Spawn job process stopped because STOP file is present")
            break

        nanny_config = get_nanny_config(yaml_params)
        todo_file = os.path.join(nanny_config.home, nanny_config.todo_file)
        job_name_pfx = nanny_config.job_name_pfx
        scheduler = nanny_config.scheduler

        lock_file = todo.lock_file_name(todo_file)

        # Count queued jobs with our job name
        nqueued = count_jobs_in_queue(scheduler, job_name_pfx)

        # Submit until we have the desired number of jobs in the queue
        if nqueued < nanny_config.max_queue:
            job_configs = build_job_configs(yaml_params)
            todo.wait_set_todo_lock(lock_file)
            todo_list = todo.read_todo(todo_file)
            todo.remove_todo_lock(lock_file)

            # Plan the next submission from pre-built job configs (config-first)
            bundle = plan_submission(todo_list, job_configs, require_step)

            # Check completion and purge scratch files for complete jobs
            if check_count == 0:
                check_jobs(
                    yaml_params,
                    nanny_config=nanny_config,
                    job_configs=job_configs,
                )
                check_count = nanny_config.check_interval

            if bundle and bundle.ncases > 0:
                # Make input
                make_inputs(yaml_params, bundle.job_config.step, bundle.cfgno_steps)

                # Submit the job
                status, jobid = submit_job(
                    nanny_config, bundle.job_config, bundle.ncases
                )

                cfgnos = ", ".join(c[0] for c in bundle.cfgno_steps)
                date = subprocess.check_output("date", shell=True).rstrip().decode()
                print(
                    date,
                    "Submitted job",
                    jobid,
                    "for",
                    cfgnos,
                    "step",
                    bundle.job_config.step,
                )

                # Job submissions succeeded
                # Edit the todo_file, marking the lattice queued and
                # indicating the jobid
                if status == 0:
                    todo.wait_set_todo_lock(lock_file)
                    todo_list = todo.read_todo(todo_file)
                    mark_queued_todo_entries(
                        bundle.job_config.step,
                        bundle.cfgno_steps,
                        jobid,
                        todo_list,
                        bundle.job_config.barrier,
                    )
                    todo.write_todo(todo_file, todo_list)
                    todo.remove_todo_lock(lock_file)
                else:
                    # Job submission failed
                    if status == 1:
                        # Fatal error
                        print("Quitting")
                        sys.exit(1)
                    else:
                        print("Will retry submitting", bundle.cfgno_steps, "later")

        sys.stdout.flush()

        subprocess.call(["sleep", str(nanny_config.wait)])
        check_count -= 1

        # Reload parameters in case of hot changes
        yaml_params = utils.io.load_param(YAML)
