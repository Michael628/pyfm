import sys
import os
import subprocess
from pyfm import utils

from pyfm.nanny.validator import check_jobs
from pyfm.nanny.inputgen import write_input_file

from functools import reduce


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
    for line in sorted(todo_list, key=utils.todo.key_todo_entries):
        a = todo_list[line]
        if len(a) < 2:
            print("ERROR: bad todo line format")
            print(a)
            sys.exit(1)

        index, cfgno, new_step = utils.todo.find_next_unfinished_task(a)
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


def make_inputs(param, step, cfgno_steps):
    ncases = len(cfgno_steps)
    input_files = []

    for i in range(ncases):
        (cfgno_series, _) = cfgno_steps[i]
        (series, cfgno) = cfgno_series.split(".")

        infile = write_input_file(step, param, series, cfgno)

        input_files.append(infile)

    # Set environment variable for job scripts
    os.environ["INPUTLIST"] = " ".join(input_files)
    return input_files


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
        jobid = "0000"
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
    try:
        hostname = subprocess.check_output("hostname", shell=True).rstrip().decode()
        print(date, "Spawn job process", os.getpid(), "started on", hostname)
    except subprocess.CalledProcessError:
        print(date, "Spawn job process", os.getpid(), "started on", "localhost")

    sys.stdout.flush()

    param = utils.io.load_param(YAML)

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

        lock_file = utils.todo.lock_file_name(todo_file)

        # Count queued jobs with our job name
        nqueued = count_queue(scheduler, job_name_pfx)

        # Submit until we have the desired number of jobs in the queue
        if nqueued < param["nanny"]["max_queue"]:
            utils.todo.wait_set_todo_lock(lock_file)
            todo_list = utils.todo.read_todo(todo_file)
            utils.todo.remove_todo_lock(lock_file)

            # List a set of cfgnos
            step, cfgno_steps = next_cfgno_steps(max_cases, todo_list)
            ncases = len(cfgno_steps)

            # Check completion and purge scratch files for complete jobs
            if check_count == 0:
                check_jobs(YAML)
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
                    utils.todo.wait_set_todo_lock(lock_file)
                    todo_list = utils.todo.read_todo(todo_file)
                    mark_queued_todo_entries(step, cfgno_steps, jobid, todo_list)
                    utils.todo.write_todo(todo_file, todo_list)
                    utils.todo.remove_todo_lock(lock_file)
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
        param = utils.io.load_param(YAML)


############################################################
def main():
    # Set permissions
    os.system("umask 022")

    YAML = "params.yaml"

    nanny_loop(YAML)


############################################################
if __name__ == "__main__":
    main()
