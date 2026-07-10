import pytest

from pyfm.nanny.core import NannyConfig, Scheduler
from pyfm.nanny.jobconfig import JobConfig
from pyfm.nanny.submitter import (
    get_submit_command,
    get_jobid,
    plan_submission,
    mark_queued_todo_entries,
    make_inputs,
)
import pyfm.nanny.todo as todo


class TestGetSubmitCommand:
    @pytest.fixture
    def nanny_config(self, tmp_path):
        return NannyConfig(
            home=str(tmp_path),
            todo_file="todo",
            max_queue=2,
            wait=60,
            check_interval=5,
            job_name_pfx="test",
            scheduler=Scheduler.SLURM,
            formatting={},
            logging_level="INFO",
            runid="test-run",
        )

    @pytest.fixture
    def job_config(self, tmp_path):
        # Create a dummy job script so os.stat doesn't fail
        script = tmp_path / "run.sh"
        script.write_text("#!/bin/bash\necho hello")
        return JobConfig(
            run=str(script),
            job_type="hadrons",
            tasks={},
            step="hadrons",
            io="input",
            wall_time="01:00:00",
            ppn=4,
            nodes=2,
            lattice=[16, 16, 16, 32],
            geom=[1, 1, 1, 4],
            params={},
            formatting={},
            logging_level="INFO",
            runid="test-run",
        )

    def test_slurm(self, nanny_config, job_config):
        cmd = get_submit_command(nanny_config, job_config, "test-h1", 2, 8)
        assert cmd == f"sbatch -N 2 -n 8 -J test-h1 -t 01:00:00 {job_config.run}"

    def test_pbs(self, nanny_config, job_config, tmp_path):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler=Scheduler.PBS)
        cmd = get_submit_command(nc, job_config, "test-h1", 2, 8)
        assert cmd == f"qsub -l select=2 -l walltime=01:00:00 -N test-h1 {job_config.run}"

    def test_lsf(self, nanny_config, job_config):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler=Scheduler.LSF)
        cmd = get_submit_command(nc, job_config, "test-h1", 2, 8)
        assert cmd == f"bsub -nnodes 2 -J test-h1 {job_config.run}"

    def test_interactive(self, nanny_config, job_config):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler=Scheduler.INTERACTIVE)
        cmd = get_submit_command(nc, job_config, "test-h1", 2, 8)
        assert cmd == f"./{job_config.run}"

    def test_unhandled_scheduler(self, nanny_config, job_config):
        from dataclasses import replace

        # COBALT is a valid scheduler but not handled by get_submit_command
        nc = replace(nanny_config, scheduler=Scheduler.COBALT)
        with pytest.raises(SystemExit):
            get_submit_command(nc, job_config, "test-h1", 2, 8)


class TestGetJobid:
    def test_slurm(self):
        reply = ["Submitted batch job 10059729"]
        assert get_jobid(Scheduler.SLURM, reply) == "10059729"

    def test_pbs(self):
        reply = ["3314170.kaon2.fnal.gov submitted"]
        assert get_jobid(Scheduler.PBS, reply) == "3314170"

    def test_lsf(self):
        reply = ["Job <99173> is submitted to default queue <batch>"]
        assert get_jobid(Scheduler.LSF, reply) == "99173"

    def test_interactive(self):
        reply = ["done"]
        assert get_jobid(Scheduler.INTERACTIVE, reply) == "0000"

    def test_cobalt(self):
        reply = ["** Project 'semileptonic'; job rerouted to queue 'prod-short'", "1607897"]
        assert get_jobid(Scheduler.COBALT, reply) == "1607897"


def _job_config(step, max_cases):
    return JobConfig(
        run="x",
        job_type="stub",
        step=step,
        io="i",
        wall_time="0:1",
        ppn=1,
        nodes=1,
        lattice=[1, 1, 1, 1],
        geom=[1, 1, 1, 1],
        params={},
        max_cases=max_cases,
        formatting={},
        logging_level="INFO",
        runid="r",
    )


class TestPlanSubmission:
    def test_finds_ready_tasks(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "smear", "0"],
        }
        bundle = plan_submission(todo_list, {"smear": _job_config("smear", 10)})
        assert bundle.job_config.step == "smear"
        assert len(bundle.cfgno_steps) == 2

    def test_respects_max_cases(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "smear", "0"],
            "a.140": ["a.140", "smear", "0"],
        }
        bundle = plan_submission(todo_list, {"smear": _job_config("smear", 2)})
        assert len(bundle.cfgno_steps) == 2

    def test_groups_by_step(self):
        # Only bundles configs with the same step
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "hadrons", "0"],
        }
        bundle = plan_submission(
            todo_list,
            {"smear": _job_config("smear", 10), "hadrons": _job_config("hadrons", 10)},
        )
        assert bundle.job_config.step == "smear"
        assert len(bundle.cfgno_steps) == 1
        assert bundle.cfgno_steps[0][0] == "a.60"

    def test_skips_blocked_entries(self):
        todo_list = {
            "a.60": ["a.60", "smear_Q", "1000", "hadrons", "0"],
            "a.100": ["a.100", "smear_X", "1000", "hadrons", "0"],
        }
        bundle = plan_submission(todo_list, {"hadrons": _job_config("hadrons", 10)})
        assert bundle.job_config.step == "hadrons"
        assert len(bundle.cfgno_steps) == 1
        assert bundle.cfgno_steps[0][0] == "a.100"

    def test_returns_none_when_all_done(self):
        todo_list = {
            "a.60": ["a.60", "smear_X", "1000"],
        }
        bundle = plan_submission(todo_list, {"smear": _job_config("smear", 10)})
        assert bundle is None

    def test_step_request_filter(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "hadrons", "0"],
        }
        bundle = plan_submission(
            todo_list,
            {"smear": _job_config("smear", 10), "hadrons": _job_config("hadrons", 10)},
            step_request="hadrons",
        )
        assert bundle.job_config.step == "hadrons"
        assert len(bundle.cfgno_steps) == 1

    def test_per_job_max_cases_drives_bundling(self):
        # max_cases read from the job config, not a passed int
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "smear", "0"],
            "a.140": ["a.140", "smear", "0"],
        }
        bundle = plan_submission(todo_list, {"smear": _job_config("smear", 3)})
        assert len(bundle.cfgno_steps) == 3


class TestMarkQueuedTodoEntries:
    def test_marks_with_barrier(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "smear", "0"],
        }
        cfgno_steps = [["a.60", 1], ["a.100", 1]]
        mark_queued_todo_entries("smear", cfgno_steps, "99999", todo_list, barrier=True)
        assert todo_list["a.60"][1] == "smear_Q"
        assert todo_list["a.60"][2] == "99999"
        assert todo_list["a.100"][1] == "smear_Q"

    def test_marks_without_barrier(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
        }
        cfgno_steps = [["a.60", 1]]
        mark_queued_todo_entries("smear", cfgno_steps, "99999", todo_list, barrier=False)
        assert todo_list["a.60"][1] == "smear_Qcont"
        assert todo_list["a.60"][2] == "99999"
