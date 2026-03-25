import pytest

from pyfm.nanny.core import NannyConfig, JobConfig
from pyfm.nanny.submitter import (
    get_submit_command,
    get_jobid,
    next_cfgno_steps,
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
            max_cases=4,
            max_queue=2,
            wait=60,
            check_interval=5,
            job_name_pfx="test",
            scheduler="SLURM",
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
            formatting={},
            logging_level="INFO",
            runid="test-run",
            params={},
        )

    def test_slurm(self, nanny_config, job_config):
        cmd = get_submit_command(nanny_config, job_config, "test-h1", 2, 8)
        assert cmd == f"sbatch -N 2 -n 8 -J test-h1 -t 01:00:00 {job_config.run}"

    def test_pbs(self, nanny_config, job_config, tmp_path):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler="PBS")
        cmd = get_submit_command(nc, job_config, "test-h1", 2, 8)
        assert (
            cmd == f"qsub -l nodes=2 -l walltime=01:00:00 -N test-h1 {job_config.run}"
        )

    def test_lsf(self, nanny_config, job_config):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler="LSF")
        cmd = get_submit_command(nc, job_config, "test-h1", 2, 8)
        assert cmd == f"bsub -nnodes 2 -J test-h1 {job_config.run}"

    def test_interactive(self, nanny_config, job_config):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler="INTERACTIVE")
        cmd = get_submit_command(nc, job_config, "test-h1", 2, 8)
        assert cmd == f"./{job_config.run}"

    def test_unknown_scheduler(self, nanny_config, job_config):
        from dataclasses import replace

        nc = replace(nanny_config, scheduler="UNKNOWN")
        with pytest.raises(SystemExit):
            get_submit_command(nc, job_config, "test-h1", 2, 8)


class TestGetJobid:
    def test_slurm(self):
        reply = ["Submitted batch job 10059729"]
        assert get_jobid("SLURM", reply) == "10059729"

    def test_pbs(self):
        reply = ["3314170.kaon2.fnal.gov submitted"]
        assert get_jobid("PBS", reply) == "3314170"

    def test_lsf(self):
        reply = ["Job <99173> is submitted to default queue <batch>"]
        assert get_jobid("LSF", reply) == "99173"

    def test_interactive(self):
        reply = ["done"]
        assert get_jobid("INTERACTIVE", reply) == "0000"

    def test_cobalt(self):
        reply = [
            "** Project 'semileptonic'; job rerouted to queue 'prod-short'",
            "1607897",
        ]
        assert get_jobid("Cobalt", reply) == "1607897"


class TestNextCfgnoSteps:
    def test_finds_ready_tasks(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "smear", "0"],
        }
        step, cfgno_steps = next_cfgno_steps(10, todo_list)
        assert step == "smear"
        assert len(cfgno_steps) == 2

    def test_respects_max_cases(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "smear", "0"],
            "a.140": ["a.140", "smear", "0"],
        }
        step, cfgno_steps = next_cfgno_steps(2, todo_list)
        assert len(cfgno_steps) == 2

    def test_groups_by_step(self):
        # Only bundles configs with the same step
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "hadrons", "0"],
        }
        step, cfgno_steps = next_cfgno_steps(10, todo_list)
        assert step == "smear"
        assert len(cfgno_steps) == 1
        assert cfgno_steps[0][0] == "a.60"

    def test_skips_blocked_entries(self):
        todo_list = {
            "a.60": ["a.60", "smear_Q", "1000", "hadrons", "0"],
            "a.100": ["a.100", "smear_X", "1000", "hadrons", "0"],
        }
        step, cfgno_steps = next_cfgno_steps(10, todo_list)
        assert step == "hadrons"
        assert len(cfgno_steps) == 1
        assert cfgno_steps[0][0] == "a.100"

    def test_empty_when_all_done(self):
        todo_list = {
            "a.60": ["a.60", "smear_X", "1000"],
        }
        step, cfgno_steps = next_cfgno_steps(10, todo_list)
        assert step is None
        assert len(cfgno_steps) == 0

    def test_step_request_filter(self):
        todo_list = {
            "a.60": ["a.60", "smear", "0"],
            "a.100": ["a.100", "hadrons", "0"],
        }
        step, cfgno_steps = next_cfgno_steps(10, todo_list, step_request="hadrons")
        assert step == "hadrons"
        assert len(cfgno_steps) == 1


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
        mark_queued_todo_entries(
            "smear", cfgno_steps, "99999", todo_list, barrier=False
        )
        assert todo_list["a.60"][1] == "smear_Qcont"
        assert todo_list["a.60"][2] == "99999"
