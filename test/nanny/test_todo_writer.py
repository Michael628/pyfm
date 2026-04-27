import multiprocessing
from unittest.mock import patch, call

import pytest

from pyfm.nanny.todo_writer import parse_cfgs, validate_steps, add_entries
from pyfm.nanny.todo import read_todo


# ---------------------------------------------------------------------------
# Helpers for concurrency test
# ---------------------------------------------------------------------------

def _writer_process(todo_file, series, cfgnos, steps, ready_event, start_event):
    """Signal ready, wait for coordinated start, then call add_entries."""
    ready_event.set()
    start_event.wait()
    add_entries(todo_file, series, cfgnos, steps)


class TestAddEntriesConcurrency:
    def test_concurrent_writers_produce_no_corruption(self, tmp_path):
        """Two concurrent add_entries calls must not corrupt or lose lines."""
        todo_file = str(tmp_path / "todo")

        ctx = multiprocessing.get_context("fork")
        ready_a = ctx.Event()
        ready_b = ctx.Event()
        start = ctx.Event()

        cfgnos_a = [str(c) for c in range(100, 110)]
        cfgnos_b = [str(c) for c in range(200, 210)]

        proc_a = ctx.Process(
            target=_writer_process,
            args=(todo_file, "a", cfgnos_a, ["smear"], ready_a, start),
        )
        proc_b = ctx.Process(
            target=_writer_process,
            args=(todo_file, "b", cfgnos_b, ["smear"], ready_b, start),
        )

        proc_a.start()
        proc_b.start()

        ready_a.wait(timeout=5)
        ready_b.wait(timeout=5)
        start.set()

        proc_a.join(timeout=10)
        proc_b.join(timeout=10)

        assert proc_a.exitcode == 0, "writer A failed"
        assert proc_b.exitcode == 0, "writer B failed"

        lines = (tmp_path / "todo").read_text().splitlines()
        assert len(lines) == 20, f"expected 20 lines, got {len(lines)}: {lines}"
        for line in lines:
            parts = line.split()
            assert len(parts) == 3, f"corrupted line: {repr(line)}"
            assert parts[1] == "smear"
            assert parts[2] == "0"


# ---------------------------------------------------------------------------
# parse_cfgs
# ---------------------------------------------------------------------------

class TestParseCfgs:
    def test_cfg_list_returns_strings(self):
        result = parse_cfgs(cfg=[200, 220], cfg_range=None)
        assert result == ["200", "220"]

    def test_cfg_range_produces_correct_list(self):
        result = parse_cfgs(cfg=None, cfg_range=[200, 400, 20])
        expected = [str(c) for c in range(200, 401, 20)]
        assert result == expected

    def test_cfg_range_inclusive_stop(self):
        result = parse_cfgs(cfg=None, cfg_range=[10, 12, 1])
        assert result == ["10", "11", "12"]

    def test_cfg_range_empty_raises(self):
        with pytest.raises(ValueError, match="empty range"):
            parse_cfgs(cfg=None, cfg_range=[400, 200, 20])

    def test_cfg_range_wrong_length_raises(self):
        with pytest.raises(ValueError, match="exactly 3 integers"):
            parse_cfgs(cfg=None, cfg_range=[200, 400])

    def test_neither_provided_raises(self):
        with pytest.raises(ValueError):
            parse_cfgs(cfg=None, cfg_range=None)


# ---------------------------------------------------------------------------
# validate_steps
# ---------------------------------------------------------------------------

class TestValidateSteps:
    def test_all_valid_passes(self):
        job_setup = {"smear": {}, "hadrons": {}}
        validate_steps(["smear", "hadrons"], job_setup)  # should not raise

    def test_invalid_step_raises(self):
        job_setup = {"smear": {}, "hadrons": {}}
        with pytest.raises(ValueError) as exc_info:
            validate_steps(["smear", "bogus"], job_setup)
        msg = str(exc_info.value)
        assert "bogus" in msg
        assert "smear" in msg
        assert "hadrons" in msg

    def test_multiple_invalid_steps_all_listed(self):
        job_setup = {"smear": {}}
        with pytest.raises(ValueError) as exc_info:
            validate_steps(["smear", "foo", "bar"], job_setup)
        msg = str(exc_info.value)
        assert "foo" in msg
        assert "bar" in msg


# ---------------------------------------------------------------------------
# add_entries
# ---------------------------------------------------------------------------

LOCK_MODULE = "pyfm.nanny.todo_writer"


class TestAddEntries:
    def test_writes_correctly_formatted_lines(self, tmp_path):
        todo_file = str(tmp_path / "todo")
        with patch(f"{LOCK_MODULE}.wait_set_todo_lock") as mock_wait, \
             patch(f"{LOCK_MODULE}.remove_todo_lock") as mock_remove:
            add_entries(todo_file, "a", ["200", "220"], ["smear", "hadrons"])

        lines = (tmp_path / "todo").read_text().splitlines()
        assert lines[0] == "a.200 smear 0 hadrons 0"
        assert lines[1] == "a.220 smear 0 hadrons 0"

    def test_round_trip_with_read_todo(self, tmp_path):
        todo_file = str(tmp_path / "todo")
        with patch(f"{LOCK_MODULE}.wait_set_todo_lock"), \
             patch(f"{LOCK_MODULE}.remove_todo_lock"):
            add_entries(todo_file, "b", ["100"], ["smear", "hadrons"])

        result = read_todo(todo_file)
        assert "b.100" in result
        assert result["b.100"] == ["b.100", "smear", "0", "hadrons", "0"]

    def test_appends_without_overwriting(self, tmp_path):
        todo_file = tmp_path / "todo"
        todo_file.write_text("a.60 smear 0\n")

        with patch(f"{LOCK_MODULE}.wait_set_todo_lock"), \
             patch(f"{LOCK_MODULE}.remove_todo_lock"):
            add_entries(str(todo_file), "a", ["200"], ["smear"])

        lines = todo_file.read_text().splitlines()
        assert len(lines) == 2
        assert lines[0] == "a.60 smear 0"
        assert lines[1] == "a.200 smear 0"

    def test_creates_file_if_absent(self, tmp_path):
        todo_file = str(tmp_path / "new_todo")
        with patch(f"{LOCK_MODULE}.wait_set_todo_lock"), \
             patch(f"{LOCK_MODULE}.remove_todo_lock"):
            add_entries(todo_file, "a", ["300"], ["smear"])

        assert (tmp_path / "new_todo").exists()

    def test_lock_acquired_and_released(self, tmp_path):
        todo_file = str(tmp_path / "todo")
        with patch(f"{LOCK_MODULE}.wait_set_todo_lock") as mock_wait, \
             patch(f"{LOCK_MODULE}.lock_file_name", return_value="todo.lock") as mock_lfn, \
             patch(f"{LOCK_MODULE}.remove_todo_lock") as mock_remove:
            add_entries(todo_file, "a", ["200"], ["smear"])

        mock_wait.assert_called_once_with("todo.lock")
        mock_remove.assert_called_once_with("todo.lock")

    def test_lock_released_even_on_error(self, tmp_path):
        todo_file = str(tmp_path / "todo")
        with patch(f"{LOCK_MODULE}.wait_set_todo_lock"), \
             patch(f"{LOCK_MODULE}.lock_file_name", return_value="todo.lock"), \
             patch(f"{LOCK_MODULE}.remove_todo_lock") as mock_remove, \
             patch("builtins.open", side_effect=OSError("disk full")):
            with pytest.raises(OSError):
                add_entries(todo_file, "a", ["200"], ["smear"])

        mock_remove.assert_called_once_with("todo.lock")
