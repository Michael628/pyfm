import pytest
from unittest.mock import MagicMock, patch

import pyfm.nanny.todo as todo


class TestReadTodo:
    def test_reads_all_entries(self, todo_fixture_path):
        result = todo.read_todo(todo_fixture_path)
        assert len(result) == 7

    def test_keys_are_cfgno(self, todo_fixture_path):
        result = todo.read_todo(todo_fixture_path)
        assert "a.60" in result
        assert "b.60" in result
        assert "c.100" in result

    def test_values_are_split_tokens(self, todo_fixture_path):
        result = todo.read_todo(todo_fixture_path)
        assert result["a.60"] == ["a.60", "smear_X", "1000001", "hadrons", "0"]

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("a.60 smear 0\n\nb.60 smear 0\n")
        result = todo.read_todo(str(f))
        assert len(result) == 2

    def test_skips_whitespace_only_lines(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("a.60 smear 0\n   \n\tb.60 smear 0\n")
        result = todo.read_todo(str(f))
        assert len(result) == 2

    def test_skips_comment_lines(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("# comment\na.60 smear 0\n   # another comment\nb.60 smear 0\n")
        result = todo.read_todo(str(f))
        assert set(result.keys()) == {"a.60", "b.60"}

    def test_strips_inline_comments(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("a.60 smear 0 # submit smear first\nb.60 hadrons 0 # submit hadrons\n")
        result = todo.read_todo(str(f))
        assert result["a.60"] == ["a.60", "smear", "0"]
        assert result["b.60"] == ["b.60", "hadrons", "0"]


class TestWriteTodo:
    def test_round_trip(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("b.100 smear 0\na.60 hadrons 0\n")
        todo_list = todo.read_todo(str(f))
        todo.write_todo(str(f), todo_list)
        result = todo.read_todo(str(f))
        assert set(result.keys()) == {"a.60", "b.100"}

    def test_writes_sorted(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("b.100 smear 0\na.60 hadrons 0\na.100 hadrons 0\n")
        todo_list = todo.read_todo(str(f))
        todo.write_todo(str(f), todo_list)
        lines = f.read_text().strip().splitlines()
        keys = [line.split()[0] for line in lines]
        assert keys == ["a.60", "a.100", "b.100"]


class TestKeyTodoEntries:
    def test_same_stream_numeric_order(self):
        entries = ["a.100", "a.60", "a.1000"]
        result = sorted(entries, key=todo.key_todo_entries)
        assert result == ["a.60", "a.100", "a.1000"]

    def test_different_streams(self):
        entries = ["b.60", "a.100"]
        result = sorted(entries, key=todo.key_todo_entries)
        assert result == ["a.100", "b.60"]


class TestReadTodoDuplicates:
    def test_warns_on_duplicate(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("a.60 smear 0\na.60 hadrons 0\n")
        mock_logger = MagicMock()
        with patch("pyfm.utils.get_logger", return_value=mock_logger):
            todo.read_todo(str(f))
        mock_logger.warn.assert_called_once()
        call_args = mock_logger.warn.call_args[0][0]
        assert "a.60" in call_args

    def test_last_write_wins(self, tmp_path):
        f = tmp_path / "todo"
        f.write_text("a.60 smear 0\na.60 hadrons 0\n")
        result = todo.read_todo(str(f))
        assert len(result) == 1
        assert result["a.60"] == ["a.60", "hadrons", "0"]


class TestFindNextUnfinishedTask:
    def test_finds_ready_task(self):
        line = ["a.60", "smear", "0"]
        result = todo.find_next_unfinished_task(line)
        assert result == (1, "a.60", "smear")

    def test_skips_completed(self):
        line = ["a.60", "smear_X", "1000", "hadrons", "0"]
        result = todo.find_next_unfinished_task(line)
        assert result == (3, "a.60", "hadrons")

    def test_blocked_by_queued(self):
        # _Q is a barrier — should not find contract
        line = ["b.60", "smear_X", "1000", "hadrons_Q", "2000", "contract", "0"]
        result = todo.find_next_unfinished_task(line)
        assert result is None

    def test_blocked_by_xxfix(self):
        # _XXfix is a barrier
        line = ["a.100", "smear_XXfix", "1000", "hadrons", "0"]
        result = todo.find_next_unfinished_task(line)
        assert result is None

    def test_not_blocked_by_qcont(self):
        # _Qcont is NOT a barrier — should find contract
        line = ["b.100", "smear_X", "1000", "hadrons_Qcont", "2000", "contract", "0"]
        result = todo.find_next_unfinished_task(line)
        assert result == (5, "b.100", "contract")

    def test_not_blocked_by_completed(self):
        line = ["b.140", "smear_X", "1000", "hadrons_X", "2000", "contract", "0"]
        result = todo.find_next_unfinished_task(line)
        assert result == (5, "b.140", "contract")

    def test_all_done(self):
        line = ["c.100", "smear_X", "1000", "hadrons_X", "2000", "contract_X", "3000"]
        result = todo.find_next_unfinished_task(line)
        assert result is None

    def test_step_request_filter(self):
        line = ["a.60", "smear_X", "1000", "hadrons", "0"]
        # Request only "hadrons" step
        result = todo.find_next_unfinished_task(line, step_request="hadrons")
        assert result == (3, "a.60", "hadrons")
        # Request non-existent step
        result = todo.find_next_unfinished_task(line, step_request="contract")
        assert result is None

    def test_step_request_blocked_by_unsubmitted_predecessor(self):
        # smear is unsubmitted (not a Q/XXfix barrier). Requesting "hadrons"
        # must NOT skip past the unsubmitted predecessor: hadrons is not yet
        # ready to run, so it must not be bundled.
        line = ["a.60", "smear", "0", "hadrons", "0"]
        result = todo.find_next_unfinished_task(line, step_request="hadrons")
        assert result is None
        # Without the request, the next ready task is smear itself.
        result = todo.find_next_unfinished_task(line)
        assert result == (1, "a.60", "smear")

    def test_step_request_allowed_past_noncont_predecessor(self):
        # A non-blocking predecessor (Qcont) does not gate the requested step.
        line = ["b.100", "smear_Qcont", "1000", "hadrons", "0"]
        result = todo.find_next_unfinished_task(line, step_request="hadrons")
        assert result == (3, "b.100", "hadrons")


class TestFindNextQueuedTask:
    def test_finds_queued(self):
        line = ["b.60", "smear_X", "1000", "hadrons_Q", "2000", "contract", "0"]
        result = todo.find_next_queued_task(line)
        assert result == (3, "b.60", "hadrons_Q")

    def test_finds_qcont(self):
        line = ["b.100", "smear_X", "1000", "hadrons_Qcont", "2000"]
        result = todo.find_next_queued_task(line)
        assert result == (3, "b.100", "hadrons_Qcont")

    def test_no_queued(self):
        line = ["a.60", "smear_X", "1000", "hadrons", "0"]
        result = todo.find_next_queued_task(line)
        assert result is None
