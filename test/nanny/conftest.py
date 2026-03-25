from pathlib import Path

import pytest

NANNY_TEST_DIR = Path(__file__).parent
NANNY_DATA_DIR = NANNY_TEST_DIR / "data"


@pytest.fixture
def nanny_data_dir():
    return NANNY_DATA_DIR


@pytest.fixture
def todo_fixture_path(nanny_data_dir):
    return str(nanny_data_dir / "todo_fixture")
