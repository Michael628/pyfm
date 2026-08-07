import pytest
from click.testing import CliRunner


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def fake_yaml_params():
    return {"nanny": {"queue": "default"}, "tasks": []}
