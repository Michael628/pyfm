import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import xml_diff  # noqa: E402

from pyfm import utils

TASKS_TEST_DIR = Path(__file__).parent


@pytest.fixture
def tasks_data_dir():
    return TASKS_TEST_DIR


@pytest.fixture
def params(tasks_data_dir):
    return utils.io.load_param(str(tasks_data_dir / "params.yaml"))


@pytest.fixture
def contract_params(tasks_data_dir):
    return utils.io.load_param(str(tasks_data_dir / "params_contract.yaml"))


@pytest.fixture
def assert_xml_equal():
    def _compare(actual_path, expected_path):
        actual = ET.parse(actual_path).getroot()
        expected = ET.parse(expected_path).getroot()
        xml_diff.normalize_element(actual)
        xml_diff.normalize_element(expected)
        equal, diffs = xml_diff.elements_equal(actual, expected)
        assert equal, "XML mismatch:\n" + "\n".join(diffs)

    return _compare
