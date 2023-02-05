import pytest
from pfam.utils import load_json


@pytest.fixture(scope="module")
def test_data_path():
    return "./tests/data"


@pytest.fixture(scope="module")
def word2id():
    return load_json("./tests/data/experiments/BOUXZ/word2id.json")


@pytest.fixture(scope="module")
def fam2label():
    return load_json("./tests/data/experiments/BOUXZ/fam2label.json")


@pytest.fixture(scope="module")
def label2fam():
    return load_json("./tests/data/experiments/BOUXZ/label2fam.json")


@pytest.fixture(scope="module")
def params():
    return load_json("./tests/data/experiments/BOUXZ/params.json")
