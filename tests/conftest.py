import pytest

from pdrisk.data import load_data, split


@pytest.fixture(scope="session")
def data():
    return load_data()


@pytest.fixture(scope="session")
def splits(data):
    return split(*data)
