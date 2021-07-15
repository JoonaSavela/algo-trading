import os
from algtra import data


def test_get_data_dir_exists():
    assert os.path.exists(data.get_data_dir())


def test_get_data_dir_ending():
    required_ending = "Seagate Expansion Drive/data"

    assert data.get_data_dir()[-len(required_ending) :] == required_ending
