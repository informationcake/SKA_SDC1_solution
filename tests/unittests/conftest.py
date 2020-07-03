import pytest


@pytest.fixture
def image_path():
    return "tests/testdata/images"


@pytest.fixture
def image_name():
    return "test_B1_1000h.fits"
