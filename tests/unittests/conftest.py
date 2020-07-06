import pytest


@pytest.fixture
def images_dir():
    return "tests/testdata/images"


@pytest.fixture
def test_image_name():
    return "test_B1_1000h.fits"


@pytest.fixture
def pb_image_name():
    return "PrimaryBeam_B1.fits"
