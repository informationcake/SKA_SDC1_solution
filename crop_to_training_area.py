import argparse

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

from ska.sdc1.utils.image_utils import save_subimage

# Training area limits (RA, Dec)
TRAIN_LIM = {
    9200: {
        "ra_min": -0.04092,
        "ra_max": 0.0,
        "dec_min": -29.9400,
        "dec_max": -29.9074,
    },
    1400: {"ra_min": -0.2688, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.7265},
    560: {"ra_min": -0.6723, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.4061},
}


def crop_to_training_area(image_path, out_path, freq, pad_factor=1.0):
    """
    For a given SDC1 image, write a new FITS file containing only the training
    area.
    Training area defined by RA/Dec, which doesn't map perfectly to pixel values.

    Args:
        image_path (`str`): Path to input image
        out_path (`str`): Path to write sub-image to
        freq (`int`): [560, 1400, 9200] SDC1 image frequency (different training areas)
        pad_factor (`float`, optional): Area scaling factor to include edges
    """
    hdu = fits.open(image_path)[0]
    wcs = WCS(hdu.header)

    # Lookup training limits for given frequency
    ra_max = TRAIN_LIM[freq]["ra_max"]
    ra_min = TRAIN_LIM[freq]["ra_min"]
    dec_max = TRAIN_LIM[freq]["dec_max"]
    dec_min = TRAIN_LIM[freq]["dec_min"]

    # Centre of training area pixel coordinate:
    train_centre = SkyCoord(
        ra=(ra_max + ra_min) / 2, dec=(dec_max + dec_min) / 2, frame="fk5", unit="deg",
    )

    # Opposing corners of training area:
    train_min = SkyCoord(ra=ra_min, dec=dec_min, frame="fk5", unit="deg",)
    train_max = SkyCoord(ra=ra_max, dec=dec_max, frame="fk5", unit="deg",)

    # Training area approx width
    pixel_width = (
        abs(skycoord_to_pixel(train_max, wcs)[0] - skycoord_to_pixel(train_min, wcs)[0])
        * pad_factor
    )

    # Training area approx height
    pixel_height = (
        abs(skycoord_to_pixel(train_max, wcs)[1] - skycoord_to_pixel(train_min, wcs)[1])
        * pad_factor
    )

    save_subimage(
        image_path,
        out_path,
        skycoord_to_pixel(train_centre, wcs),
        (pixel_height, pixel_width),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="Path to input image", type=str)
    parser.add_argument("-o", help="Path to output image", type=str)
    parser.add_argument(
        "-f", help="Image frequency band (560||1400||9200, MHz)", default=1400, type=int
    )
    parser.add_argument(
        "-p", help="Padding factor to include edges", default=1.0, type=float
    )
    args = parser.parse_args()

    crop_to_training_area(args.i, args.o, args.f, args.p)
