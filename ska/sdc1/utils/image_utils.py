import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def save_subimage(image_path, out_path, position, size, overwrite=True):
    """
    Write a sub-section of an image to a new FITS file.

    Adapted from https://docs.astropy.org/en/stable/nddata/utils.html

    Args:
        image_path (`str`): Path to input image
        out_path (`str`): Path to write sub-image to
        position (`tuple`): Pixel position of sub-image centre (x, y)
        size (`tuple`): Size in pixels of sub-image (ny, nx)
    """

    # Load the image and the WCS
    hdu = fits.open(image_path)[0]
    wcs = WCS(hdu.header)

    # Make the cutout, including the WCS.
    # Keep only 2D, drop additional axis with celestial.
    # SKA image has 4D so hdu.data[0,0,:,:].
    if len(hdu.data.shape) == 4:
        data_to_write = hdu.data[0, 0, :, :]
    else:
        data_to_write = hdu.data
    cutout = Cutout2D(
        data_to_write,
        position=position,
        size=size,
        wcs=wcs.celestial,
        mode="partial",
        fill_value=np.nan,
    )

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file.
    hdu.writeto(out_path, overwrite=overwrite)
    return cutout
