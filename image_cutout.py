# Python 3.6. Written by Alex Clarke
# Adapted from https://docs.astropy.org/en/stable/nddata/utils.html
# Breakup a large fits image into smaller ones, with overlap, and save to disk.
# Sourecfinding is run on each cutout, and catalogues are sifted to remove duplicates from the overlap.

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def save_cutout(input_image, position, size, part):

    # Load the image and the WCS
    hdu = fits.open(input_image)[0]
    wcs = WCS(hdu.header)

    # Make the cutout, including the WCS
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file
    cutout_filename = input_image[:-4]+'_'+part+'_.fits'
    hdu.writeto(cutout_filename, overwrite=True)


if __name__ == '__main__':


    input_image = '560mhz8hours.fits'
    position = (15000, 15000)
    size = (2000, 2000)
    save_cutout(input_image, position, size, 1)
    
