# Python 3.6. Written by Alex Clarke
# Adapted from https://docs.astropy.org/en/stable/nddata/utils.html
# Breakup a large fits image into smaller ones, with overlap, and save to disk.
# Sourecfinding is run on each cutout, and catalogues are sifted to remove duplicates from the overlap.

import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


def save_cutout(input_image, position, size, part):
    # Load the image and the WCS
    hdu = fits.open(input_image)[0]
    wcs = WCS(hdu.header)
    
    # Make the cutout, including the WCS. Keep only 2D, drop additional axis with .celestial. SKA image has 4D so hdu.data[0,0,:,:].
    cutout = Cutout2D(hdu.data[0,0,:,:], position=position, size=size, wcs=wcs.celestial, mode='partial', fill_value=np.nan)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file, labelled by n parts.
    cutout_filename = input_image[:-5]+'_'+str(part)+'_.fits'
    hdu.writeto(cutout_filename, overwrite=True)

    
    
if __name__ == '__main__':
    
    # load image to get properties
    input_image = '560mhz8hours.fits'
    f = fits.open(input_image)
    # currently hard coded to only accept square images... fix later.
    im_width = f[0].header['NAXIS1']
    f.close()
    # assuming input fits image is square, choose value to divide x and y axes into. total images = split_into**2.
    split_into = 4
    # get centre positions for each new fits image. assuming x=y.
    positions = np.array(range(1,(split_into*2),2))*(im_width/(split_into*2))
    # round to integer as in pixel coordinates.
    positions = positions.astype(int)
    # create 2D array with coordinates: [ [x1,y1], [x2,y2], [x3,y3]... ]
    position_coords_inpixels = np.array([positions,positions]).T
    # create buffer of 10% so images overlap
    size = (im_width/split_into) * 1.1
    # size array needs to be same shape as position_coords_inpixels
    size_inpixels = np.array([[size,size]]*split_into).astype(int)
    # loop over images to be cut out
    for i in range(split_into):
        print(' Cutting out image {0} of {1}'.format(i+1, split_into))
        save_cutout(input_image, tuple(position_coords_inpixels[i]), tuple(size_inpixels[i]), i)
