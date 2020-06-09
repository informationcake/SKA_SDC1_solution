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
    cutout = Cutout2D(hdu.data, position=position, size=size, wcs=wcs, mode=partial, fill_Value=np.nan)

    # Put the cutout image in the FITS HDU
    hdu.data = cutout.data

    # Update the FITS header with the cutout WCS
    hdu.header.update(cutout.wcs.to_header())

    # Write the cutout to a new FITS file
    cutout_filename = input_image[:-4]+'_'+part+'_.fits'
    hdu.writeto(cutout_filename, overwrite=True)


if __name__ == '__main__':


    input_image = '560mhz8hours.fits'
    # assuming input fits image is square, choose value to divide x and y axes into. total images = split_into**2.
    split_into = 2
    # get centre positions for each new fits image. assuming x=y.
    positions = np.array(range(1,(split_into*2),2))*(im_width/(split_into*2))
    # round to integer as in pixel coordinates.
    positions = positions.astype(int)
    # create 2D array with coordinates: [ [x1,y1], [x2,y2], [x3,y3]... ]
    position_coords_inpixels = np.array([positions,positions]).T
    # create buffer of 10% so images overlap
    size = (im_width/split_into) * 1.1
    # size array needs to be same shape as position_coords_inpixels
    size_inpixels = np.array([[size,size]]*split_into)
    
    # loop over images to be cut out
    for i in range(split_into):
        print(' Cutting out image {0} of {0}'.format(i, split_into))
        save_cutout(input_image, position_coords_inpixels, size_inpixels, i)
    
