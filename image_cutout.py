# Python 3.6. Written by Alex Clarke
# Adapted from https://docs.astropy.org/en/stable/nddata/utils.html
# Breakup a large fits image into smaller ones, with overlap, and save to disk.
# Sourecfinding is run on each cutout, and catalogues are sifted to remove duplicates from the overlap.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
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
    cutout_filename = input_image[:-5]+'_'+str(part)+'.fits'
    hdu.writeto(cutout_filename, overwrite=True)
    return cutout
    
    
if __name__ == '__main__':
    
    # load image to get properties
    input_image = '560mhz8hours.fits'
    f = fits.open(input_image)
    # currently hard coded to only accept square images... fix later.
    im_width = f[0].header['NAXIS1']
    data = f[0].data[0,0,:,:]
    f.close()
    # assuming input fits image is square, choose value to divide x and y axes into. total images = split_into**2.
    split_into = 4
    # get centre positions for each new fits image. assuming x=y.
    positions = np.array(range(1,(split_into*2),2))*(im_width/(split_into*2))
    # round to integer as in pixel coordinates.
    positions = positions.astype(int) # keep as original 
    positions_x = positions # make copy to append to in loop
    positions_y = positions # make copy to append to in loop
    # stack them to create list of positions of length = split_into**2.
    for i in range(split_into-1):
        positions_x = np.hstack(( positions_x, positions ))
        positions_y = np.hstack(( positions_y, np.roll(positions,i+1) ))
    # create 2D array with coordinates: [ [x1,y1], [x2,y2], [x3,y3]... ]
    position_coords_inpixels = np.array([positions_x,positions_y]).T
    # create buffer of 10% so images overlap
    size = (im_width/split_into) * 1.1
    # size array needs to be same shape as position_coords_inpixels
    size_inpixels = np.array([[size,size]]*(split_into**2)).astype(int)
    # loop over images to be cut out
    plt.figure()
    plt.imshow(data, origin='lower')
    colourlist=iter(cm.rainbow(np.linspace(0,1,split_into**2)))
    for i in range(split_into**2):
        print(' Cutting out image {0} of {1}'.format(i+1, split_into**2))
        cutout = save_cutout(input_image, tuple(position_coords_inpixels[i]), tuple(size_inpixels[i]), i)
        cutout.plot_on_original(color=next(colourlist))
    print(' Saving cutout arrangement as {0}'.format(input_image+'_cutouts.png'))
    plt.savefig(input_image+'_cutouts.png')
 
