# Python 3.6. Written by Alex Clarke
# Breakup a large fits image into smaller ones, with overlap, and save to disk.
# Sourecfinding is run on each cutout, and catalogues are sifted to remove duplicates from the overlap.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import bdsf
import glob
import pickle

from matplotlib.pyplot import cm
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from memory_profiler import profile

# list of functions
# load/save pickle objects
# save_cutout
# do_image_chopping
# make_image_cubes
# do_sourcefinding



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
	
	

#Loading/saving python data objects
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def update_header_from_cutout2D(hdu, cutout):
    # update data
    newdata = np.zeros((1,1,cutout.data.shape[0], cutout.data.shape[1]), dtype=np.float32)
    newdata[0,0,:,:] = cutout.data
    hdu.data = newdata
    # update header cards returned from cutout2D wcs:
    hdu.header.set('CRVAL1', cutout.wcs.wcs.crval[0])
    hdu.header.set('CRVAL2', cutout.wcs.wcs.crval[1])
    hdu.header.set('CRPIX1', cutout.wcs.wcs.crpix[0])
    hdu.header.set('CRPIX2', cutout.wcs.wcs.crpix[1])
    hdu.header.set('CDELT1', cutout.wcs.wcs.cdelt[0])
    hdu.header.set('CDELT2', cutout.wcs.wcs.cdelt[1])
    hdu.header.set('NAXIS1', cutout.wcs.pixel_shape[0])
    hdu.header.set('NAXIS2', cutout.wcs.pixel_shape[1])
    return hdu



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def do_image_chopping(input_image, split_into):
    hdu = fits.open(input_image)[0]
    wcs = WCS(hdu.header)
    # currently hard coded to only accept square images
    im_width = hdu.header['NAXIS1'] # get image width
    print(' Input fits image dimensions: {0}'.format(im_width))
    print(' Cutting into {0} images of dimensions {1}'.format(split_into**2, im_width/split_into))
    
    # get centre positions for each new fits image. assuming x=y. divide image width by split_into*2
    positions = np.array(range(1,(split_into*2),2))*(im_width/(split_into*2))
    # round to integer as in pixel coordinates. this approximation shouldn't matter since we include a buffer later
    positions = positions.astype(int) # keep as original
    positions_x = positions # make copy to append to in loop
    positions_y = positions # make copy to append to in loop
    
    # Make a 2D array of all centre positions. length = split_into**2.
    for i in range(split_into-1):
        # stack x coords repeating split_into times.
        positions_x = np.hstack(( positions_x, positions ))  # e.g. [ x1, x2, x3, x4,  x1, x2, x3, x4,  repeat split_into times] 
        # stack y coords, but np.roll shifts array indices by 1 to get different combinations
        positions_y = np.hstack(( positions_y, np.roll(positions,i+1) )) # e.g. [ (y1, y2, y3, y4), (y2, y3, y4, y1), (y3, y4, y1, y2), ... ]
    
    # create 2D array with coordinates: [ [x1,y1], [x2,y2], [x3,y3]... ]
    position_coords_inpixels = np.array([positions_x,positions_y]).T
    # create buffer of 5% so images overlap. This can be small... only needs to account for image edge cutting through 
    size = (im_width/split_into) * 1.05 # e.g. 4000 pixel image becomes 4200. sifting to remove duplicates later
    # size array needs to be same shape as position_coords_inpixels
    size_inpixels = np.array([[size,size]]*(split_into**2)).astype(int)
    
    # loop over images to be cut out
    plt.figure() # plot original image and overlay cutout boundaries at the end.
    data[data<1e-7]=1e-7 # min pixel brightness to display
    data[data>1e-5]=1e-5 # max pixel brightness to display
    plt.imshow(hdu.data[0,0,:,:], origin='lower')
    colourlist=iter(cm.rainbow(np.linspace(0,1,split_into**2))) # each cutout a different colour
    for i in range(split_into**2):
        print(' Cutting out image {0} of {1}'.format(i+1, split_into**2))
	cutout = Cutout2D(hdu.data[0,0,:,:], position=tuple(position_coords_inpixels[i], size=tuple(size_inpixels[i]), mode='trim', wcs=wcs.celestial, copy=True)
	cutout.plot_on_original(color=next(colourlist))
   	# Update the FITS header with the cutout WCS by hand using my own function
    	hdu = update_header_from_cutout2D(hdu, cutout)
    	hdu.writeto(input_image[:-5]+'_'+str(i)+'_cutout.fits', overwrite=True) # Write the cutout to a new FITS file
    print(' Saving cutout arrangement as {0}'.format(input_image+'_cutouts.png'))
    plt.savefig(input_image+'_cutout_annotation.png')
    


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------


    
# make image cube for pybdsf spectral index mode, looping over all cutouts
def make_image_cubes_for_cutouts():
    # get cutout file names, must be in same order so they are matched correctly
    images_560 = sorted(glob.glob('560*_cutout.fits'))
    images_1400 = sorted(glob.glob('1400*_cutout.fits'))
    # loop over image cutouts to make cube for each of them
    for file560, file1400, i in zip(images_560, images_1400, range(len(images_560))):
        print(' Making cube {0} of {1}'.format(i, len(images_560)-1))
        hdu560 = fits.open(file560)[0]
        hdu1400 = fits.open(file1400)[0]
        # make cube from the input files along freq axis
        cube = np.zeros((2,hdu560.data.shape[0],hdu560.data.shape[1]))
        cube[0,:,:] = hdu560.data[0,0,:,:] # add 560 Mhz data
        cube[1,:,:] = hdu1400.data[0,0,:,:] # add 1400 Mhz data
        hdu_new = fits.PrimaryHDU(data=cube, header=hdu560.header)
        # update frequency info in the header. It puts 560MHz as ch0, but incorrectly assigns the interval to the next freq channel
        hdu_new.header.set('CDELT3', 840000000) # 1400 MHz - 560 MHz = 840 MHz.
        hdu_new.writeto('cube_cutout_'+str(i)+'.fits')
    
        
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
                                  
                                    
         
def do_sourcefinding(imagename, si=True):
    # get beam info manually. SKA image seems to cause PyBDSF issues finding this info.
    f = fits.open(imagename)
    beam_maj = f[0].header['BMAJ']
    beam_min = f[0].header['BMIN']
    #beam_pa = f[0].header['BPA'] # not in SKA fits header, but we know it's circular
    beam_pa = 0
    f.close()
    # using some sensible and thorough hyper-parameters. PSF_vary and adaptive_rms_box is more computationally intensive, but needed.
    if si==True:
        img = bdsf.process_image(imagename, adaptive_rms_box=False, spectralindex_do=True, advanced_opts=True,\
            atrous_do=False, output_opts=True, output_all=True, opdir_overwrite='append', beam=(beam_maj, beam_min, beam_pa),\
            blank_limit=None, thresh='hard', thresh_isl=4.0, thresh_pix=5.0, \
            collapse_mode='average', collapse_wt='unity', frequency_sp=[560e6, 1400e6])
									
    if si==False:                                
        img = bdsf.process_image(imagename, adaptive_rms_box=True, advanced_opts=True,\
            atrous_do=False, output_opts=True, output_all=True, opdir_overwrite='append', beam=(beam_maj, beam_min, beam_pa),\
            blank_limit=None, thresh='hard', thresh_isl=4.0, thresh_pix=5.0, psf_snrtop=0.30)



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    
    

if __name__ == '__main__':
    # divide x and y axes by split_into. This gives split_into**2 output images.
    # a 3 by 3 grid allows pybdsf to run efficiently (fails on the 4GB 32k x 32k pixel image) whilst avoiding cutting through the centre of the image
    split_into = 3
    
    # load image to get properties
    input_image_560 = '560mhz8hours.fits'
    input_image_1400 = '1400mhz8hours.fits'
    
    # cut up images and save to disk
    do_image_chopping(input_image_560, split_into)
    do_image_chopping(input_image_1400, split_into)
    
    # make image cube of the frequencies per cutout and save to disk, so pybdsf can use spectral index mode
    make_image_cubes()
    
    # do source finding. Multithread this part? crashes. for loop is safer.             
    # sourcefinding on cube to get spectral indcies (si=True)
    imagenames = sorted(glob.glob('cube_cutout_*.fits'))
    for image in imagenames:
        do_sourcefinding(image, si=True)
                                    
                                    
    # sourcefinding on individual frequency bands
    #imagenames = glob.glob('*_cutout.fits')
    #for image in imagenames:
    #    do_sourcefinding(image)

