# Python 3.6. Written by Alex Clarke
# Create catalogue with spectral indices from 560 and 1400 MHz images.

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import itertools
import bdsf
import pickle
import aplpy

from matplotlib.pyplot import cm
from astropy.io import fits
#from astropy.nddata import Cutout2D
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
import montage_wrapper as montage

# list of functions
# load/save pickle objects
# make_2d_fits
# crop_560MHz_to1400MHz
# regrid_montage
# make_image_cube
# do_sourcefinding



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



#Loading/saving python data objects
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    


# remove extra axis and update header and wcs to be consistent
def make_2d_fits(fits_image_name):
    print(' Making 2d fits files...')
    hdu = fits.open(fits_image_name)[0]
    hdu.data = hdu.data[0,0,:,:]
    hdu.header.update(WCS(hdu.header).celestial.to_header())
    hdu.header.remove('CRPIX3')
    hdu.header.remove('CRVAL3')
    hdu.header.remove('CDELT3')
    hdu.header.remove('CTYPE3')
    hdu.header.remove('CRPIX4')
    hdu.header.remove('CRVAL4')
    hdu.header.remove('CDELT4')
    hdu.header.remove('CTYPE4')
    hdu.writeto(fits_image_name[:-5]+'_2d.fits')
    
    
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def crop_560MHz_to1400MHz(fits_image_name):
    print(' Cropping image...')
    # Adapted from https://docs.astropy.org/en/stable/nddata/utils.html
    hdu = fits.open(fits_image_name)[0]
    wcs = WCS(hdu.header)
    # cutout to 1400 MHz area. hard coded, since 1400 MHz beam is 2.5 times smaller than 560 MHz beam.
    cutout = Cutout2D(hdu.data, position=(32768/2,32768/2), size=(32768/2.5,32768/2.5), mode='trim', wcs=wcs, copy=True)
    hdu.data = cutout.data # Put the cutout image in the FITS HDU
    hdu.header.update(cutout.wcs.to_header()) # Update the FITS header with the cutout WCS
    hdu.writeto(fits_image_name[:-5]+'_CropTo1400mhzFOV.fits', overwrite=True) # Write the cutout to a new FITS file
    


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    


def regrid_montage(fits_image_input, fits_image_ref):
    print(' Regredding image...')
    # get header of 560 MHz image to match to
    montage.mGetHdr(fits_image_ref, 'hdu560_tmp.hdr')
    # regrid 1400 MHz cropped image (32k pixels) to 560 MHz image (13k pixels). This convolves and regrids to match 560 MHz image.
    montage.reproject(in_images=fits_image_input, out_images=fits_image_input[:-5]+'_regrid.fits', header='hdu560_tmp.hdr', exact_size=True)
    os.remove('hdu560_tmp.hdr') # get rid of header text file saved to disk
    


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    


# make image cube for pybdsf spectral index mode
def make_image_cube(hdu560, hdu1400):
    print(' Making image cube...')
    # make cube from the input files along freq axis
    cube = np.zeros((2,hdu560.data.shape[0],hdu560.data.shape[1]))
    cube[0,:,:] = hdu560.data[:,:] # add 560 Mhz data
    cube[1,:,:] = hdu1400.data[:,:] # add 1400 Mhz data
    hdu_new = fits.PrimaryHDU(data=cube, header=hdu560.header)
    # update frequency info in the header. It puts 560MHz as ch0, but incorrectly assigns the interval to the next freq channel
    hdu_new.header.set('CDELT3', 840000000) # 1400 MHz - 560 MHz = 840 MHz.
    hdu_new.writeto('cube_560_1400.fits')
            


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    


def do_sourcefinding(imagename):
    # get beam info manually. SKA image seems to cause PyBDSF issues finding this info.
    f = fits.open(imagename)
    beam_maj = f[0].header['BMAJ']
    beam_min = f[0].header['BMIN']
    #beam_pa = f[0].header['BPA'] # not in SKA fits header, but we know it's circular
    beam_pa = 0
    f.close()
    # Run sourcefinding using some sensible hyper-parameters. PSF_vary and adaptive_rms_box is more computationally intensive, off for now
    img = bdsf.process_image(imagename, adaptive_rms_box=False, spectralindex_do=True, advanced_opts=True,\
        atrous_do=False, psf_vary_do=False, psf_snrcut=5.0, psf_snrcutstack=10.0,\
        output_opts=True, output_all=True, opdir_overwrite='append', beam=(beam_maj, beam_min, beam_pa),\
        blank_limit=None, thresh='hard', thresh_isl=5.0, thresh_pix=7.0, psf_snrtop=0.30,\
        collapse_mode='single') # use 560 Mhz image as ch0
    # save the img object as a pickle file, so we can do interactive checks after pybdsf has run
    save_obj(img, 'pybdsf_processimage_'+imagename[:-5])
                


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    

   
if __name__ == '__main__':

    # remove extra axis and update header and wcs to be consistent
    make_2d_fits('560mhz8hours.fits')
    make_2d_fits('1400mhz8hours.fits')
    
    # crop 560 MHz image to size of 1400 MHz image
    crop_560MHz_to1400MHz('560mhz8hours_2d.fits')
    
    # Convolve and regrid 1400 MHz image to match that of the 560 MHz image. Uses Montage.
    regrid_montage('1400mhz8hours_2d.fits', '560mhz8hours_2d_CropTo1400mhzFOV.fits')
    
    # load images now at same resolution, same sky area, same pixel size
    hdu560 = fits.open('560mhz8hours_2d_CropTo1400mhzFOV.fits')[0]
    hdu1400 = fits.open('1400mhz8hours_2d_regrid.fits')[0]
    # make image cube 'cube_560_1400.fits'
    make_image_cube(hdu560, hdu1400)
    
    # do source finding with spectral index mode on the image cube
    do_sourcefinding('cube_560_1400.fits')
    
    
    
