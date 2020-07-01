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
from astropy.convolution import convolve, Gaussian2DKernel
from astropy import units as u
from astropy.coordinates import SkyCoord
import montage_wrapper as montage

# list of functions
# load/save pickle objects
# make_2d_fits
# do_primarybeam_correction
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
    hdu.writeto(fits_image_name[:-5]+'_2d.fits', overwrite=True)
    
    
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def do_primarybeam_correction(pbname, imagename):
    print(' Preparing to apply the primary beam correction...')
    image = fits.open(imagename)[0]
    pb = fits.open(pbname)[0]
    wcs = WCS(pb.header)
    # cutout pb field of view to match image field of view
    x_size = image.header['NAXIS1']
    x_pixel_deg = image.header['CDELT2'] # CDELT1 is negative, so take positive one
    size = (x_size*x_pixel_deg*u.degree, x_size*x_pixel_deg*u.degree) # angular size of cutout, using astropy coord. approx 32768*0.6 arcseconds.
    position = SkyCoord(pb.header['CRVAL1']*u.degree, pb.header['CRVAL2']*u.degree) # RA and DEC of beam PB pointing
    print(' Cutting out image FOV from primary beam image...')
    cutout = Cutout2D(pb.data[0,0,:,:], position=position, size=size, mode='trim', wcs=wcs.celestial, copy=True)
    pb.data = cutout.data # Put the cutout image in the FITS HDU
    pb.header.update(cutout.wcs.to_header()) # Update the FITS header with the cutout WCS
    pb.writeto(pbname[:-5]+'_PBCOR.fits', overwrite=True) # Write the cutout to a new FITS file

    # regrid PB image cutout to match pixel scale of the image FOV
    print(' Regridding image...')
    # get header of image to match PB to
    montage.mGetHdr(imagename, 'hdu_tmp.hdr')
    # regrid pb image (270 pixels) to size of ref image (32k pixels)
    montage.reproject(in_images=pbname[:-5]+'_PBCOR.fits', out_images=pbname[:-5]+'_PBCOR_regrid.fits', header='hdu_tmp.hdr', exact_size=True)
    os.remove('hdu_tmp.hdr') # get rid of header text file saved to disk

    # do pb correction
    pb = fits.open(pbname[:-5]+'_PBCOR_regrid.fits')[0]
    # fix nans introduced in primary beam by montage at edges
    print(pb.data)
    mask = np.isnan(pb.data)
    pb.data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pb.data[~mask])
    image.data = image.data / pb.data
    image.writeto(imagename[:-5]+'_PBCOR.fits', overwrite=True)

    
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def crop_560MHz_to1400MHz(fits_image_name, ref_image_name):
    print(' Cropping image...')
    # Adapted from https://docs.astropy.org/en/stable/nddata/utils.html
    ref_hdu = fits.open(ref_image_name)[0] # the 1400mhz image
    hdu = fits.open(fits_image_name)[0]
    wcs = WCS(hdu.header)
    # cutout to 1400 MHz area. hard coded, since 1400 MHz beam is 2.5 times smaller than 560 MHz beam.
    x_size = ref_hdu.header['NAXIS1']
    x_pixel_deg = ref_hdu.header['CDELT2'] # CDELT1 is negative, so take positive one
    size = (x_size*x_pixel_deg*u.degree, x_size*x_pixel_deg*u.degree) # angular size of cutout, using astropy coord. approx 32768*0.6 arcseconds.
    position = SkyCoord(hdu.header['CRVAL1']*u.degree, hdu.header['CRVAL2']*u.degree) # RA and DEC of beam PB pointing
    cutout = Cutout2D(hdu.data, position=position, size=size, mode='trim', wcs=wcs, copy=True)
    hdu.data = cutout.data # Put the cutout image in the FITS HDU
    hdu.header.update(cutout.wcs.to_header()) # Update the FITS header with the cutout WCS
    hdu.writeto(fits_image_name[:-5]+'_CropTo1400mhzFOV.fits', overwrite=True) # Write the cutout to a new FITS file
    


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    


def convolve_regrid(fits_image_input, fits_image_ref): 
    # first need to convolve to match low freq resolution
    print(' Convolving image...')
    hdu1400 = fits.open(fits_image_input)[0]
    hdu560 = fits.open(fits_image_ref)[0]
    # get target resolution
    cdelt2_1400 = hdu1400.header['CDELT2']
    # current resolution
    cdelt2_560 = hdu560.header['CDELT2']
    # convolve size
    sigma = cdelt2_560/cdelt2_1400
    # By default, the Gaussian kernel will go to 4 sigma in each direction. Set to go double this.
    psf = Gaussian2DKernel(sigma/2, x_size=16, y_size=16, mode='oversample', factor=10)
    hdu1400_data_convolved = convolve(hdu1400.data, psf, boundary='extend')
    hdu1400.data = hdu1400_data_convolved
    # save convolved image
    hdu1400.writeto(fits_image_input[:-5]+'_convolved.fits', overwrite=True)
    
    # use montage to regrid image so they both have same pixel dimensions in prep for making cube
    print(' Regredding image...')
    # get header of 560 MHz image to match to
    montage.mGetHdr(fits_image_ref, 'hdu560_tmp.hdr')
    # regrid 1400 MHz cropped image (32k pixels) to 560 MHz image (13k pixels). This regrids to match 560 MHz image.
    montage.reproject(in_images=fits_image_input[:-5]+'_convolved.fits', out_images=fits_image_input[:-5]+'_convolved_regrid.fits', header='hdu560_tmp.hdr', exact_size=True)
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
    # update frequency info in the header
    hdu_new.header.set('CRPIX3', 1) # Need ref pix=1
    hdu_new.header.set('CDELT3', 840000000) # 1400 MHz - 560 MHz = 840 MHz.
    hdu_new.header.set('CRVAL3', 560000000) # ch0 freq
    hdu_new.header.set('CTYPE3', 'FREQ    ') # 3rd axis is freq
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
        collapse_mode='single', collapse_ch0=0, incl_chan=True, specind_snr=5.0, frequency_sp=[560e6, 1400e6]) # use 560 Mhz image as ch0
    # save the img object as a pickle file, so we can do interactive checks after pybdsf has run
    save_obj(img, 'pybdsf_processimage_'+imagename[:-5])
                


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    

   
if __name__ == '__main__':

    # remove extra axis and update header and wcs to be consistent
    make_2d_fits('560mhz8hours.fits')
    make_2d_fits('1400mhz8hours.fits')
    
    do_primarybeam_correction('560mhz_primarybeam.fits', '560mhz8hours_2d.fits')
    do_primarybeam_correction('1400mhz_primarybeam.fits', '1400mhz8hours_2d.fits')
    
    # crop 560 MHz image to size of 1400 MHz image
    crop_560MHz_to1400MHz('560mhz8hours_2d_PBCOR.fits', '1400mhz8hours_2d_PBCOR.fits')
    
    # Convolve and regrid 1400 MHz image to match that of the 560 MHz image. Uses Montage.
    convolve_regrid('1400mhz8hours_2d_PBCOR.fits', '560mhz8hours_2d_PBCOR_CropTo1400mhzFOV.fits')
    
    # load images now at same resolution, same sky area, same pixel size
    hdu560 = fits.open('560mhz8hours_2d_PBCOR_CropTo1400mhzFOV.fits')[0]
    hdu1400 = fits.open('1400mhz8hours_2d_PBCOR_convolved_regrid.fits')[0]
    # make image cube 'cube_560_1400.fits'
    make_image_cube(hdu560, hdu1400)
    
    # do source finding with spectral index mode on the image cube
    do_sourcefinding('cube_560_1400.fits')
    
    
    
