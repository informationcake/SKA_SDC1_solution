# Python 3.6. Written by Alex Clarke
# Create catalogue with spectral indices from 560 and 1400 MHz images.

import os
import numpy as np
from numpy import fft
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

# list of functions:
# load/save pickle objects
# make_2d_fits
# do_primarybeam_correction
# crop_560MHz_to1400MHz
# crop_training_area
# convolve_regrid
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
    hdu.header = (WCS(hdu.header).celestial.to_header())
    '''
    hdu.header.remove('CRPIX3')
    hdu.header.remove('CRVAL3')
    hdu.header.remove('CDELT3')
    hdu.header.remove('CTYPE3')
    hdu.header.remove('CRPIX4')
    hdu.header.remove('CRVAL4')
    hdu.header.remove('CDELT4')
    hdu.header.remove('CTYPE4')
    '''
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
    print(' A small buffer of NaNs is introduced around the image by Montage when regridding to match the size, \n these have been set to the value of their nearest neighbours to maintain the same image dimensions')
    mask = np.isnan(pb.data)
    pb.data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pb.data[~mask])
    image.data = image.data / pb.data
    image.writeto(imagename[:-5]+'_PBCOR.fits', overwrite=True)

    
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def crop_560MHz_to1400MHz(imagename, ref_imagename):
    print(' Cropping 560 MHz image to 1400 MHz FOV...')
    # Adapted from https://docs.astropy.org/en/stable/nddata/utils.html
    ref_hdu = fits.open(ref_imagename)[0] # the 1400mhz image
    hdu = fits.open(imagename)[0]
    wcs = WCS(hdu.header)
    # cutout to 1400 MHz area. hard coded, since 1400 MHz beam is 2.5 times smaller than 560 MHz beam.
    x_size = ref_hdu.header['NAXIS1']
    x_pixel_deg = ref_hdu.header['CDELT2'] # CDELT1 is negative, so take positive one
    size = (x_size*x_pixel_deg*u.degree, x_size*x_pixel_deg*u.degree) # angular size of cutout, using astropy coord. approx 32768*0.6 arcseconds.
    position = SkyCoord(hdu.header['CRVAL1']*u.degree, hdu.header['CRVAL2']*u.degree) # RA and DEC of beam PB pointing
    cutout = Cutout2D(hdu.data, position=position, size=size, mode='trim', wcs=wcs, copy=True)
    hdu.data = cutout.data # Put the cutout image in the FITS HDU
    hdu.header.update(cutout.wcs.to_header()) # Update the FITS header with the cutout WCS
    hdu.writeto(imagename[:-5]+'_CropTo1400mhzFOV.fits', overwrite=True) # Write the cutout to a new FITS file
    


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    
    

def crop_training_area(imagename):
    print(' Cropping image to training area...')
    # crop area common to 560 and 1400 MHz images
    # 1400: {"ra_min": -0.2688, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.7265}
    # 560: {"ra_min": -0.6723, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.4061}
    # 1400 MHz area is common to both
    ra_min = -0.2688
    ra_max = 0.0
    dec_min = -29.9400
    dec_max = -29.7265

    hdu = fits.open(imagename)[0]
    wcs = WCS(hdu.header)

    #train_min = SkyCoord(ra=ra_min, dec=dec_min, frame="fk5", unit="deg")
    #train_max = SkyCoord(ra=ra_max, dec=dec_max, frame="fk5", unit="deg")
    #pixel_width = (abs(skycoord_to_pixel(train_max, wcs)[0] - skycoord_to_pixel(train_min, wcs)[0])* pad_factor)
    #pixel_height = (abs(skycoord_to_pixel(train_max, wcs)[1] - skycoord_to_pixel(train_min, wcs)[1])* pad_factor)

    position = SkyCoord(ra=(ra_max + ra_min) / 2, dec=(dec_max + dec_min) / 2, frame="fk5", unit="deg") # RA and DEC of centre
    # angular size of cutout, converting min/max training positions in RA and DEC to angular distance on sky
    #size = ( (dec_max - dec_min)*u.degree, (ra_max*np.cos(dec_max) - ra_min*np.cos(dec_min))*u.degree ) # order is (ny, nx)
    size = ( (dec_max - dec_min)*u.degree, (ra_max - ra_min)*u.degree ) # order is (ny, nx)
    
    cutout = Cutout2D(hdu.data, position=position, size=size, mode='trim', wcs=wcs, copy=True)
    hdu.data = cutout.data # Put the cutout image in the FITS HDU
    hdu.header.update(cutout.wcs.to_header()) # Update the FITS header with the cutout WCS
    hdu.writeto(imagename[:-5]+'_trainarea.fits', overwrite=True) # Write the cutout to a new FITS file
                


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    


def convolve_regrid(imagename, ref_imagename):
    # first need to convolve to match low freq resolution
    print(' Convolving image...')
    hdu1400 = fits.open(imagename)[0]
    hdu560 = fits.open(ref_imagename)[0]
    # degrees per pixel
    cdelt2_1400 = hdu1400.header['CDELT2']
    cdelt2_560 = hdu560.header['CDELT2']
    # how many pixels across the fwhm of 1400 MHz beam in 1400 MHz image:
    fwhm1400_pix_in1400 = hdu1400.header['BMAJ'] / cdelt2_1400 # = 2.48
    # how many pixels across the fwhm of 560 MHz beam in 1400 MHz image:
    fwhm560_pix_in1400 = hdu560.header['BMAJ'] / cdelt2_1400 # = 6.21
    # convert fwhm to sigma
    sigma1400_orig = fwhm1400_pix_in1400 / np.sqrt(8 * np.log(2))
    sigma1400_target560 = fwhm560_pix_in1400 / np.sqrt(8 * np.log(2))
    # calculate gaussian kernels (only need the 560 MHz one to convolve with)
    # By default, the Gaussian kernel will go to 8 sigma in each direction. Go much larger to get better sampling (odd number required).
    psf1400_orig = Gaussian2DKernel(sigma1400_orig, x_size=29, y_size=29, mode='oversample', factor=10)
    psf1400_target560 = Gaussian2DKernel(sigma1400_target560, x_size=29, y_size=29, mode='oversample', factor=10)

    # work out convolution kernel required to achieve 560 MHz psf in final image
    # 1400 MHz psf convolved with Y = 560 MHz psf
    # convolution in multiplication in frequency space, so:
    ft1400 = fft.fft2(psf1400_orig)
    ft560 = fft.fft2(psf1400_target560)
    ft_kernel = (ft560/ft1400)
    kernel = fft.ifft2(ft_kernel)
    kernel = fft.fftshift(kernel) # centre the kernel

    # convolve input beam with kernel to check output beam is correct, and make plot
    outbeam = convolve(psf1400_orig, kernel.real, boundary='extend') # normalising kernel is on by default.
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,3))
    #im = plt.imshow()
    im1 = ax1.imshow(psf1400_orig)
    im2 = ax2.imshow(psf1400_target560)
    im3 = ax3.imshow(kernel.real)
    im4 = ax4.imshow(outbeam)
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    plt.colorbar(im3, ax=ax3)
    plt.colorbar(im4, ax=ax4)
    plt.subplots_adjust(wspace = 0.3)
    ax1.set_title('1400 MHz PSF')
    ax2.set_title('560 MHz PSF')
    ax3.set_title('Kernel')
    ax4.set_title('1400 convolved with kernel')
    plt.savefig('convolution-kernel-'+imagename[:-5]+'.png', bbox_inches='tight')
    
    # Convolve 1400 MHz image with new kernal to get 560 MHz resolution
    hdu1400_data_convolved = convolve(hdu1400.data, kernel.real, boundary='extend') # normalising kernel is on by default.
    # update data and correct for Jy/beam scale change (proportional to change in beam area)
    hdu1400.data = hdu1400_data_convolved * ((hdu560.header['BMAJ']**2)/(hdu1400.header['BMAJ']**2))
    
    # save convolved image
    hdu1400.writeto(imagename[:-5]+'_convolved.fits', overwrite=True)
    # use montage to regrid image so they both have same pixel dimensions in prep for making cube
    print(' Regredding image...')
    # get header of 560 MHz image to match to
    montage.mGetHdr(ref_imagename, 'hdu560_tmp.hdr')
    # regrid 1400 MHz cropped image to 560 MHz image ref
    montage.reproject(in_images=imagename[:-5]+'_convolved.fits', out_images=imagename[:-5]+'_convolved_regrid.fits', header='hdu560_tmp.hdr', exact_size=True)
    os.remove('hdu560_tmp.hdr') # get rid of header text file saved to disk
    


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
   


# make image cube for pybdsf spectral index mode
def make_image_cube(imagename560, imagename1400, outfilename):
    print(' Making image cube...')
    
    # fix freq info in each image header
    hdu560 = fits.open(imagename560, mode='update')
    hdu1400 = fits.open(imagename1400, mode='update')
    hdu560[0].header.set('CRPIX3', 1) # Need ref pix=1
    hdu560[0].header.set('CRVAL3', 560000000) # ch0 freq
    hdu560[0].header.set('CTYPE3', 'FREQ    ') # 3rd axis is freq
    hdu560.flush() # write changes
    hdu1400[0].header.set('CRPIX3', 1) # Need ref pix=1
    hdu1400[0].header.set('CRVAL3', 1400000000) # ch0 freq
    hdu1400[0].header.set('CTYPE3', 'FREQ    ') # 3rd axis is freq
    hdu1400.flush() # write changes
    hdu560 = fits.open(imagename560)[0]
    hdu1400 = fits.open(imagename1400)[0]
    # make cube from the input files along freq axis
    cube = np.zeros((2, hdu560.data.shape[0], hdu560.data.shape[1]))
    cube[0,:,:] = hdu560.data[:,:] # add 560 Mhz data
    cube[1,:,:] = hdu1400.data[:,:] # add 1400 Mhz data
    hdu_new = fits.PrimaryHDU(data=cube, header=hdu560.header)
    # update frequency info in the header of cube
    hdu_new.header.set('CRPIX3', 1) # Need ref pix=1
    hdu_new.header.set('CRVAL3', 560000000) # ch0 freq
    hdu_new.header.set('CDELT3', 840000000) # 1400 MHz - 560 MHz = 840 MHz.
    hdu_new.header.set('CTYPE3', 'FREQ    ') # 3rd axis is freq
    hdu_new.writeto(outfilename, overwrite=True)
            


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------


    
def do_sourcefinding_cube(imagename, ch0=0):
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
        collapse_mode='single', collapse_ch0=ch0, incl_chan=True, specind_snr=5.0, frequency_sp=[560e6, 1400e6]) # use 560 Mhz image as ch0
    # save the img object as a pickle file, so we can do interactive checks after pybdsf has run
    # turns out this doesn't work you have to run it inside an interactive python session
    # save_obj(img, 'pybdsf_processimage_'+imagename[:-5])



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
    img = bdsf.process_image(imagename, adaptive_rms_box=False, advanced_opts=True,\
        atrous_do=False, psf_vary_do=False, psf_snrcut=5.0, psf_snrcutstack=10.0,\
        output_opts=True, output_all=True, opdir_overwrite='append', beam=(beam_maj, beam_min, beam_pa),\
        blank_limit=None, thresh='hard', thresh_isl=5.0, thresh_pix=7.0, psf_snrtop=0.30) # 
    # save the img object as a pickle file, so we can do interactive checks after pybdsf has run
    # turns out this doesn't work you have to run it inside an interactive python session
    # save_obj(img, 'pybdsf_processimage_'+imagename[:-5])
    
    
    
    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    

    
if __name__ == '__main__':

    # remove extra axis and update header and wcs to be consistent
    #make_2d_fits('560mhz8hours.fits')
    #make_2d_fits('1400mhz8hours.fits')
    
    #do_primarybeam_correction('560mhz_primarybeam.fits', '560mhz8hours_2d.fits')
    #do_primarybeam_correction('1400mhz_primarybeam.fits', '1400mhz8hours_2d.fits')
    
    
    
    ### TRAINING AREA common to 560 and 1400 MHz images ###
    # Make cutouts of the training area common to both images
    crop_training_area('1400mhz8hours_2d_PBCOR.fits')
    crop_training_area('560mhz8hours_2d_PBCOR.fits')
    # convolve and regrid to match 1400 MHz image to 560 MHz image
    convolve_regrid('1400mhz8hours_2d_PBCOR_trainarea.fits', '560mhz8hours_2d_PBCOR_trainarea.fits')
    
    # Make image cube, images now at same resolution, same sky area, same pixel size
    make_image_cube('560mhz8hours_2d_PBCOR_trainarea.fits', '1400mhz8hours_2d_PBCOR_trainarea_convolved_regrid.fits', 'cube_560_1400_train.fits')
    
    # Do sourcefinding on each image in the training area separately
    do_sourcefinding('560mhz8hours_2d_PBCOR_trainarea.fits')
    do_sourcefinding('1400mhz8hours_2d_PBCOR_trainarea_convolved_regrid.fits')
    
    # do source finding with spectral index mode on the image cube
    do_sourcefinding_cube('cube_560_1400_train.fits', ch0=0)
    do_sourcefinding_cube('cube_560_1400_train.fits', ch0=1)
    
    
    
    
    

    ### Whole image ###
    # crop 560 MHz image to size of 1400 MHz image
    #crop_560MHz_to1400MHz('560mhz8hours_2d_PBCOR.fits', '1400mhz8hours_2d_PBCOR.fits')
    # Convolve and regrid 1400 MHz image to match that of the 560 MHz image
    #convolve_regrid('1400mhz8hours_2d_PBCOR.fits', '560mhz8hours_2d_PBCOR_CropTo1400mhzFOV.fits')
    # Make image cube, images now at same resolution, same 1400 MHz sky area, same pixel size
    #make_image_cube('560mhz8hours_2d_PBCOR_CropTo1400mhzFOV.fits', '1400mhz8hours_2d_PBCOR_convolved_regrid.fits', 'cube_560_1400.fits')
    # do source finding with spectral index mode on the image cube
    #do_sourcefinding('cube_560_1400.fits')
    
    
    
    
