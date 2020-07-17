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



def do_primarybeam_correction(pbname, imagename):
    print(' Preparing to apply the primary beam correction to {0}'.format(imagename))
    hdu = fits.open(imagename)[0]
    pb = fits.open(pbname)[0]
    wcs = WCS(pb.header)

    # cutout pb field of view to match image field of view
    x_size = hdu.header['NAXIS1']
    x_pixel_deg = hdu.header['CDELT2'] # CDELT1 is negative, so take positive one
    size = (x_size*x_pixel_deg*u.degree, x_size*x_pixel_deg*u.degree) # angular size of cutout, using astropy coord. approx 32768*0.6 arcseconds.
    position = SkyCoord(pb.header['CRVAL1']*u.degree, pb.header['CRVAL2']*u.degree) # RA and DEC of beam PB pointing
    print(' Cutting out image FOV from primary beam image...')
    cutout = Cutout2D(pb.data[0,0,:,:], position=position, size=size, mode='trim', wcs=wcs.celestial, copy=True)

    # Update the FITS header with the cutout WCS by hand using my own function
    # don't use cutout.wcs.to_header() because it doesn't account for the freq and stokes axes. is only compatible with 2D fits images.
    #pb.header.update(cutout.wcs.to_header()) #
    pb = update_header_from_cutout2D(pb, cutout)
    # write updated fits file to disk
    pb.writeto(pbname[:-5]+'_cutout.fits', overwrite=True) # Write the cutout to a new FITS file

    # regrid PB image cutout to match pixel scale of the image FOV
    print(' Regridding image...')
    # get header of image to match PB to
    montage.mGetHdr(imagename, 'hdu_tmp.hdr')
    # regrid pb image (270 pixels) to size of ref image (32k pixels)
    montage.reproject(in_images=pbname[:-5]+'_cutout.fits', out_images=pbname[:-5]+'_cutout_regrid.fits', header='hdu_tmp.hdr', exact_size=True)
    os.remove('hdu_tmp.hdr') # get rid of header text file saved to disk

    # update montage output to float32
    pb = fits.open(pbname[:-5]+'_cutout_regrid.fits', mode='update')
    newdata = np.zeros((1,1,pb[0].data.shape[0], pb[0].data.shape[1]), dtype=np.float32)
    newdata[0,0,:,:] = pb[0].data
    pb[0].data = newdata # naxis will automatically update to 4 in the header

    # fix nans introduced in primary beam by montage at edges and write to new file
    print(' A small buffer of NaNs is introduced around the image by Montage when regridding to match the size, \n these have been set to the value of their nearest neighbours to maintain the same image dimensions')
    mask = np.isnan(pb[0].data)
    pb[0].data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), pb[0].data[~mask])
    pb.flush()
    pb.close()

    # apply primary beam correction
    pb = fits.open(pbname[:-5]+'_cutout_regrid.fits')[0]
    hdu.data = hdu.data / pb.data
    hdu.writeto(imagename[:-5]+'_PBCOR.fits', overwrite=True)
    print(' Primary beam correction applied to {0}'.format(imagename[:-5]+'_PBCOR.fits') )



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def crop_images(imagename_560, imagename_1400):
    # Using Cutout2D from https://docs.astropy.org/en/stable/nddata/utils.html
    # Cropping 1400 and 560 MHz images to smaller FOV
    # take 1400 MHz FOV and crop by a further 7000 pixels (chosen by eye) to remove most of the primary beam noise
    print(' Cropping 560 and 1400 MHz images to same field of view...')
    hdu560 = fits.open(imagename_560)[0]
    hdu1400 = fits.open(imagename_1400)[0]
    wcs560 = WCS(hdu560.header)
    wcs1400 = WCS(hdu1400.header)
    # cutout to 1400 MHz area. hard coded, since 1400 MHz beam is 2.5 times smaller than 560 MHz beam.
    x_size = hdu1400.header['NAXIS1']
    x_size = x_size - 14000 # crop in a further 7000 pixels each side
    x_pixel_deg = hdu1400.header['CDELT2'] # CDELT1 is negative, so take positive one
    size = (x_size*x_pixel_deg*u.degree, x_size*x_pixel_deg*u.degree) # angular size of cutout, using astropy coord. approx 32768*0.6 arcseconds.
    position = SkyCoord(hdu560.header['CRVAL1']*u.degree, hdu560.header['CRVAL2']*u.degree) # RA and DEC of beam PB pointing

    # crop 560 MHz image
    cutout = Cutout2D(hdu560.data[0,0,:,:], position=position, size=size, mode='trim', wcs=wcs560.celestial, copy=True)
    # Update the FITS header with the cutout WCS by hand using my own function
    hdu560 = update_header_from_cutout2D(hdu560, cutout)
    hdu560.writeto(imagename_560[:-5]+'_crop.fits', overwrite=True) # Write the cutout to a new FITS file

    # crop 1400 MHz image
    cutout = Cutout2D(hdu1400.data[0,0,:,:], position=position, size=size, mode='trim', wcs=wcs1400.celestial, copy=True)
    # Update the FITS header with the cutout WCS by hand using my own function
    hdu1400 = update_header_from_cutout2D(hdu1400, cutout)
    hdu1400.writeto(imagename_1400[:-5]+'_crop.fits', overwrite=True) # Write the cutout to a new FITS file



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
    #CRVAL3 = hdu.header['CRVAL3']
    wcs = WCS(hdu.header)

    position = SkyCoord(ra=(ra_max + ra_min) / 2, dec=(dec_max + dec_min) / 2, frame="fk5", unit="deg") # RA and DEC of centre
    # angular size of cutout, converting min/max training positions in RA and DEC to angular distance on sky
    #size = ( (dec_max - dec_min)*u.degree, (ra_max*np.cos(dec_max) - ra_min*np.cos(dec_min))*u.degree ) # order is (ny, nx)
    size = ( (dec_max - dec_min)*u.degree, (ra_max - ra_min)*u.degree ) # order is (ny, nx)

    cutout = Cutout2D(hdu.data[0,0,:,:], position=position, size=size, mode='trim', wcs=wcs.celestial, copy=True)
    # Update the FITS header with the cutout WCS by hand using my own function
    hdu = update_header_from_cutout2D(hdu, cutout)
    hdu.writeto(imagename[:-5]+'_trainarea.fits', overwrite=True) # Write the cutout to a new FITS file



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def convolve_regrid(imagename, ref_imagename, make_beam_plot=True):
    # first need to convolve to match low freq resolution
    print(' Convolving image...')
    hdu1400 = fits.open(imagename)[0]
    hdu560 = fits.open(ref_imagename)[0]
    #CRVAL3 = hdu1400.header['CRVAL3']
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

    # convolve input beam with kernel to check output beam is correct, and make plot?
    if make_beam_plot==True:
        outbeam = convolve(psf1400_orig, kernel.real, boundary='extend') # normalising kernel is on by default.
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20,3))
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
    # convolve only works with 2D arrays
    hdu1400_data_convolved = convolve(hdu1400.data[0,0,:,:], kernel.real, boundary='extend') # normalising kernel is on by default.
    # correct for Jy/beam scale change (proportional to change in beam area)
    hdu1400_data_convolved = hdu1400_data_convolved * ((hdu560.header['BMAJ']**2)/(hdu1400.header['BMAJ']**2))
    # add back to header in 4D
    newdata = np.zeros((1,1,hdu1400_data_convolved.data.shape[0], hdu1400_data_convolved.data.shape[1]), dtype=np.float32)
    newdata[0,0,:,:] = hdu1400_data_convolved
    hdu1400.data = newdata # data now 4D
    hdu1400.writeto(imagename[:-5]+'_convolved.fits', overwrite=True) # save to disk

    # use montage to regrid image so they both have same pixel dimensions in prep for making cube
    print(' Regredding image...')
    # get header of 560 MHz image to match to
    montage.mGetHdr(ref_imagename, 'hdu560_tmp.hdr')
    # regrid 1400 MHz cropped image to 560 MHz image ref
    montage.reproject(in_images=imagename[:-5]+'_convolved.fits', out_images=imagename[:-5]+'_convolved_regrid.fits', header='hdu560_tmp.hdr', exact_size=True)
    os.remove('hdu560_tmp.hdr') # get rid of header text file saved to disk

    # montage got rid of dtype info, and made the data 2D, fix these:
    hdu = fits.open(imagename[:-5]+'_convolved_regrid.fits', mode='update')
    newdata = np.zeros((1,1,hdu[0].data.shape[0], hdu[0].data.shape[1]), dtype=np.float32)
    newdata[0,0,:,:] = hdu[0].data
    hdu[0].data = newdata # naxis will automatically update to 4 in the header
    hdu.flush() # writes changes back as opened in 'update' mode.
    # note that hdu.close() fails because montage changes order of header cards.
    # hdu.flush() appears to have built in tests which correct header card order before saving it to disk.



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



# make image cube for pybdsf spectral index mode and update freq info in headers
def make_image_cube(imagename560, imagename1400, outfilename):
    # make cube from the input files along freq axis
    print(' Making image cube...')
    hdu560 = fits.open(imagename560)[0]
    hdu1400 = fits.open(imagename1400)[0]
    cube = np.zeros((2, hdu560.data[0,0,:,:].shape[0], hdu560.data[0,0,:,:].shape[1]))
    cube[0,:,:] = hdu560.data[0,0,:,:] # add 560 Mhz data
    cube[1,:,:] = hdu1400.data[0,0,:,:] # add 1400 Mhz data
    hdu_cube = fits.PrimaryHDU(data=cube, header=hdu560.header)
    # update frequency info in the header of cube
    hdu_cube.header.set('CRPIX3', 1) # Need ref pix=1
    hdu_cube.header.set('CRVAL3', 560000000) # ch0 freq
    hdu_cube.header.set('CDELT3', 840000000) # 1400 MHz - 560 MHz = 840 MHz.
    hdu_cube.header.set('CTYPE3', 'FREQ    ') # 3rd axis is freq
    hdu_cube.writeto(outfilename, overwrite=True)



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def do_sourcefinding_cube(imagename, collapse_mode='average', ch0=0):
    # get beam info manually. SKA image seems to cause PyBDSF issues finding this info.
    hdu = fits.open(imagename)
    beam_maj = hdu[0].header['BMAJ']
    beam_min = hdu[0].header['BMIN']
    #beam_pa = f[0].header['BPA'] # not in SKA fits header, but we know it's circular
    beam_pa = 0
    hdu.close()
    # Run sourcefinding using some sensible hyper-parameters. PSF_vary and adaptive_rms_box is more computationally intensive, off for now
    img = bdsf.process_image(imagename, adaptive_rms_box=False, spectralindex_do=True, advanced_opts=True,\
        atrous_do=False, psf_vary_do=False, psf_snrcut=5.0, psf_snrcutstack=10.0,\
        output_opts=True, output_all=True, opdir_overwrite='append', beam=(beam_maj, beam_min, beam_pa),\
        blank_limit=None, thresh='hard', thresh_isl=5.0, thresh_pix=7.0, psf_snrtop=0.30,\
        collapse_mode=collapse_mode, collapse_ch0=ch0, collapse_wt='unity',\
        incl_chan=True, specind_snr=5.0, frequency_sp=[560e6, 1400e6]) # use 560 Mhz image as ch0
    # save the img object as a pickle file, so we can do interactive checks after pybdsf has run
    # turns out this doesn't work you have to run it inside an interactive python session
    # save_obj(img, 'pybdsf_processimage_'+imagename[:-5])



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



def do_sourcefinding(imagename):
    # get beam info manually. SKA image seems to cause PyBDSF issues finding this info.
    hdu = fits.open(imagename, mode='update')
    beam_maj = hdu[0].header['BMAJ']
    beam_min = hdu[0].header['BMIN']
    #beam_pa = hdu[0].header['BPA'] # not in SKA fits header, but we know it's circular
    beam_pa = 0
    hdu.close()
    # Run sourcefinding using some sensible hyper-parameters. PSF_vary and adaptive_rms_box is more computationally intensive, off for now
    img = bdsf.process_image(imagename, adaptive_rms_box=False, advanced_opts=True,\
        atrous_do=False, psf_vary_do=False, psf_snrcut=5.0, psf_snrcutstack=10.0,\
        output_opts=True, output_all=True, opdir_overwrite='append', beam=(beam_maj, beam_min, beam_pa),\
        blank_limit=None, thresh='hard', thresh_isl=4.0, thresh_pix=5.0, psf_snrtop=0.30,\
        do_cache=True) #
    # save the img object as a pickle file, so we can do interactive checks after pybdsf has run
    # turns out this doesn't work you have to run it inside an interactive python session
    # save_obj(img, 'pybdsf_processimage_'+imagename[:-5])



    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------



if __name__ == '__main__':

    # Applying primary beam correction
    do_primarybeam_correction('560mhz_primarybeam.fits', '560mhz1000hours.fits')
    do_primarybeam_correction('1400mhz_primarybeam.fits', '1400mhz1000hours.fits')

    #####################################################################
    ### TRAINING AREA ###
    # common to 560 and 1400 MHz images ###
    # Make cutouts of the training area common to both images
    crop_training_area('1400mhz1000hours_PBCOR.fits')
    crop_training_area('560mhz1000hours_PBCOR.fits')
    # convolve and regrid to match 1400 MHz image to 560 MHz image
    convolve_regrid('1400mhz1000hours_PBCOR_trainarea.fits', '560mhz1000hours_PBCOR_trainarea.fits', make_beam_plot=True)
    # Make image cube, images now at same resolution, same sky area, same pixel size
    make_image_cube('560mhz1000hours_PBCOR_trainarea.fits', '1400mhz1000hours_PBCOR_trainarea_convolved_regrid.fits', 'cube_560_1400_train.fits')
    # Do sourcefinding on each image in the training area separately
    do_sourcefinding('560mhz1000hours_PBCOR_trainarea.fits')
    do_sourcefinding('1400mhz1000hours_PBCOR_trainarea_convolved_regrid.fits')

    # do source finding with spectral index mode on the image cube
    do_sourcefinding_cube('cube_560_1400_train.fits', collapse_mode='average')
    # collapse_mode='average' means sourcefinding is done on an averaged image so spectral indices are consistent.
    # this means Total_flux and Peak_flux are calculated at (560+1400/2) MHz. However, Total_flux_ch1 and Total_flux_ch2 give fluxes at 560 and 1400 MHz respectively.
    # Currently in spectral index mode peak_fluxes are not returned per freq band.
    # currently the log incorrectly says the ch0=1 is 560 MHz instead of 1400, i think this is a bug as Spec_Indx is correctly calculated. I may have the cube header in an unexpected format?



    #####################################################################
    ### Whole image ###
    # Do sourcefinding on each image
    do_sourcefinding('560mhz1000hours_PBCOR.fits')
    do_sourcefinding('1400mhz1000hours_PBCOR.fits')

    # Now make image cube for PyBDSF to calculate spectral indices where images overlap
    # Crop 560 and 1400 MHz images to same field of view, also removing most primary beam noise
    crop_images('560mhz1000hours_PBCOR.fits', '1400mhz1000hours_PBCOR.fits')
    # Convolve and regrid 1400 MHz image to match that of the 560 MHz image
    convolve_regrid('1400mhz1000hours_PBCOR_crop.fits', '560mhz1000hours_PBCOR_crop.fits', make_beam_plot=True)
    # Make image cube, images now at same resolution, same 1400 MHz sky area, same pixel size
    make_image_cube('560mhz1000hours_PBCOR_crop.fits', '1400mhz1000hours_PBCOR_crop_convolved_regrid.fits', 'cube_560_1400.fits')
    # do source finding with spectral index mode on the image cube
    # same caveats as before
    do_sourcefinding_cube('cube_560_1400.fits', collapse_mode='average')



    # notes
    '''
    Currently there is no peak_flux available at 560 and 1400 MHz for sources when using PyBDSF spectral index mode on the fits cube. There is only total_flux and peak_flux at (560+1400)/2 MHz, and total_flux_ch1 (560 MHz), total_flux_ch2 (1400 MHz). I can't find the relevant PyBDSF parameter to return peak_flux per channel... perhaps i'm still missing it. A work around is to cross-match the spectral index catalogue with the individual band catalogues, and get peak fluxes from the matches.

    Note that you can run PyBDSF spectral index mode without averaging the two frequencies, and just use one of the frequencies as the map to find islands on. In this case, total and peak flux is returned per frequency. But, this is not a consistent way to get spectral indices, since the effective apperture is different depending on your choice of using ch0 or ch1 to find islands.

    To do: I need to test using a 'detection_image', instead of using the 'collapse_mode' parameter. This should mean a consistent image is used for finding source islands (the detection image, which i would make by hand averaging the 560 and 1400 MHz images), and total and peak fluxes are extracted per source.
    '''


    #
