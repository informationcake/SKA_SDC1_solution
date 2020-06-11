# Python3.6. Written by Alex Clarke. 
# Run PyBDSF to find sources in a fits image. Also do some hyperparameter tuning.

import bdsf
from memory_profiler import profile

# function allows it to be profiled with @profile. Run as 'mprof run sourcefind.py'. Then 'mprof plot' to get RAM vs time plot.
@profile
def do_sourcefinding(imagename):
    img = bdsf.process_image(imagename, adaptive_rms_box=True, advanced_opts=True,\
        atrous_do=False, psf_vary_do=True, psf_snrcut=5.0, psf_snrcutstack=10.0,\
        output_opts=True, output_all=True, opdir_overwrite='append', beam=(4.1666e-4, 4.1666e-4, 0),\
        blank_limit=None, thresh='hard', thresh_isl=5.0, thresh_pix=7.0, psf_snrtop=0.30)
    return img

if __name__ == '__main__':

    imagename = '560mhz8hours.fits'
    
    do_sourcefinding(imagename)
