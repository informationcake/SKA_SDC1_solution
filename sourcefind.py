
import bdsf
from memory_profiler import profile

# function allows it to be profiled with @profile
def run_pybdsf(file):
    img = bdsf.process_image(file, adaptive_rms_box=True, advanced_opts=True,\
        atrous_do=False, psf_vary_do=True, psf_snrcut=5.0, psf_snrcutstack=10.0,\
        output_opts=True, output_all=True, opdir_overwrite='append', beam=(4.1666e-4, 4.1666e-4, 0),\
        blank_limit=None, thresh='hard', thresh_isl=5.0, thresh_pix=7.0, psf_snrtop=0.30)
    return img


if __name__ == '__main__':

    file = "560mhz8hours.fits"
    @profile
    run_pybdsf(file)
