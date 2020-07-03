import os

from astropy.io import fits

import bdsf


class SourceFinder:
    """
    Find sources using PyBDSF
    """

    def __init__(self, image_path, image_name):
        self.image_path = image_path
        self.image_name = image_name
        self._run_complete = False
        self.beam_maj = None
        self.beam_min = None
        self.beam_pa = None

    def run(self):
        self._run_complete = False
        os.chdir(self.image_path)
        self.set_beam_props_from_hdu()
        img = bdsf.process_image(
            self.image_name,
            adaptive_rms_box=True,
            advanced_opts=True,
            atrous_do=False,
            psf_vary_do=True,
            psf_snrcut=5.0,
            psf_snrcutstack=10.0,
            output_opts=True,
            output_all=True,
            opdir_overwrite="append",
            beam=(self.beam_maj, self.beam_min, self.beam_pa),
            blank_limit=None,
            thresh="hard",
            thresh_isl=5.0,
            thresh_pix=7.0,
            psf_snrtop=0.30,
        )
        self._run_complete = True
        return  # img

    def set_beam_props(self, beam_maj=None, beam_min=None, beam_pa=None):
        self.beam_maj = beam_maj
        self.beam_min = beam_min
        self.beam_pa = beam_pa

    def set_beam_props_from_hdu(self):
        try:
            with fits.open(self.image_name) as hdu:
                self.beam_maj = hdu[0].header["BMAJ"]
                self.beam_min = hdu[0].header["BMIN"]
                self.beam_pa = 0
        except:
            raise Exception()

    def get_sources(self):
        return
