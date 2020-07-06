import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from ska.sdc1.utils.image_utils import crop_to_training_area, save_subimage


class Image2d:
    def __init__(self, freq, path, pb_path, prep=False):
        self._freq = freq
        self._path = path
        self._pb_path = pb_path
        self._prep = prep

        self._segments = []
        self._train = None

    @property
    def freq(self):
        return self._freq

    @property
    def path(self):
        return self._path

    @property
    def pb_path(self):
        return self._pb_path

    @property
    def segments(self):
        return self._segments

    @property
    def train(self):
        return self._train

    def preprocess(self, split_n=1, overwrite=True):
        """
        Perform preprocessing steps:
        1) Apply PB correction
        2) Split into split_n * split_n sub-images
        3) Output separate training image
        4) Only output necessary data dimensions
        """
        # self._apply_pb_corr()
        self._split_image(split_n, overwrite)
        self._create_train(overwrite)

    def _split_image(self, split_n, overwrite):
        self._segments = []

        with fits.open(self.path) as hdu:
            im_width = hdu[0].header["NAXIS1"]
            im_height = hdu[0].header["NAXIS2"]

        # Get positions of grid centrepoints
        pos_x = (
            np.array(range(1, (split_n * 2), 2)) * (im_width / (split_n * 2))
        ).astype(int)
        pos_y = (
            np.array(range(1, (split_n * 2), 2)) * (im_height / (split_n * 2))
        ).astype(int)

        # Mesh into 2d array of coordinates
        pos_coords_pix_arr = np.array(np.meshgrid(pos_x, pos_y)).T.reshape(-1, 2)

        # Calculate image size array, with 5% buffer so images overlap.
        size_pix = int(round((im_width / split_n) * 1.05, 0))
        size_pix_arr = np.ones_like(pos_coords_pix_arr) * size_pix

        # Write subimages to disk, update self.segments
        for i, coord in enumerate(pos_coords_pix_arr):
            seg_out_path = self.path[:-5] + "_seg_{}.fits".format(i)
            save_subimage(
                self.path,
                seg_out_path,
                tuple(pos_coords_pix_arr[i]),
                tuple(size_pix_arr[i]),
                overwrite=overwrite,
            )
            self._segments.append(seg_out_path)

    def _create_train(self, pad_factor=1.0):
        self._train = None
        train_path = self.path[:-5] + "_train.fits"
        crop_to_training_area(self.path, train_path, self.freq, pad_factor)
        self._train = train_path

    def _apply_pb_corr(self):
        with fits.open(self.path)[0] as image_hdu:
            # cutout pb field of view to match image field of view
            x_size = image_hdu.header["NAXIS1"]
            x_pixel_deg = image_hdu.header[
                "CDELT2"
            ]  # CDELT1 is negative, so take positive one
        size = (
            x_size * x_pixel_deg * u.degree,
            x_size * x_pixel_deg * u.degree,
        )
        with fits.open(self.pb_path)[0] as pb_hdu:
            # RA and DEC of beam PB pointing
            pb_pos = SkyCoord(
                pb_hdu.header["CRVAL1"] * u.degree, pb_hdu.header["CRVAL2"] * u.degree
            )
        pb_cor_path = self.pb_path[:-5] + "_pb_corr.fits"
        pb_cor_rg_path = self.pb_path[:-5] + "_pb_corr_regrid.fits"
        save_subimage(self.pb_path, pb_cor_path, pb_pos, size)

        # TODO: regrid PB image cutout to match pixel scale of the image FOV
        # print(" Regridding image...")
        # # get header of image to match PB to
        # montage.mGetHdr(imagename, "hdu_tmp.hdr")
        # # regrid pb image (270 pixels) to size of ref image (32k pixels)
        # montage.reproject(
        #     in_images=pb_cor_path,
        #     out_images=pb_cor_rg_path,
        #     header="hdu_tmp.hdr",
        #     exact_size=True,
        # )
        # os.remove("hdu_tmp.hdr")  # get rid of header text file saved to disk

        # # do pb correction
        # pb = fits.open(pbname[:-5] + "_PBCOR_regrid.fits")[0]
        # # fix nans introduced in primary beam by montage at edges
        # print(pb.data)
        # mask = np.isnan(pb.data)
        # pb.data[mask] = np.interp(
        #     np.flatnonzero(mask), np.flatnonzero(~mask), pb.data[~mask]
        # )
        # image.data = image.data / pb.data
        # image.writeto(imagename[:-5] + "_PBCOR.fits", overwrite=True)

    def _delete_train(self):
        if self.train is None:
            return
        os.remove(self._train)

    def _delete_segments(self):
        for fp in self.segments:
            os.remove(fp)
