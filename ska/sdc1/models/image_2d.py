import numpy as np
from astropy.io import fits

from ska.sdc1.utils.image_utils import save_subimage


class Image2d:
    def __init__(self, freq, path, pb_path, prep=False):
        self._freq = freq
        self._path = path
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
        self._split_image(split_n, overwrite)

    def _split_image(self, split_n, overwrite):
        with fits.open(self.path) as hdu:
            im_width = hdu[0].header["NAXIS1"]
            im_height = hdu[0].header["NAXIS2"]

        # Get positions of subgrid centrepoints
        pos_x = (
            np.array(range(1, (split_n * 2), 2)) * (im_width / (split_n * 2))
        ).astype(int)
        pos_y = (
            np.array(range(1, (split_n * 2), 2)) * (im_height / (split_n * 2))
        ).astype(int)

        # Mesh into 2d array of coordinates
        pos_coords_pix_arr = np.array(np.meshgrid(pos_x, pos_y)).T.reshape(-1, 2)
        print(pos_coords_pix_arr)

        # Calculate image size array, with 5% buffer so images overlap.
        size_pix = int(round((im_width / split_n) * 1.05, 0))
        size_pix_arr = np.ones_like(pos_coords_pix_arr) * size_pix
        print(len(pos_coords_pix_arr))

        for i, coord in enumerate(pos_coords_pix_arr):
            save_subimage(
                self.path,
                self.path[:-5] + "_seg_{}.fits".format(i),
                tuple(pos_coords_pix_arr[i]),
                tuple(size_pix_arr[i]),
                overwrite=overwrite,
            )
