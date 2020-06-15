from astropy.table import Table, vstack, unique
import glob, matplotlib
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm


def combine_cats():
