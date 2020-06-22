# Python 3.6. Written by Alex Clarke
# Combine catalogues at different frequencies, calculate spectral index

import numpy as np
import glob, matplotlib, pickle
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.table import Table, vstack, unique, setdiff
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord






  # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------






#Loading/saving python data objects
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)






    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------
    
    
    
    
    
       
if __name__ == '__main__':
    
    # Load sifted catalogue in, assuming it has been saved as a Pandas Dataframe
    cat_1400 = Table.read('main_catalogue_sifted_1400mhz.fits')
    cat_560 = Table.read('main_catalogue_sifted_560mhz.fits')

    c560 = SkyCoord(c560['RA'], c560['DEC'], unit=(u.deg,u.deg))
    c1400 = SkyCoord(c1400['RA'], c1400['DEC'], unit=(u.deg,u.deg))

    idx, sep2d, dist3d = c1400.match_to_catalog_sky(c560, 1) # idx is index of c560 that matches the source in c1400
    
    # now join tables on these indices
    cat_1400['idx_match_to_560'] = idx
    df_560 = cat_560.to_pandas()
    df_1400 = cat_1400.to_pandas()
    df_all = df_560.join(df_1400.set_index('idx_match_to_560'), lsuffix='560', rsuffix='1400')
    
    # Calculate spectral index for peak and total fluxes...
    df_all['spectral_index_peak'] = np.log(df_all['Peak_flux560']/df_all['Peak_flux1400'])/np.log(560e6/1400e6)
    df_all['spectral_index_total'] = np.log(df_all['Total_flux560']/df_all['Total_flux1400'])/np.log(560e6/1400e6)
    
    
    
    
    
    
    
    
    
    
    
    
    #
    
