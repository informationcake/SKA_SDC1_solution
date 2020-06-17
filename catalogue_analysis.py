# Python 3.6. Written by Alex Clarke
# Read in catalogue, do statistics and source classification

import pickle, itertools
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import umap
import datashader as ds
import datashader.transfer_functions as tf
from datashader.transfer_functions import shade, stack
from functools import partial
from datashader.utils import export_image
from datashader.colors import *

       
       
       
       
       
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
    cat = load_obj('main_catalogue_sifted_df')
    # fix source code type weirdness
    cat['s_code_str'] = cat['S_Code'].astype('str').str[2:-1]
    # Make resolved column quantifying how extended a source is
    cat['resolved'] = ( cat['Total_flux'] - cat['Peak_flux'] ).abs()

    # Investigations for now, tidy up into functions soon...


    # Plot histogram of resolved, coloured by source type
    # S = a single-Gaussian source that is the only source in the island
    # C = a single-Gaussian source in an island with other sources
    # M = a multi-Gaussian source
    bins_res = 10 ** np.linspace(np.log10(1e-8), np.log10(10), 100)
    plt.hist(cat[cat.s_code_str=='S'].resolved, bins=bins_res, histtype='step', color='red', label='S')
    plt.hist(cat[cat.s_code_str=='C'].resolved, bins=bins_res, histtype='step', color='blue', label='C')
    plt.hist(cat[cat.s_code_str=='M'].resolved, bins=bins_res, histtype='step', color='green', label='M')
    plt.xlabel('|Total flux - Peak flux|')
    plt.ylabel('Number')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Histogram_resolved.png')


    # Plot total flux vs peak flux
    plt.scatter(cat.Total_flux, cat.Peak_flux,s=0.4)
    plt.xlabel('Total flux, Jy')
    plt.ylabel('Peak flux, Jy/beam')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('Total_Vs_Peak.png')
    
       
    # Histogram of peak and total fluxes
    bins_res = 10 ** np.linspace(np.log10(1e-8), np.log10(10), 100)
    plt.hist(cat.Total_flux, bins=bins_flux, histtype='step', label='Total flux')
    plt.hist(cat.Peak_flux, bins=bins_flux, histtype='step', label='Peak flux')
    #plt.hist(cat.Isl_Total_flux, bins=bins_flux, histtype='step', label='Island total flux')
    plt.xlabel('Jy')
    plt.ylabel('Number')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('Histogram_PeakTotal_flux.png')






    #
