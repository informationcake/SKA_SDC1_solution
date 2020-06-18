# Python 3.6. Written by Alex Clarke
# Read in catalogue, do statistics and source classification

import pickle, itertools
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
	cat['fratio'] = ( cat['Total_flux'] / cat['Peak_flux'] )

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

	# peak/total ratio
	plt.hist(cat.fratio, bins=200, histtype='step', label='All')
	plt.hist(cat[cat.s_code_str=='S'].fratio, bins=200, histtype='step', color='red', label='S')
	plt.hist(cat[cat.s_code_str=='C'].fratio, bins=200, histtype='step', color='blue', label='C')
	plt.hist(cat[cat.s_code_str=='M'].fratio, bins=200, histtype='step', color='green', label='M')
	plt.xlabel('Total flux / Peak flux')
	plt.ylabel('Number')
	plt.xlim(0,15)
	plt.legend()
	plt.yscale('log')
	plt.show()
	
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


    # Source axis, Maj vs Min
    plt.scatter(cat[cat.s_code_str=='S'].Maj*3600, cat[cat.s_code_str=='S'].Min*3600, s=0.3, label='S')
    plt.scatter(cat[cat.s_code_str=='C'].Maj*3600, cat[cat.s_code_str=='C'].Min*3600, s=0.3, label='C')
    plt.scatter(cat[cat.s_code_str=='M'].Maj*3600, cat[cat.s_code_str=='M'].Min*3600, s=0.3, label='M')
    plt.xlabel('Source major axis, arcsec')
    plt.ylabel('Source minor axis, arcsec')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('MajVsMin_sourceaxis.png')

    plt.scatter(cat.Maj*3600, cat.Min*3600, s=0.3, c=cat.resolved)
	plt.colorbar()
	plt.show()

    # create png
	cvs = ds.Canvas(plot_width=200, plot_height=200)
	agg = cvs.points(cat, 'Maj', 'Min', ds.mean('resolved'))
	img = tf.shade(agg, how='log')
	#export_image(img, 'DS-majmin', fmt='.png', background='white')

	# generate figure with png created and append colourbar axis
	fig = plt.figure(figsize=(10,10)) # y axis larger to fit cbar in
	#img = img.imread('DS-majmin.png')
	plt.imshow(img)
	plt.show()
    #
	
	
	
	#
	
	data_cols = ['Total_flux', 'Peak_flux', 'Maj', 'Min', 'Isl_Total_flux', 'resolved']
	
	
	
	#
	
	
	
	
