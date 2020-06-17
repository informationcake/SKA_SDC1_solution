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
    
    
    

