# Python 3.6. Written by Alex Clarke
# Combine catalogues from image cutouts, sift to remove duplicates, create quality control figures and plots

import numpy as np
import glob, matplotlib
from matplotlib import pyplot as plt

from astropy import units as u
from astropy.table import Table, vstack, unique, setdiff
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord




  

    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    
    
    
    

def combine_cats():
    catalogues = glob.glob('*.srl.FITS')
    # initialise master catalogue and append to it
    master_catalogue = Table.read(catalogues[0])
    for catalogue in catalogues[1:]:
        print(catalogue)
        master_catalogue = vstack([master_catalogue,Table.read(catalogue)], join_type='exact')
    return master_catalogue




  

    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    
    
    
    
    
def sift_catalogue(cat, tolerance, iterations):
    # sift catalogue several times to account for matches in up to 4 images
    print(' There are initially {0} sources...'.format(len(cat)))
    cat_original = Table(cat) # make copy for reference
    # iterate since there could be more than one duplicate match per source. square fields = up to 4 duplicate matches from overlaps?
    for i in range(iterations):
        n_init = len(cat)
        print(' Iteration {0}...'.format(i))
        # Generate list of coordinates from catalogue
        c = SkyCoord(cat['RA'], cat['DEC'], unit=(u.deg,u.deg))
        # Determining the nearest neighbour of each source
        idx, sep2d, dist3d = c.match_to_catalog_sky(c, 2) # nthneighborint=2 since we are matching a cat to itself
        # Identifying sources closer than the tolerance threshold, which are not from the same mosaic
        inds = np.nonzero( (sep2d < tolerance*u.deg) )[0] # indices where separation is less than tolerance
        cat.remove_rows(idx[inds]) # sifted catalogue, removing duplicates
        print(' Removed {0} sources this iteration, leaving {1} sources'.format(n_init - len(cat), len(cat) ))
        
    cat_deleted = setdiff(cat_original, cat) # see deleted sources
    print(' Removed {0} sources in total, leaving {1} in the catalogue'.format(len(cat_deleted), len(cat)))
    return cat, cat_deleted
      
      
      
  


    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    
    
    


def plot_test(filename, cat, label='', zoomin=True):
    #filename = '560mhz8hours.fits'
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header).celestial # get data wcs
    data = hdu.data[0,0,:,:] # get data axes
    data[data<1e-7] = 1e-7 # min pixel brightness to display
    data[data>1e-4] = 1e-4 # max pixel brightness to display
    ax = plt.subplot(projection=wcs)
    norm = simple_norm(data, 'log')
    im = ax.imshow(data, norm=norm)
    ax.scatter(cat['RA'], cat['DEC'], transform=ax.get_transform('fk5'), s=300, edgecolor='white', facecolor='none')
    plt.colorbar(im)
    if zoomin==True:
        ax.axis([15000,16000,15000,16000],transform=ax.get_transform('world')) # zoom in?
        #ax.axis([ra1,ra2,dec1,dec2],transform=ax.get_transform('fk5')) # ra/dec deciaml degrees coords
    #plt.show()
    plt.savefig('image'+label+'.png')



    
  

    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    
    
    

    
if __name__ == '__main__':
    
    # Copy catalogues (*.srl.FITS) into a common directory with the main fits image (32k x 32k)
    
    # Combine catalgoues into single FITS catalogues
    master_catalogue = combine_cats()

    # Sift catalogue to remove duplicate matches
    master_catalogue_sifted, removed_cat = sift_catalogue(master_catalogue, tolerance=0.5/3600, iterations=5) # tolerance=0.5 arcsec
    #print('There are {0} sources, after removing {1} duplicates'.format( len(master_catalogue_sifted), len(removed_cat) ) )
    
    # plot field and overlay catalogue sources
    filename = '560mhz8hours.fits' # place in directory
    plot_test(filename, master_catalogue_sifted, zoomin=True, label='sifted')
    plot_test(filename, master_catalogue_sifted, zoomin=False, label='sifted_zoom')
    plot_test(filename, removed_cat, zoomin=False, label='removed_duplicate_sources')
    
    
    
    
    
