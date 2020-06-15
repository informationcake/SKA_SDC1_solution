# Python 3.6. Written by Alex Clarke
# Combine catalogues from image cutouts, sift to remove duplicates, create quality control figures and plots

from astropy.table import Table, vstack, unique
import glob, matplotlib
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import simple_norm




  

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

    
    
    
    
    
    def sift_catalogue():


      
      
  

    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    
    
    
    


def plot_test(filename, master_catalogue, zoomin=True):
    #filename = '560mhz8hours.fits'
    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header).celestial # get data wcs
    data = hdu.data[0,0,:,:] # get data axes
    data[data<1e-7] = 1e-7 # min pixel brightness to display
    data[data>1e-4] = 1e-4 # max pixel brightness to display
    ax = plt.subplot(projection=wcs)
    norm = simple_norm(data, 'log')
    ax.imshow(data, norm=norm)
    ax.scatter(master_catalogue['RA'], master_catalogue['DEC'], transform=ax.get_transform('fk5'), s=300, edgecolor='white', facecolor='none')
    plt.colorbar(im)
    if zoomin==True:
        ax.axis([15000,16000,15000,16000],transform=ax.get_transform('world')) # zoom in?
    plt.savefig('test_imagecentre.png')



    
  

    # ------ ------ ------ ------ ------ ------ ------ ------ ------ ------

    
    
    

    
if __name__ == '__main__':
    
    # Copy catalogues (*.srl.FITS) into a common directory with the main fits image (32k x 32k)
    
    # Combine catalgoues into single FITS catalogues
    master_catalogue = combine_cats()
    
    # Sift catalogue to remove duplicate matches
    #master_catalogue_sifted = sift_catalogue()
    
    
    # plot field and overlay catalogue sources
    filename = '560mhz8hours.fits' # place in directory
    plot_test(filename, master_catalogue, zoomin=True)
    plot_test(filename, master_catalogue, zoomin=False)
    
    
    
    
    
