# Python3.6. Written by Alex Clarke.
# Make plot of spectral index histogram after running PyBDSF on the cube.

import matplotlib.pyplot as plt
from astropy.table import Table

# load the spectral index catalogue generated from the cube
cube_cat = Table.read('cube_560_1400.pybdsm.srl.FITS')

# linear plot
df_cube_cat = cube_cat.to_pandas()
plt.hist(df_cube_cat.Spec_Indx, bins=200)
plt.xlabel('Spectral Index: 560 - 1400 MHz')
plt.ylabel('Number of sources')
plt.xlim(-2,2)
plt.title('Cube with islands detected on unity weighted average image')
plt.show()

# log y plot
plt.hist(df_cube_cat.Spec_Indx, bins=200)
plt.xlabel('Spectral Index: 560 - 1400 MHz')
plt.ylabel('Number of sources')
plt.xlim(-2,2)
plt.yscale('log')
plt.title('Cube with islands detected on unity weighted average image')
plt.show()
