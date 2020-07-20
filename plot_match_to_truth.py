# Python3.6. Written by Alex Clarke
# Matches PyBDSF catalogues to truth catalogue and makes plot of measured vs truth flux for each band

import matplotlib.pyplot as plt
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord


# assume catalogues are in current directory
cat560_truth = Table.read('560mhz_truthcat.csv', format='ascii.fast_no_header') # renamed .txt to .csv to please Table.read
cat1400_truth = Table.read('1400mhz_truthcat.csv', format='ascii.fast_no_header')
cat560 = Table.read('560mhz1000hours_PBCOR.pybdsm.srl.FITS') # output from PyBDSF in relevant folder
cat1400 = Table.read('1400mhz1000hours_PBCOR.pybdsm.srl.FITS')

# match to truth catalogue for 1400 MHz
c = SkyCoord(cat1400['RA'], cat1400['DEC'], unit=(u.deg,u.deg))
c_t = SkyCoord(cat_truth1400['col2'], cat_truth1400['col3'], unit=(u.deg,u.deg)) # column headers need better names. 2 and 3 are RA and DEC.
idx, sep2d, dist3d = c.match_to_catalog_sky(c_t, 1)
cat1400['idx_truth'] = idx
df_1400 = cat1400.to_pandas()
df_1400_truth = cat_truth1400.to_pandas()
df_all = df_1400.set_index('idx_truth').join(df_1400_truth)

# make plot of truth total flux against measured total flux
plt.scatter(df_all.col6, df_all.Total_flux, s=0.4) # df_all.col6 is the total flux.
xpoints = ypoints = plt.xlim() # y=x line to guide eye for flux match
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, scalex=False, scaley=False)
plt.xlabel('Truth total flux')
plt.ylabel('Measured total flux')
plt.xscale('log')
plt.yscale('log')
plt.title('1400 MHz')
plt.show()



# match to truth catalogue for 560 MHz
c = SkyCoord(cat560['RA'], cat560['DEC'], unit=(u.deg,u.deg))
c_t = SkyCoord(cat_truth560['col2'], cat_truth560['col3'], unit=(u.deg,u.deg))
idx, sep2d, dist3d = c.match_to_catalog_sky(c_t, 1)
cat560['idx_truth'] = idx
df_560 = cat560.to_pandas()
df_560_truth = cat_truth560.to_pandas()
df_all = df_560.set_index('idx_truth').join(df_560_truth)

plt.scatter(df_all.col6, df_all.Total_flux, s=0.4) # df_all.col6 is the total flux.
xpoints = ypoints = plt.xlim() # y=x line to guide eye for flux match
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=3, scalex=False, scaley=False)
plt.xlabel('Truth total flux')
plt.ylabel('Measured total flux')
plt.xscale('log')
plt.yscale('log')
plt.title('560 MHz')
plt.show()



#
