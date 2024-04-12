#!/usr/bin/python
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares
from matplotlib.colors import LogNorm
from datetime import datetime

startTime = datetime.now()

# Setup matplotlib configurations
plt.rcParams.update({
    'lines.linewidth': 2,
    'font.family': 'serif',
    'font.serif': 'serif',
    'text.usetex': True,
    'font.size': 25,
    'figure.max_open_warning': 0
})

# Global settings
fitting = True
sed_plotting = False
plotting = True

savefits = "NGC7538_beta_2p0_220_220px-new2"
os.makedirs("SED_plots_optimized", exist_ok=True)

# Open FITS files once and reuse
pacs70_data = fits.open('ngc7538_70mu_cgs.fits')[0].data
pacs160_data = fits.open('ngc7538_160mu_cgs.fits')[0].data
spire250_data = fits.open('ngc7538_250mu_cgs.fits')[0].data
spire350_data = fits.open('ngc7538_350mu_cgs.fits')[0].data
header = fits.open('ngc7538_350mu_cgs.fits')[0].header

# Planck function
def B(x, T):
    return 2*6.626e-27*x**3/9e20/(np.exp(6.626e-27*x/1.38e-16/T)-1)

# Residual function for fitting
def res(p, x, y):
    beta = 2
    return y - np.log(B(x, p[0]) * (1 - np.exp(-p[1] * (160e-4/3e10*x)**beta)))

# Prepare for fitting
numpix_x, numpix_y = header["NAXIS1"], header["NAXIS2"]  # Changed to 1 for testing
output = np.zeros((3, numpix_x, numpix_y))

# Main processing loop
if fitting:
    for k in range(numpix_x):
        for l in range(numpix_y):

            freq = np.array([4.28E+12, 1.87E+12, 1.199E+12, 8.565E+11])
            sed = np.array([pacs70_data[k, l], pacs160_data[k, l], spire250_data[k, l], spire350_data[k, l]])
            lam = np.array([70., 160., 250., 350.])

            p0 = [20., 6.3e-3, 1.]
            popt = least_squares(res, p0, args=(3.e10/lam/1e-4, np.log(sed))).x
            output[:, k, l] = [popt[0], popt[1], 2]
            
            print(k, l, popt[0], popt[1])
            
            fig = plt.figure(figsize=(10,7), dpi=100)
            ax1 = fig.add_subplot(111)
            freq1 = np.arange(2e11, 9e12, 1.5e10)
            
            if sed_plotting:
                ax1.scatter(freq, sed, color = 'black', label = "70, 160, 250, 350")
                ax1.plot(freq1, B(freq1, popt[0]) * popt[1] * (freq1/1.87E+12)**2, color = 'red')
                plt.xlabel(r"Frequency (Hz)")
                plt.ylabel("Flux (erg/s/cm$^2$/Hz/sr)")
                ax1.invert_xaxis()
                plt.xlim(freq1.max() * 1.5, freq1.min() * 0.75)
                plt.yscale('log')
                plt.xscale('log')
                plt.legend(loc = 'best')
                plt.title("SED ({})".format(k, l))
                fig.savefig("SED_plots_optimized/SED_{}".format((k,l))+".png")
            else:
                continue

fits.writeto(savefits + ".fits", output, header, overwrite=True)

# Generate plots if enabled
if plotting:
    SED_mybubble = fits.open(savefits + '.fits')[0].data
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=100)
    titles = ["$T_{\mathrm{dust}}$ (K)", "$\\tau_{160\,\\mu m}$"]

    for i, ax in enumerate(axes):
        # Ensure data has only positive values for LogNorm
        data = SED_mybubble[i]
        data[data <= 0] = np.min(data[data > 0])  # replace non-positive values with the smallest positive value

        im = ax.imshow(data, cmap="viridis", origin="lower", norm=LogNorm(vmin=data.min(), vmax=data.max()))
        fig.colorbar(im, ax=ax, pad=0.0, fraction=.08)
        ax.set(title=titles[i], xlabel=r"RA (J2000)", ylabel=r"Dec (J2000)")
    
    fig.tight_layout()
    plt.savefig(savefits + ".pdf", dpi=300)

print('The time you spent (h:mm:ss.s) is:', datetime.now() - startTime)
print("Good job! At least, your code has worked without any interruption.")



