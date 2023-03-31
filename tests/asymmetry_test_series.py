import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing

from astropy import units as u
from photutils.aperture import EllipticalAperture, CircularAnnulus
import galsim

# from asymmetry import get_asymmetry
sys.path.append('../')
from galaxy_generator import gen_image, gen_galaxy, petrosian_sersic, create_clumps, add_source_to_image, sky_noise, petrosian_sersic
from asymmetry import get_asymmetry, get_residual

num_cores = multiprocessing.cpu_count()-3

plt.rcParams['font.size'] = 9
plt.rcParams['axes.xmargin'] = .05  # x margin.  See `axes.Axes.margins`
plt.rcParams['axes.ymargin'] = .05  # y margin.  See `axes.Axes.margins`


##### Telescope parameters
filt = 'r'
bandpass_file = "../passband_sdss_" + filt
bandpass = galsim.Bandpass(bandpass_file, wave_type = u.angstrom)
## gain, exptime and diameter of telescope
telescope_params = {'g':4.8, 't_exp':53.91, 'D':2.5}
## effective wavelength and width of filter
transmission_params = {'eff_wav':616.5, 'del_wav':137}

def get_perfect_galaxy(mag, r_eff, fov_reff=10, pxscale=0.396, sersic_n=1, q=1, beta=0):
    
    sdss_ra = 150
    sdss_dec = 2.3
    
    # Calculate field of view in degrees
    fov = fov_reff * r_eff / 3600
    
    # generate blank image with fov and wcs info
    field_image, wcs = gen_image(sdss_ra, sdss_dec, pxscale, fov, fov)

    # create a galaxy with given params
    galaxy = gen_galaxy(mag=mag, re=r_eff, n=sersic_n, q=q, beta=beta, telescope_params=telescope_params, 
                        transmission_params=transmission_params, bandpass=bandpass)
    
    # get petrosian radius of galaxy in px
    r_pet = petrosian_sersic(fov, r_eff, 1)/pxscale

    return field_image, galaxy, r_pet



def single_galaxy_run(filepath, mag, r_eff, sersic_n, q, n_clumps, sky_mag, psf_fwhm, pxscale=0.396):

    ##### Generate the galaxy image
    # Generate galaxy model
    field, galaxy, r_pet = get_perfect_galaxy(mag, r_eff, fov_reff=20, sersic_n=sersic_n, q=q)
    # generate all the clumps and their positions
    clumps, all_xi, all_yi = create_clumps(field, r_pet, n_clumps, mag, telescope_params, transmission_params, bandpass)
    # noiseless, psf_free image
    image_perfect = add_source_to_image(field, galaxy, clumps, all_xi, all_yi, psf_fwhm=0)
    # convolve sources with psf and add to image
    image_psf = add_source_to_image(field, galaxy, clumps, all_xi, all_yi, psf_fwhm)
    # add sky and poisson noise
    image_noisy, sky_flux = sky_noise(image_psf, sky_mag, pxscale, telescope_params, transmission_params, bandpass, rms_noise=True)

    image_perfect = image_perfect.array
    image_noisy = image_noisy.array



    ###### Calculate asymmetries
    a_cas_real, x0_real = get_asymmetry(image_perfect, ap_size=2*r_pet, sky_type='annulus', a_type='cas', bg_corr='residual', xtol=0.5)
    a_sq_real, _ = get_asymmetry(image_perfect, ap_size=2*r_pet, sky_type='annulus', a_type='squared', bg_corr='full', xtol=0.5)
    a_cas, x0_cas = get_asymmetry(image_noisy, ap_size=2*r_pet, sky_type='annulus', a_type='cas', bg_corr='residual', xtol=0.5)
    a_sq, x0_sq = get_asymmetry(image_noisy, ap_size=2*r_pet, sky_type='annulus', a_type='squared', bg_corr='full', xtol=0.5)
    a_sq_nocorr, _ = get_asymmetry(image_noisy, ap_size=2*r_pet, sky_type='annulus', a_type='squared', bg_corr='residual', xtol=0.5)
    
    ##### Calculate SNR
    ap_real = EllipticalAperture(x0_real, 2*r_pet, 2*r_pet)
    ap_sky = CircularAnnulus(x0_real, 2*1.5*r_pet, 2*2*r_pet)
    var = sky_flux + image_perfect
    snr = image_perfect / np.sqrt(var)
    snr_px = ap_real.do_photometry(snr)[0][0] / ap_real.do_photometry(np.ones_like(snr))[0][0]

    ##### Store output
    res = dict(a_cas_real=a_cas_real, a_sq_real=a_sq_real, a_cas=a_cas, a_sq=a_sq, a_sq_nocorr=a_sq_nocorr)
    aps = [x0_real, x0_cas, x0_sq]
    imgs = [image_perfect, image_noisy]
    inputs = dict(
        mag=mag, r_eff=r_eff, r_pet=r_pet, sersic_n=sersic_n, q=q, n_clumps=n_clumps, 
        sky_mag=sky_mag, psf=psf_fwhm, snr=snr_px
    )
    output = {
        'input' : inputs,
        'images' : imgs,
        'apertures' : aps,
        'a' : res
    }
    with open(filepath, 'wb') as f:
        pickle.dump(output, f)




if __name__ == '__main__':

    ###### Parallelize over different galaxies. For each, do a PSF and SNR series.

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="number of galaxies to generate")
    parser.add_argument("path", help="folder to store images and asymmetries in")
    args = parser.parse_args()

    ## Range of values to try
    lims = {
        'mag' : (10, 16),
        'sky_mag' : (20, 26),
        'n_clumps' : (5, 30),
        'psf_fwhm' : (0, 2),
        'sersic_n' : (1, 3),
    }

    # Generate parameters for n galaxies
    N = int(args.N)
    mags = stats.uniform.rvs(loc=lims['mag'][0], scale=lims['mag'][1] - lims['mag'][0], size=N)
    ns = stats.uniform.rvs(loc=lims['sersic_n'][0], scale=lims['sersic_n'][1] - lims['sersic_n'][0], size=N)
    sky_mags = stats.uniform.rvs(loc=lims['sky_mag'][0], scale=lims['sky_mag'][1] - lims['sky_mag'][0], size=N)
    n_clumps = np.random.randint(low=lims['n_clumps'][0], high=lims['n_clumps'][1], size=N)
    psfs = stats.uniform.rvs(loc=lims['psf_fwhm'][0], scale=lims['psf_fwhm'][1] - lims['psf_fwhm'][0], size=N)
    qs = stats.uniform.rvs(loc=0.2, scale=0.8, size=N)
    rs = -1.9*mags + 35 + stats.norm.rvs(loc=0, scale=1.5, size=N)
    rs[rs <= 1] = 1
    rs[rs >= 20] = 20

    ### Run the execution in parallel
    Parallel(n_jobs=num_cores)(delayed(single_galaxy_run)(
        filepath=f'{args.path}/{i}.pkl', mag=mags[i], r_eff=rs[i], sersic_n=ns[i],
        q=qs[i], n_clumps=n_clumps[i], sky_mag=sky_mags[i], psf_fwhm=psfs[i]
    ) for i in tqdm(range(N), total=N) )

