import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing
from skimage.transform import rescale

from astropy import units as u
from photutils.aperture import EllipticalAperture, CircularAnnulus, CircularAperture, ApertureStats
import galsim

# from asymmetry import get_asymmetry
sys.path.append('../')
from galaxy_generator import simulate_perfect_galaxy, add_source_to_image, sky_noise
from asymmetry import _asymmetry_func, fourier_deconvolve, _asymmetry_center, _sky_properties
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel



num_cores = multiprocessing.cpu_count()

def make_galaxy(mag, r_eff, psf_fwhm, sky_mag, n_clumps, fov_reff, pxscale, sersic_n, q, beta):
    

    perfect_img, params, rpet = simulate_perfect_galaxy(mag=mag, r_eff=r_eff, pxscale=pxscale, fov_reff=fov_reff, sersic_n=sersic_n, 
                                                    q=q,beta=beta, n_clumps=n_clumps)
    
    image_psf = add_source_to_image(**params, psf_fwhm=psf_fwhm, pxscale=pxscale, psf_method="astropy")
    image_noise, noise = sky_noise(image_psf, sky_mag=sky_mag, pixel_scale=pxscale, rms_noise=True)
    return perfect_img, image_noise, rpet



def single_galaxy_run(filepath, mag, r_eff, sersic_n, q, beta, n_clumps, sky_mag, psf_fwhm, pxscale,
                      ap_frac=1.5, psf_err=0):
    
    ##### Generate the galaxy image
    # Generate galaxy model. r_pet is in pixels
    image_perfect, image_noisy, r_pet = make_galaxy(mag, r_eff, psf_fwhm, sky_mag, n_clumps, 13, pxscale, sersic_n, q, beta)

    # Calculate background asymmetry
    bgsize = int(0.1*image_noisy.shape[0]) # 10% of the image
    sky_a, sky_norm, sky_std = _sky_properties(image_noisy, bgsize, a_type='squared')
    
    # Calculate the centre of the squared asymmetry
    ap_size = ap_frac * r_pet 
    # TODO: THIS STEP IS SLOW
    x0 = _asymmetry_center(image_noisy, ap_size, sky_a, a_type='squared')  
    
    # Get snr
    ap_source = CircularAperture(x0, ap_size)
    snr = ap_source.do_photometry(image_perfect / sky_std)[0][0] / ap_source.area

    # Deconvolve the image
    psf_fwhm = psf_fwhm + np.random.normal(loc=0, scale=psf_err) if psf_err > 0 else psf_fwhm
    psf_sigma = psf_fwhm  * gaussian_fwhm_to_sigma / pxscale
    psf = Gaussian2DKernel(psf_sigma, x_size=image_noisy.shape[1], y_size=image_noisy.shape[0])
    img_deconv = fourier_deconvolve(image_noisy, psf, sky_std)

    ###### Calculate asymmetries
    # TODO: THIS STEP IS VERY SLOW
    a_cas_real = _asymmetry_func(x0, image_perfect, ap_size, 'cas', 'annulus', bg_corr='residual')
    a_sq_real = _asymmetry_func(x0, image_perfect, ap_size, 'squared', 'annulus', bg_corr='residual')
    a_cas = _asymmetry_func(x0, image_noisy, ap_size, 'cas', 'annulus', bg_corr='residual')
    a_cas_corr = _asymmetry_func(x0, image_noisy, ap_size, 'cas_corr', 'annulus', bg_corr='residual')
    a_sq = _asymmetry_func(x0, image_noisy, ap_size, 'squared', 'annulus', bg_corr='full')
    a_fourier = _asymmetry_func(x0, img_deconv, ap_size, 'squared', 'annulus', bg_corr='full')

    ##### Store output
    output = {'a_cas_real' : a_cas_real, 'a_sq_real' : a_sq_real, 'a_cas' : a_cas, 'a_cas_cor' : a_cas_corr, 'a_sq' : a_sq, 'a_fourier' : a_fourier,
               'mag' : mag, 'psf_fwhm' : psf_fwhm, 'pxscale' : pxscale, 'snr' : snr, 'sky_mag' : sky_mag,
              'r_eff' : r_eff, 'r_pet' : r_pet, 'sersic_n' : sersic_n, 'q' : q, 'beta' : beta, 'n_clumps' : n_clumps}

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
        'mag' : (11, 18),
        'sky_mag' : (20, 26),
        'n_clumps' : (5, 50),
        'psf_fwhm' : (0.2, 3),
        'sersic_n' : (1, 6),
        'pxscale' : (0.1, 0.5)
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
    betas = stats.uniform.rvs(loc=0, scale=2*np.pi, size=N)
    pxscales = stats.uniform.rvs(loc=0.1, scale=0.5, size=N)
    for i in range(N):
        pxscales[i] = min(pxscales[i], psfs[i]/2)

    # Fix radii
    rs[rs <= 1] = 1
    rs[rs >= 20] = 20

    # Set q to above 0.5 where sersic index is high
    q_ids = np.where((qs < 0.5) & (ns >= 4))
    qs_new = stats.uniform.rvs(loc=0.5, scale=0.5, size=len(q_ids))
    qs[q_ids] = qs_new

    ### Run the execution in parallel
    Parallel(n_jobs=num_cores)(delayed(single_galaxy_run)(
        filepath=f'{args.path}/{i}.pkl', mag=mags[i], r_eff=rs[i], sersic_n=ns[i],
        q=qs[i], beta=betas[i], n_clumps=n_clumps[i], sky_mag=sky_mags[i], psf_fwhm=psfs[i],
        pxscale=pxscales[i], ap_frac=1.5, psf_err=0
    ) for i in tqdm(range(N), total=N) )


