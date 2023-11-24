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
from galaxy_generator import simulate_perfect_galaxy, add_source_to_image, sky_noise, get_galaxy_rng_vals, get_augmentation_rng_vals
from asymmetry import fourier_deconvolve, get_asymmetry, fourier_rescale
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel



num_cores = multiprocessing.cpu_count()

def get_a_values(img, rpet, err, psf_fwhm, pxscale, perfect_pxscale):
    # Rpet is in pixels
    
    ap_size = 1.5*rpet
    
    # Non-fourier asymmetries
    a_sq = get_asymmetry(img, ap_size, a_type='squared', sky_type='annulus', bg_corr='full', sky_annulus=[2, 3])[0]
    a_cas = get_asymmetry(img, ap_size, a_type='cas', sky_type='annulus', bg_corr='residual', sky_annulus=[2, 3])[0]
    a_cas_corr = get_asymmetry(img, ap_size, a_type='cas_corr', sky_type='annulus', bg_corr='residual', sky_annulus=[2,3])[0]

    # Fourier asymmetry: rescale the image then deconvolve
    if psf_fwhm > 0:
        img_rescaled = fourier_rescale(img, pxscale, perfect_pxscale)
        err_rescaled = fourier_rescale(err, pxscale, perfect_pxscale)
        psf = Gaussian2DKernel(psf_fwhm*gaussian_fwhm_to_sigma/perfect_pxscale, x_size=img.shape[0])
        img_deconv = fourier_deconvolve(img_rescaled, psf, err_rescaled, convolve_nyquist=True)
        a_fourier = get_asymmetry(
            img_deconv, ap_size*pxscale/perfect_pxscale, a_type='squared', sky_type='annulus', bg_corr='full', sky_annulus=[2,3]
        )[0]
    else:
        a_fourier = a_sq
        
    # Return all measurements
    res = {'a_cas' : a_cas, 'a_cas_corr' : a_cas_corr, 'a_sq' : np.sqrt(a_sq), 'a_fourier' : np.sqrt(a_fourier)}
    return res


def single_galaxy_run(filepath, gal_params, img_params, ap_frac=1.5, perfect_pxscale=0.2):

    ##### Generate the perfect galaxy image at desired pixelscale
    # Generate galaxy model. r_pet is in ARCSEC.
    # Convolve with a PSF to make the "perfect" image nyquist-sampled
    image_perfect, galaxy_dict, r_pet = simulate_perfect_galaxy(pxscale=perfect_pxscale,  **gal_params)
    image_perfect = add_source_to_image(**galaxy_dict, psf_fwhm=2*perfect_pxscale, pxscale=perfect_pxscale, psf_method='astropy')
    
    # Generate the perfect galaxy at new pixelscale
    image_lowres, galaxy_dict, r_pet = simulate_perfect_galaxy(**img_params, **gal_params)
    
    # Create observed image
    image_psf = add_source_to_image(**galaxy_dict, **img_params, psf_method='astropy')
    image_noisy, sky_flux = sky_noise(np.abs(image_psf), **img_params, rms_noise=True)
    pxscale = img_params['pxscale']
    
    # Calculate noise level
    err = np.sqrt(image_psf + sky_flux**2)
    
    # Calculate average SNR in the aperture
    snr = image_lowres / err
    xc, yc = image_lowres.shape[1]/2, image_lowres.shape[0]/2
    ap = CircularAperture((xc,yc), ap_frac*r_pet/pxscale)
    avg_snr = ap.do_photometry(snr)[0][0]/ap.area

    # Calculate the real asymmetry
    a_cas_real = get_asymmetry(
        image_perfect, ap_frac*r_pet/perfect_pxscale, a_type='cas', sky_type='skybox', bg_corr='none', sky_annulus=[2, 3]
    )[0]
    a_sq_real = get_asymmetry(
        image_perfect, ap_frac*r_pet/perfect_pxscale, a_type='squared', sky_type='skybox', bg_corr='none', sky_annulus=[2, 3]
)[0]
    a_sq_real = np.sqrt(a_sq_real)
    
    # Calculate asyms from the noisy image
    output = get_a_values(image_noisy, r_pet/pxscale, err, img_params['psf_fwhm'], pxscale, perfect_pxscale)

    ##### Store output
    output['a_cas_real'] = a_cas_real
    output['a_sq_real'] = a_sq_real
    output['snr'] = avg_snr
    output = {**output, **gal_params, **img_params}
                         

    with open(filepath, 'wb') as f:
        pickle.dump(output, f)




if __name__ == '__main__':

    ###### Parallelize over different galaxies. For each, do a PSF and SNR series.

    # Parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("N", help="number of galaxies to generate")
    parser.add_argument("path", help="folder to store images and asymmetries in")
    args = parser.parse_args()

    # Perfect resolution pxscale
    perfect_pxscale = 0.1
    
    # Generate random params
    N = int(args.N)
    gal_params = get_galaxy_rng_vals(N)
    img_params = get_augmentation_rng_vals(N)
    
    # Fix the parameters other than the one I want to vary
    for p in img_params:
        p['sky_mag'] = 30
        p['psf_fwhm'] = 3*p['pxscale']

    ### Run the execution in parallel
    Parallel(n_jobs=num_cores)(delayed(single_galaxy_run)(
           filepath=f'{args.path}/{i}.pkl', gal_params=gal_params[i], img_params=img_params[i], ap_frac=1.5, perfect_pxscale=perfect_pxscale
    ) for i in tqdm(range(N), total=N) )


