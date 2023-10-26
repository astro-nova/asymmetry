import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
import multiprocessing
import statmorph as sm
from scipy import ndimage

from astropy import units as u
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources

from photutils.aperture import EllipticalAperture, CircularAnnulus, CircularAperture
import galsim

# from asymmetry import get_asymmetry
sys.path.append('../')
from galaxy_generator import add_source_to_image, simulate_perfect_galaxy, sky_noise, petrosian_sersic
from asymmetry import get_asymmetry, get_residual
from skimage import transform as T
from importlib import resources as impresources
import templates

num_cores = multiprocessing.cpu_count()-3

plt.rcParams['font.size'] = 9
plt.rcParams['axes.xmargin'] = .05  # x margin.  See `axes.Axes.margins`
plt.rcParams['axes.ymargin'] = .05  # y margin.  See `axes.Axes.margins`

#################### SDSS set-up ###########################
sdss_ra = 150
sdss_dec = 2.3
filt = 'r'
bandpass_file = "passband_sdss_" + filt
inp_file = (impresources.files(templates) / bandpass_file)
throughput = galsim.LookupTable.from_file(inp_file)
bandpass = galsim.Bandpass(throughput, wave_type = u.angstrom)
## gain, exptime and diameter of telescope
telescope_params = {'g':4.8, 't_exp':53.91, 'D':2.5}
## effective wavelength and width of filter
transmission_params = {'eff_wav':616.5, 'del_wav':137}
g, t_exp, D = telescope_params['g'],telescope_params['t_exp'],telescope_params['D']
transmission = bandpass(transmission_params['eff_wav'])
#############################################################


def get_perfect_galaxy(mag, r_eff, fov_reff, pxscale, sersic_n=1, q=1, beta=0):
    
    sdss_ra = 150
    sdss_dec = 2.3
    
    # Calculate field of view in degrees
    fov = fov_reff * r_eff / 3600
    
    # generate blank image with fov and wcs info
    field_image, wcs = gen_image(sdss_ra, sdss_dec, pxscale, fov, fov)

    # create a galaxy with given params
    galaxy = gen_galaxy(mag=mag, re=r_eff, n=sersic_n, q=q, beta=beta)
    
    # get petrosian radius of galaxy in px
    r_pet = petrosian_sersic(fov, r_eff, 1)/pxscale

    return field_image, galaxy, r_pet



def get_morphology_dict(source_morph, pxscale, prefix=""):
    """Saves the morphology parameters calculated by statmorph.
    API reference: https://statmorph.readthedocs.io/en/latest/api.html

    Args:
        source_morph (statmorph.SourceMorphology): morphology object from statmorph
        pxscale_arcsec (float): pixel scale [arcsec/px]
        pxscale_kpc (float): pixel scale [kpc/px]

    Returns:
        dict: dictionary object with morphology measurements
    """

    output = {
        f"{prefix}_x"                     : source_morph.xc_centroid,
        f"{prefix}_y"                     : source_morph.yc_centroid,
        f"{prefix}_asymmetry"             : source_morph.asymmetry,
        f"{prefix}_concentration"         : source_morph.concentration,           # Part of CAS (Conselice04?)
        f"{prefix}_deviation"             : source_morph.deviation,               # Part of MID (Freeman13, Peth16 )
        f"{prefix}_ellipticity_asymmetry" : source_morph.ellipticity_asymmetry,   # Ellip. rel to min. asym. point
        f"{prefix}_ellipticity_centroid"  : source_morph.ellipticity_centroid,    # Ellip. rel to centroid
        f"{prefix}_elongation_asymmetry"  : source_morph.elongation_asymmetry,    # Elong. rel to min. asym. point
        f"{prefix}_elongation_centroid"   : source_morph.elongation_centroid,     # Elong. rel to centorid
        f"{prefix}_flag"                  : source_morph.flag,                    # 1 if failed to estimate
        f"{prefix}_flag_sersic"           : source_morph.flag_sersic,             # 1 if sersic fit failed
        f"{prefix}_flux_circ"             : source_morph.flux_circ,               # Flux in 2xPetrosian radius
        f"{prefix}_flux_ellip"            : source_morph.flux_ellip,              # Flux in 2xPetrosian ellip. radius
        f"{prefix}_gini"                  : source_morph.gini,                    # Lotz04
        f"{prefix}_gini_m20_bulge"        : source_morph.gini_m20_bulge,          # Rodriguez-Gomez19
        f"{prefix}_gini_m20_merger"       : source_morph.gini_m20_merger,         # Rodriguez-Gomez19
        f"{prefix}_intensity"             : source_morph.intensity,               # Part of MID (Freeman13, Peth16)
        f"{prefix}_m20"                   : source_morph.m20,                     # Lotz04
        f"{prefix}_multimode"             : source_morph.multimode,               # Part of MID (Freeman13, Peth16)
        f"{prefix}_orientation_asymmetry" : source_morph.orientation_asymmetry,   # Orientation rel. to min asym. point
        f"{prefix}_orientation_centroid"  : source_morph.orientation_centroid,    # Orientation rel. to centroid
        f"{prefix}_outer_asymmetry"       : source_morph.outer_asymmetry,         # Wen14
        f"{prefix}_r20"                   : source_morph.r20          * pxscale,  # 20% light within 1.5Rpetro
        f"{prefix}_r50"                   : source_morph.r50          * pxscale,  # 50% light within 1.5Rpetro
        f"{prefix}_r80"                   : source_morph.r80          * pxscale,  # 80% light within 1.5Rpetro
        f"{prefix}_rhalf_circ"            : source_morph.rhalf_circ   * pxscale,  # 50% light; circ ap; min asym; total at rmax
        f"{prefix}_rhalf_ellip"           : source_morph.rhalf_ellip  * pxscale,  # 50% light; ell. ap; min asym; total at rmax
        f"{prefix}_rmax_circ"             : source_morph.rmax_circ    * pxscale,  # From min asym to edge, Pawlik16
        f"{prefix}_rmax_ellip"            : source_morph.rmax_ellip   * pxscale,  # Semimajor ax. from min asym to edge
        f"{prefix}_rpetro_circ"           : source_morph.rpetro_circ  * pxscale,  # Petrosian; wrt min asym point
        f"{prefix}_rpetro_ellip"          : source_morph.rpetro_ellip * pxscale,  # Petrosian ellip; wrt min asym point
        f"{prefix}_sersic_amplitude"      : source_morph.sersic_amplitude,        # Amplitude of sersic fit at rhalf
        f"{prefix}_sersic_ellip"          : source_morph.sersic_ellip,            # Ellipticity of sersic fit
        f"{prefix}_sersic_n"              : source_morph.sersic_n,                # Sersic index
        f"{prefix}_sersic_rhalf"          : source_morph.sersic_rhalf * pxscale,  # Sersic 1/2light radius
        f"{prefix}_sersic_theta"          : source_morph.sersic_theta,            # Orientation of sersic fit
        f"{prefix}_shape_asymmetry"       : source_morph.shape_asymmetry,         # Pawlik16
        f"{prefix}_sky_mean"              : source_morph.sky_mean,
        f"{prefix}_sky_median"            : source_morph.sky_median,
        f"{prefix}_sky_sigma"             : source_morph.sky_sigma,
        f"{prefix}_smoothness"            : source_morph.smoothness,            # Part of CAS (Conselice04?)
        f"{prefix}_sn_per_pixel"          : source_morph.sn_per_pixel,
        f"{prefix}_xc_asymmetry"          : source_morph.xc_asymmetry,          # Asym. center (x)
        f"{prefix}_yc_asymmetry"          : source_morph.yc_asymmetry           # Asym. center (y)
    }

    return output

def single_galaxy_run(filepath, mag, r_eff, sersic_n, q, n_clumps, sky_mag, psf_fwhm, pxscale):

#     print('here')
    ##### Generate the galaxy image
    # generate all the clumps and their positions
    # Generate clumps
    perfect_px = 0.168
    rs = np.random.random(n_clumps)*(0.8-0.05)+0.05
    fs = np.random.random(n_clumps)*(0-0.05) + 0.05

    smin = 0.2
    smax = 3
    sigs = np.random.random(n_clumps)*(smax-smin) + smin
    thetas = np.random.random(n_clumps)*360

    fs = np.power(10, fs)

    clump_properties = {
        'r' : rs,
        'theta' : thetas,
        'flux' : fs,
        'sigma' : sigs
    }

    # Generate galaxy model at two resolutions
    try:
        image_perfect, params_res, r_pet_res = simulate_perfect_galaxy(mag, r_eff, perfect_px, 15, sersic_n, q, 0, n_clumps, clump_properties)
    except:
        return
    img, params, r_pet = simulate_perfect_galaxy(mag, r_eff, pxscale, 15, sersic_n, q, 0, n_clumps, clump_properties)
#     print('made a galaxy')
    
    # Convolve with PSF
    image_perfect = add_source_to_image(**params_res, psf_fwhm=0, pxscale=perfect_px, psf_method="astropy")
    image_psf = add_source_to_image(**params, psf_fwhm=psf_fwhm, pxscale=pxscale, psf_method="astropy")
    image_noisy, sky_flux = sky_noise(image_psf, sky_mag=sky_mag, pixel_scale=pxscale, rms_noise=True)


    return image_perfect, image_noisy 

    # Create segmap at 2 pet
    xc = image_perfect.shape[0]/2
    segmap_res = CircularAperture((xc,xc), 1.5*r_pet_res).to_mask().to_image(image_perfect.shape)
    segmap_res = (segmap_res > 0.1).astype(int)
    segmap = detect_sources(image_noisy, sky_flux, 300)
    try:
        segmap = (ndimage.gaussian_filter(segmap.data.astype(float), 3) > 0.1).astype(int)
    except:
        return 
    
    # Create psf
    psf_std = psf_fwhm * gaussian_fwhm_to_sigma
    psf = Gaussian2DKernel(x_stddev=psf_std/pxscale).array

    ###### Run statmorph 
    morphs_real = sm.source_morphology(image_perfect, segmap=segmap_res, gain=1)[0]
    try:
        morphs_noisy = sm.source_morphology(image_noisy, segmap=segmap, gain=1, psf=psf)[0]
    except:
        return
    out_real = get_morphology_dict(morphs_real, perfect_px, prefix='_real')
    out_noisy = get_morphology_dict(morphs_noisy, pxscale, prefix='_noisy')

    ##### Calculate SNR
    ap_real = EllipticalAperture((xc,xc), 1.5*r_pet_res, 1.5*r_pet_res)
    ap_sky = CircularAnnulus((xc,xc), 2*1.5*r_pet_res, 2*2*r_pet_res)
    var = sky_flux**2 * (perfect_px/pxscale)**2 + image_perfect
    snr = image_perfect / np.sqrt(var)
    snr_px = ap_real.do_photometry(snr)[0][0] / ap_real.do_photometry(np.ones_like(snr))[0][0]

    ##### Store output
    # imgs = [image_perfect, image_noisy]
    inputs = dict(
        mag=mag, r_eff=r_eff, r_pet=r_pet, sersic_n=sersic_n, q=q, n_clumps=n_clumps, 
        sky_mag=sky_mag, psf=psf_fwhm, snr=snr_px, pxscale=pxscale
    )
    output = {
        'input' : inputs,
        # 'images' : imgs,
#         'apertures' : aps,
    }
    output.update(out_real)
    output.update(out_noisy)

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
        'mag' : (12, 16),
        'sky_mag' : (21, 27),
        'n_clumps' : (5, 60),
        'psf_fwhm' : (0, 3),
        'sersic_n' : (1, 4),
        'px' : (0.168, 0.5),
    }

    # Generate parameters for n galaxies
    N = int(args.N)
    mags = stats.uniform.rvs(loc=lims['mag'][0], scale=lims['mag'][1] - lims['mag'][0], size=N)
    ns = stats.uniform.rvs(loc=lims['sersic_n'][0], scale=lims['sersic_n'][1] - lims['sersic_n'][0], size=N)
    sky_mags = stats.uniform.rvs(loc=lims['sky_mag'][0], scale=lims['sky_mag'][1] - lims['sky_mag'][0], size=N)
    n_clumps = np.random.randint(low=lims['n_clumps'][0], high=lims['n_clumps'][1], size=N)
    psfs = stats.uniform.rvs(loc=lims['psf_fwhm'][0], scale=lims['psf_fwhm'][1] - lims['psf_fwhm'][0], size=N)
#     pxscales = stats.uniform.rvs(loc=lims['px'][0], scale=lims['px'][1] - lims['px'][0], size=N)
#     for i in range(N):
#         pxscales[i] = np.min([pxscales[i], psfs[i]/2])
    qs = stats.uniform.rvs(loc=0.2, scale=0.8, size=N)
    qs[ns > 3.5] = 1.0
    rs = -1.9*mags + 35 + stats.norm.rvs(loc=0, scale=1.5, size=N)
    rs[rs <= 1] = 1
    rs[rs >= 20] = 20

    ### Run the execution in parallel
    Parallel(n_jobs=num_cores)(delayed(single_galaxy_run)(
        filepath=f'{args.path}/{i}.pkl', mag=mags[i], r_eff=rs[i], sersic_n=ns[i],
        q=qs[i], n_clumps=n_clumps[i], sky_mag=sky_mags[i], psf_fwhm=psfs[i], pxscale=psfs[i]/2
    ) for i in tqdm(range(N), total=N) )

