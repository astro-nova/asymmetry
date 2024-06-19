# Pick your cosmology here
from astropy.cosmology import Planck18 as cosmo
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import Gaussian2DKernel, convolve 
from skimage import transform as T


def rescale_image(image, redshift, pxscale_native, physical_scale=0.5, discrete=False):
    """Uses scikit-image transform to resample a galaxy to a coarser pixel scale with a 
    wider PSF. Preserves total flux; remove the *factor**2 term to instead preserve a 
    surface density. Can also deal with integer arrays (masks, segmaps, ...).

    Args:
        image (np.ndarray): your image array, can be float, integer or boolean
        redshift (float): source redshift
        pxscale_native (float): pixel scale of the JWST image in arcsec/px
        physical_scale (float): physical scale you want per in kpc/px
        discrete (bool): whether the image is discrete (e.g., mask or segmap)

    Returns:
        img_rescaled (np.ndarray): image resampled onto a coarser grid
    """

    # For a given redshift, calculate the pixel scale you need
    pxscale_wanted = cosmo.arcsec_per_kpc_proper(redshift).value * physical_scale

    # How much we want to resample the image by
    factor = pxscale_wanted / pxscale_native

    # This assumes the PSF FWHM of your images is roughly 2 pixels 
    # Technically, you should convolve the image with a PSF such that
    # Final image = PSF[Nyquist] * Source = PSF[conv] * PSF[native] * Source
    # where * is the deconvolution operator. 
    # Doing it the way I'm doing here is an over simplification as it ignores
    # the native PSF of the source.
    # But calculating PSF[conv] properly requires doing this in Fourier space and
    # noise becaomes a problem; this should be good enough for your purposes, but also
    # let me know if you want to try proper convolution/deconvolution
    # In any case, the pixel scale has a bigger impact on CAS asymmetry than the PSF
    psf_fwhm = 2
    psf_size = psf_fwhm*gaussian_fwhm_to_sigma*factor
    psf_conv = Gaussian2DKernel(psf_size) 

    # Convolve the image (still on native px scale) with the new PSF
    img_conv = convolve(image, psf_conv)

    # Rescale the image
    if discrete:
        img_rescaled = T.rescale(img_conv, 1/factor)
    else:
        img_rescaled = T.rescale(img_conv, 1/factor, order=0, preserve_range=True)

    # T.rescale keeps constant the flux density so we need to correct to keep
    # the total flux invariant instead
    if not discrete:
        img_rescaled *= factor**2

    return img_rescaled