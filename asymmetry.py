import numpy as np
import photutils as phot
import warnings
from scipy import optimize as opt
from skimage import transform as T
from skimage import measure
from scipy import fft


def _sky_properties(img, bg_size, a_type='cas'):
    """Calculates the sky asymmetry and flux.
    IN PROGRESS: right now, just draws a sky box in the bottom-left corner. 
    The "rotated" background is simply reflected, works for a Gaussian case.
    TODO: estimate bg asymmetry properly.

    Args:
        img (np.array): an NxN image array
        bg_size (int): size of the square skybox
        a_type (str): formula to use, 'cas' or 'squared'

    Returns:
        sky_a (int): asymmetry of the background per pixel
        sky_norm (int): average background normalization per pixel (<|sky|> or <sky^2>)
    """

    assert a_type in ['cas', 'squared'], 'a_type should be "cas" or "squared"'

    # Get the skybox and rotate it
    sky = img[:bg_size, :bg_size]
    sky_rotated = sky[::-1]
    sky_size = sky.shape[0]*sky.shape[1]

    # Calculate asymmetry in the skybox
    if a_type == 'cas':
        sky_a = np.sum(np.abs(sky - sky_rotated))
        sky_norm = np.mean(np.abs(sky))

    elif a_type == 'squared':
        sky_a = 10*np.sum((sky-sky_rotated)**2)
        sky_norm = np.mean(sky**2)

    # Calculate per pixel
    sky_a /= sky_size

    return sky_a, sky_norm


def _asymmetry_func(center, img, ap_size,
        a_type='cas', sky_type='skybox', sky_a=None, sky_norm=None, 
        sky_annulus=(1.5, 2), bg_corr='full',
        e=0, theta=0
    ):
    """Calculate asymmetry of the image rotated 180 degrees about a given
    center. This function is minimized in get_asymmetry to find the A center.
    
    Args:
        center (np.array): [x0, y0] coordinates of the asymmetry center.
        img (np.array): an NxN image array.
        ap_size (float): aperture size in pixels.
        a_type (str): formula to use, 'cas' or 'squared'.
        bg_corr (str): 
            The way to correct for background between 'none', 'residual', 'full'.
            If 'none', backgorund A is not subtracted. If 'residual', background 
            A is subtracted from the residual term but not the total flux. 
            If 'full', background contribution to the residual AND the total
            flux is subtracted.
        sky_type (str): 'skybox' or 'annulus'.
            If 'skybox', sky A is calculated in a random skybox in the image. 
            If 'annulus', global sky A is calculated in an annulus around the 
            source. Sky is rotated with the image. 
        sky_a (float): 
            For sky_type=='skybox'. 
            Background A calculated in _sky_properties. 
        sky_norm (float): 
            For sky_type=='skybox'. 
            The contribution of the sky to the normalization, calculated in _sky_properties.
        sky_annulus (float, float):
            For sky_type == 'annulus'.
            The sky A is calculated within a*ap_size and b*ap_size, where (a, b) are given here.
        e (float): ellipticity for an elliptical aperture (Default: 0 , circular).
        theta (float): rotation angle for elliptical apertures (Default: 0).

    Returns:
        a (float): asymmetry value
    """

    # Input checks
    assert a_type in ['cas', 'squared'], 'a_type should be "cas" or "squared"'
    assert bg_corr in ['none', 'residual', 'full'], 'bg_corr should be "none", "residual", or "full".'
    assert sky_type in ['skybox', 'annulus'], 'sky_type should be "skybox" or "annulus".'

    # Calculate dx
    dx = 1/(img.shape[0]*img.shape[1])

    # Rotate the image about asymmetry center
    img_rotated = T.rotate(img, 180, center=center, order=0)

    # Define the aperture
    ap = phot.EllipticalAperture(
        center, a=ap_size, b=ap_size*(1-e), theta=theta)
    ap_area = ap.do_photometry(np.ones_like(img))[0][0]

    # Calculate asymmetry of the image
    if a_type == 'cas':
        total_flux = ap.do_photometry(np.abs(img))[0][0]
        residual = ap.do_photometry(np.abs(img-img_rotated))[0][0]
    elif a_type == 'squared':
        total_flux = ap.do_photometry(img**2)[0][0]
        residual = 10*ap.do_photometry((img-img_rotated)**2)[0][0]


    # Calculate sky asymmetry if sky_type is "annulus"
    if sky_type == 'annulus':
        ap_sky = phot.EllipticalAnnulus(
            center, a_in=ap_size*sky_annulus[0], a_out=ap_size*sky_annulus[1],
            b_out=ap_size*sky_annulus[1]*(1-e), theta=theta
        )
        sky_area = ap_sky.do_photometry(np.ones_like(img))[0][0]
        if a_type =='cas':
            sky_a = ap_sky.do_photometry(np.abs(img-img_rotated))[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(np.abs(img))[0][0] / sky_area
        elif a_type == 'squared':
            sky_a = 10*ap_sky.do_photometry((img-img_rotated)**2)[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(img**2)[0][0] / sky_area
    
    # Correct for the background
    if bg_corr == 'none':
        a = residual / total_flux
    elif bg_corr == 'residual':
        # print(residual, ap_area*sky_a, total_flux, ap_area*sky_norm)
        a = (residual - ap_area*sky_a) / total_flux
    elif bg_corr == 'full':
        a = (residual - ap_area*sky_a) / (total_flux - ap_area*sky_norm)

    return a



def _asymmetry_fourier(center, img, sky_a=None, sky_norm=None):
    
    # Rotate the image about asymmetry center
    img_rotated = T.rotate(img, 180, center=center, order=0)

    # Calculate asymmetry of the image
        total_flux = ap.do_photometry(np.abs(img))[0][0]
        residual = ap.do_photometry(np.abs(img-img_rotated))[0][0]


    # Calculate sky asymmetry if sky_type is "annulus"
    if sky_type == 'annulus':
        ap_sky = phot.EllipticalAnnulus(
            center, a_in=ap_size*sky_annulus[0], a_out=ap_size*sky_annulus[1],
            b_out=ap_size*sky_annulus[1]*(1-e), theta=theta
        )
        sky_area = ap_sky.do_photometry(np.ones_like(img))[0][0]
        if a_type =='cas':
            sky_a = ap_sky.do_photometry(np.abs(img-img_rotated))[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(np.abs(img))[0][0] / sky_area
        elif a_type == 'squared':
            sky_a = 10*ap_sky.do_photometry((img-img_rotated)**2)[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(img**2)[0][0] / sky_area
    
    # Correct for the background
    if bg_corr == 'none':
        a = residual / total_flux
    elif bg_corr == 'residual':
        # print(residual, ap_area*sky_a, total_flux, ap_area*sky_norm)
        a = (residual - ap_area*sky_a) / total_flux
    elif bg_corr == 'full':
        a = (residual - ap_area*sky_a) / (total_flux - ap_area*sky_norm)

    return a



def get_asymmetry(
        img, ap_size, a_type='cas', 
        sky_type='skybox', bg_size=50, sky_annulus=(1.5,2), bg_corr='residual', 
        e=0, theta=0, 
        optimizer='Nelder-Mead', xtol=0.5, atol=0.1
    ):
    """Finds asymmetry of an image by optimizing the rotation center
    that minimizes the asymmetry calculated in _asymmetry_func. 
    Uses Nelder-Mead optimization from SciPy, same as statmorph.
    
    Args:
        img (np.array): an NxN image array.
        ap_size (float): aperture size in pixels.
        a_type (str): formula to use, 'cas' or 'squared'.
        sky_type (str): 'skybox' or 'annulus'.
            If 'skybox', sky A is calculated in a random skybox in the image. 
            If 'annulus', global sky A is calculated in an annulus around the 
            source. Sky is rotated with the image. 
        bg_size (int): For sky_type == 'skybox'. size of the square skybox
        sky_annulus (float, float):
            For sky_type == 'annulus'.
            The sky A is calculated within a*ap_size and b*ap_size, where (a, b) are given here.
        bg_corr (str): 
            The way to correct for background between 'none', 'residual', 'full'.
            If 'none', backgorund A is not subtracted. If 'residual', background 
            A is subtracted from the residual term but not the total flux. 
            If 'full', background contribution to the residual AND the total
            flux is subtracted.
        e (float): ellipticity for an elliptical aperture (Default: 0 , circular).
        theta (float): rotation angle for elliptical apertures (Default: 0).
        xtol (float): desired tolerancein x when minimizing A. 
            Since we don't interpolate when rotating, setting smaller values
            than 0.5 (half-pixel precision) doesn't make too much sense. 
            SM value is 1e-6.
        atoal (float): desired tolerance in asymmetry.

    Returns:
        a (float): asymmetry value
        center (np.array): [x, y] coordinates of the optimum asymmetry center
    """
    # TODO: add desired tolerance as an input parameter

    # Calculate the background asymmetry and normalization
    if sky_type == 'skybox':
        sky_a, sky_norm = _sky_properties(img, bg_size, a_type)
    else:
        sky_a = None
        sky_norm = None

    # Initial guess for the A center: center of flux^2. 
    # Note: usually center of flux is used instead. It is often a local maximum, so the optimizer
    # can move towards a local minimum instead of the correct one. Center of flux^2 puts
    # more weight on the central object and avoids placing the first guess on a local max.
    M = measure.moments(img**2, order=2)
    x0 = (M[0, 1] / M[0, 0], M[1, 0] / M[0, 0])
    # res = opt.minimize(
        
    #     _asymmetry_func, x0=x0, method=optimizer,
    #     options={
    #         'xatol': xtol, 'fatol' : atol
    #     },
    #     args=(img, ap_size, a_type, sky_type, sky_a, sky_norm, sky_annulus, bg_corr, e, theta))

    # a = res.fun
    # center = res.x

    x0 = np.array([img.shape[1]/2, img.shape[0]/2], dtype=int)
    a = _asymmetry_func(x0, img, ap_size, a_type, sky_type, sky_a, sky_norm, sky_annulus, bg_corr, e, theta)
    center = x0

    return a, center

def get_residual(image, center, a_type):
    """Utility function that rotates the image about the center and gets the residual
    according to an asymmetry definition given by a_type."""

    assert a_type in ['cas', 'squared'], 'a_type should be "cas" or "squared"'
    img_rotated = T.rotate(image, 180, center=center)
    residual = image - img_rotated

    if a_type == 'cas':
        return np.abs(residual)
    elif a_type == 'squared':
        return residual**2


