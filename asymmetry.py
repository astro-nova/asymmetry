import numpy as np
import photutils as phot
import warnings
from scipy import optimize as opt
from skimage import transform as T
from skimage import measure
from scipy import fft
from scipy.interpolate import griddata
from astropy.stats import sigma_clipped_stats
from astropy.convolution import Gaussian2DKernel
from scipy.signal import savgol_filter

# def _sky_properties(img, bg_size, a_type='cas'):
#     """Calculates the sky asymmetry and flux.
#     IN PROGRESS: right now, just draws a sky box in the bottom-left corner. 
#     The "rotated" background is simply reflected, works for a Gaussian case.
#     TODO: estimate bg asymmetry properly.

#     Args:
#         img (np.array): an NxN image array
#         bg_size (int): size of the square skybox
#         a_type (str): formula to use, 'cas' or 'squared'

#     Returns:
#         sky_a (int): asymmetry of the background per pixel
#         sky_norm (int): average background normalization per pixel (<|sky|> or <sky^2>)
#     """

#     assert a_type in ['cas', 'squared', 'cas_corr'], 'a_type should be "cas" or "squared"'

#     # Get the skybox and rotate it
#     sky = img[:bg_size, :bg_size]
#     sky_rotated = sky[::-1]
#     sky_size = sky.shape[0]*sky.shape[1]

#     # Calculate asymmetry in the skybox
#     if a_type == 'cas':
#         sky_a = np.sum(np.abs(sky - sky_rotated))
#         sky_norm = np.mean(np.abs(sky))

#     elif a_type == 'squared':
#         sky_a = 10*np.sum((sky-sky_rotated)**2)
#         sky_norm = np.mean(sky**2)

#     elif a_type == 'cas_corr':
#         sky_a = np.sum(np.abs(sky - sky_rotated))
#         sky_norm = np.mean(sky)
#     sky_std = np.std(sky)

#     # Calculate per pixel
#     sky_a /= sky_size

#     return sky_a, sky_norm, sky_std

def _sky_properties(img, mask, a_type='cas'):
    
    bgmean, bgmed, bgsd = sigma_clipped_stats(img, mask=mask)
    if a_type == 'cas':
        sky_a = 1.6*bgsd
        sky_norm = 0.8*bgsd 
    elif a_type == 'cas_corr':
        sky_a = 1.6*bgsd
        sky_norm = 0
    elif a_type == 'squared':
        sky_a = 20*bgsd**2
        sky_norm = bgsd**2
    return sky_a, sky_norm, bgsd





def _asymmetry_center(img, ap_size, sky_a, 
                        a_type='cas_corr', e=0, theta=0,
                        optimizer='Nelder-Mead', xtol=0.5, atol=0.1):
    """Find the rotation center that minimizes asymmetry. 
    To speed up, only use the skybox background, not the annulus.
    Use the adaptive CAS asymmetry as it's the fastest.
    """

    # Initial guess for the A center: center of flux^2. 
    # Note: usually center of flux is used instead. It is often a local maximum, so the optimizer
    # can move towards a local minimum instead of the correct one. Center of flux^2 puts
    # more weight on the central object and avoids placing the first guess on a local max.
    M = measure.moments(img**2, order=2)
    x0 = (M[0, 1] / M[0, 0], M[1, 0] / M[0, 0])

    # Find the minimum of corrected |A|
    res = opt.minimize(
        
        _asymmetry_func, x0=x0, method=optimizer,
        options={
            'xatol': xtol, 'fatol' : atol
        },
        args=(img, ap_size, a_type, 'skybox', sky_a, 0, None, 'residual', e, theta))
    x0 = res.x
    return x0


def _asymmetry_func(center, img, ap_size, mask=None,
        a_type='cas', sky_type='skybox', sky_a=None, sky_norm=None, 
        sky_annulus=(1.5, 3), bg_corr='full',
        e=0, theta=0
    ):
    """Calculate asymmetry of the image rotated 180 degrees about a given
    center. This function is minimized in get_asymmetry to find the A center.

    TODO: get rid of the factor of 10
    
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
    assert a_type in ['cas', 'squared', 'cas_corr'], 'a_type should be "cas" or "squared"'
    assert bg_corr in ['none', 'residual', 'full'], 'bg_corr should be "none", "residual", or "full".'
    assert sky_type in ['skybox', 'annulus'], 'sky_type should be "skybox" or "annulus".'

    # Rotate the image about asymmetry center
    img_rotated = T.rotate(img, 180, center=center, order=3)
    mask_rotated = T.rotate(mask, 180, center=center, order=0)
    mask = mask.astype(bool) | mask_rotated.astype(bool)

    # Define the aperture
    ap = phot.EllipticalAperture(
        center, a=ap_size, b=ap_size*(1-e), theta=theta)
    ap_area = ap.do_photometry(np.ones_like(img), mask=mask)[0][0]

    # Calculate asymmetry of the image
    if a_type == 'cas':
        total_flux = ap.do_photometry(np.abs(img), mask=mask)[0][0]
        residual = ap.do_photometry(np.abs(img-img_rotated), mask=mask)[0][0]
    elif a_type == 'squared':
        total_flux = ap.do_photometry(img**2, mask=mask)[0][0]
        residual = 10*ap.do_photometry((img-img_rotated)**2, mask=mask)[0][0]
    elif a_type == 'cas_corr':
        total_flux = ap.do_photometry(img, mask=mask)[0][0]
        residual = ap.do_photometry(np.abs(img-img_rotated), mask=mask)[0][0]


    # Calculate sky asymmetry if sky_type is "annulus"
    if sky_type == 'annulus':
        ap_sky = phot.EllipticalAnnulus(
            center, a_in=ap_size*sky_annulus[0], a_out=ap_size*sky_annulus[1],
            b_out=ap_size*sky_annulus[1]*(1-e), theta=theta
        )
        sky_area = ap_sky.do_photometry(np.ones_like(img), mask=mask)[0][0]
        if a_type =='cas':
            sky_a = ap_sky.do_photometry(np.abs(img-img_rotated), mask=mask)[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(np.abs(img), mask=mask)[0][0] / sky_area
        elif a_type == 'squared':
            sky_a = 10*ap_sky.do_photometry((img-img_rotated)**2, mask=mask)[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(img**2, mask=mask)[0][0] / sky_area
        elif a_type == 'cas_corr':
            sky_a = ap_sky.do_photometry(np.abs(img-img_rotated), mask=mask)[0][0] / sky_area
            sky_norm = ap_sky.do_photometry(img, mask=mask)[0][0] / sky_area

    
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
        img, ap_size, mask=None, a_type='cas', 
        sky_type='skybox', bg_size=50, sky_annulus=(1.5,3), bg_corr='residual', 
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
    if mask is None:
        mask = np.zeros_like(img)

    # Calculate the background asymmetry and normalization
    sky_a, sky_norm, bgsd = _sky_properties(img, mask, a_type)



    x0 = np.array([img.shape[1]/2, img.shape[0]/2], dtype=int)
    res = opt.minimize(
        
        _asymmetry_func, x0=x0, method=optimizer,
        options={
            'xatol': xtol, 'fatol' : atol
        },
        args=(img, ap_size, mask, a_type, 'skybox', sky_a, sky_norm, 0, bg_corr, e, theta))


    a = _asymmetry_func(res.x, img, ap_size, mask, a_type, sky_type, sky_a, sky_norm, sky_annulus, bg_corr, e, theta)
    center = res.x

    # x0 = np.array([img.shape[1]/2, img.shape[0]/2], dtype=int)
    # a = _asymmetry_func(x0, img, ap_size, a_type, sky_type, sky_a, sky_norm, sky_annulus, bg_corr, e, theta)
    # center = x0

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


def _fit_snr_old(img_fft, noise_fft, snr_thresh=3, quant_thresh=0.98):
    """Given an FFT of an image and a noise level, estimate SNR(omega)
    by interpolating high SNR regions and setting high-frequency SNR to 1k less than SNR max.
    """
    

    # Calculate from the image SNR
    snr = np.abs(img_fft) / noise_fft
    snr_min = np.log10(np.max(snr)) - 4  # Minimum SNR is 100000 times dimmer than the center
    
    # Only look at one quarter of the array (FFT is reflected along x and y)
    xc = int(img_fft.shape[0]/2)
    snr_corner = snr[:xc, :xc]
    
    # Image x, y arrays as placeholders
    xs = np.arange(xc+1)
    XS, YS = np.meshgrid(xs, xs)
    
    # Choose indices where SNR is high 
    snr_lim = np.quantile(snr_corner, quant_thresh)
    snr_lim = np.max([snr_lim, snr_thresh])
    good_ids = np.nonzero(snr_corner > snr_lim)
    good_log_snr = np.log10(snr_corner[good_ids])
    
    # Select regions dominated by noise and set their SNR to snr_min
    noise_ids = np.nonzero(snr_corner < 1)   
    noise_log_snr = snr_min*np.ones(len(noise_ids[0]))

    # SNR array to interpolate
    log_snr = np.concatenate((good_log_snr, noise_log_snr))
    snr_ids = np.hstack((good_ids, noise_ids))
    snr_ids = (snr_ids[0], snr_ids[1])
    xs = XS[snr_ids]
    ys = YS[snr_ids]

    # Add a low SNR at highest frequency edges to help interpolation
    boundaries = np.arange(xc+1)
    xs = np.concatenate((xs, np.ones_like(boundaries)*(xc+1), boundaries))
    ys = np.concatenate((ys, boundaries, np.ones_like(boundaries)*(xc+1)))
    log_snr = np.concatenate((log_snr, snr_min*np.ones_like(boundaries), snr_min*np.ones_like(boundaries)))

    # Interpolate
    snr_grid = griddata((xs, ys), log_snr, (XS, YS), method='linear')
    

    # Expand the grid (corner) back to the original shape by doubling in X and Y
    j = -1 
    k = -1 if (snr.shape[0] % 2 == 1) else -2
    fit_snr = np.ones_like(snr)
    fit_snr[:xc,:xc] = snr_grid[:j, :j]
    fit_snr[xc:,:xc] = snr_grid[k::-1, :j]
    fit_snr[:xc,xc:] = snr_grid[:j, k::-1]
    fit_snr[xc:,xc:] = snr_grid[k::-1, k::-1]
    
    # Undo the log
    fit_snr = np.power(10, fit_snr)

    # Rewrite the good SNR regions with real values
    good_ids = np.nonzero(snr > snr_lim)
    fit_snr[good_ids] = snr[good_ids]
    
    return fit_snr




def _fit_snr_old(img_fft, noise_fft, snr_thresh=3, quant_thresh=0.98):
    """Given an FFT of an image and a noise level, estimate SNR(omega)
    by interpolating high SNR regions and setting high-frequency SNR to 1k less than SNR max.
    """
    

    # Calculate from the image SNR
    snr = np.abs(img_fft) / noise_fft
    snr_min = np.log10(np.max(snr)) - 4  # Minimum SNR is 100000 times dimmer than the center
    
    # Only look at one quarter of the array (FFT is reflected along x and y)
    xc = int(img_fft.shape[0]/2)
    snr_corner = snr[:xc, :xc]
    
    # Image x, y arrays as placeholders
    xs = np.arange(xc+1)
    XS, YS = np.meshgrid(xs, xs)
    
    # Choose indices where SNR is high 
    snr_lim = np.quantile(snr_corner, quant_thresh)
    snr_lim = np.max([snr_lim, snr_thresh])
    good_ids = np.nonzero(snr_corner > snr_lim)
    good_log_snr = np.log10(snr_corner[good_ids])
    
    # Select regions dominated by noise and set their SNR to snr_min
    noise_ids = np.nonzero(snr_corner < 1)   
    noise_log_snr = snr_min*np.ones(len(noise_ids[0]))

    # SNR array to interpolate
    log_snr = np.concatenate((good_log_snr, noise_log_snr))
    snr_ids = np.hstack((good_ids, noise_ids))
    snr_ids = (snr_ids[0], snr_ids[1])
    xs = XS[snr_ids]
    ys = YS[snr_ids]

    # Add a low SNR at highest frequency edges to help interpolation
    boundaries = np.arange(xc+1)
    xs = np.concatenate((xs, np.ones_like(boundaries)*(xc+1), boundaries))
    ys = np.concatenate((ys, boundaries, np.ones_like(boundaries)*(xc+1)))
    log_snr = np.concatenate((log_snr, snr_min*np.ones_like(boundaries), snr_min*np.ones_like(boundaries)))

    # Interpolate
    snr_grid = griddata((xs, ys), log_snr, (XS, YS), method='linear')
    

    # Expand the grid (corner) back to the original shape by doubling in X and Y
    j = -1 
    k = -1 if (snr.shape[0] % 2 == 1) else -2
    fit_snr = np.ones_like(snr)
    fit_snr[:xc,:xc] = snr_grid[:j, :j]
    fit_snr[xc:,:xc] = snr_grid[k::-1, :j]
    fit_snr[:xc,xc:] = snr_grid[:j, k::-1]
    fit_snr[xc:,xc:] = snr_grid[k::-1, k::-1]
    
    # Undo the log
    fit_snr = np.power(10, fit_snr)

    # Rewrite the good SNR regions with real values
    good_ids = np.nonzero(snr > snr_lim)
    fit_snr[good_ids] = snr[good_ids]
    
    return fit_snr


def _fit_snr(img_fft, noise_fft, psf_fft, snr_thresh=5, quant_thresh=0.8):
    """Given an FFT of an image and a noise level, estimate SNR(omega)
    by interpolating high SNR regions and setting high-frequency SNR to 1k less than SNR max.
    """
    
    snr_fft = np.abs(img_fft) / np.abs(noise_fft)

    # Calculate from the image SNR
    snr_med = np.median(np.abs(snr_fft))
    snr_min = np.min([np.log10(np.max(snr_fft))-3, np.log10(snr_med)])  # Minimum SNR is 1000 times dimmer than the center
    
    # Only look at one quarter of the array (FFT is reflected along x and y)
    xc = int(img_fft.shape[0]/2)
    snr_corner = snr_fft[:xc, :xc]
    
    # Image x, y arrays as placeholders
    xs = np.arange(xc+1)
    XS, YS = np.meshgrid(xs, xs)
    
#     # Choose indices where SNR is high 
    snr_lim = np.quantile(snr_corner, quant_thresh)
    snr_lim = np.max([snr_lim, snr_thresh])
    good_ids = np.nonzero(snr_corner > snr_lim)
    good_log_snr = np.log10(snr_corner[good_ids])

    # SNR array to interpolate
    log_snr = good_log_snr
    snr_ids = good_ids
    snr_ids = (snr_ids[0], snr_ids[1])
    xs = XS[snr_ids]
    ys = YS[snr_ids]

    # Add a low SNR at highest frequency edges to help interpolation
    boundaries = np.arange(xc+1)
    xs = np.concatenate((xs, np.ones_like(boundaries)*(xc+1), boundaries))
    ys = np.concatenate((ys, boundaries, np.ones_like(boundaries)*(xc+1)))
    log_snr = np.concatenate((log_snr, snr_min*np.ones_like(boundaries), snr_min*np.ones_like(boundaries)))

    # Interpolate
    snr_grid = griddata((xs, ys), log_snr, (XS, YS), method='linear')
    
    # Expand the grid (corner) back to the original shape by doubling in X and Y
    j = -1 
    k = -1 if (snr_fft.shape[0] % 2 == 1) else -2
    fit_snr = np.ones_like(snr_fft)
    fit_snr[:xc,:xc] = snr_grid[:j, :j]
    fit_snr[xc:,:xc] = snr_grid[k::-1, :j]
    fit_snr[:xc,xc:] = snr_grid[:j, k::-1]
    fit_snr[xc:,xc:] = snr_grid[k::-1, k::-1]
    
    # Undo the log
    fit_snr = np.power(10, fit_snr)

    # Rewrite the good SNR regions with real values
    good_ids = np.nonzero(snr_fft > snr_lim)
    fit_snr[good_ids] = snr_fft[good_ids]
    
    # Correct the SNR by the PSF
    fit_snr = fit_snr / ( psf_fft + 1/fit_snr )
    
    return fit_snr

def fourier_deconvolve(img, psf, noise, convolve_nyquist=False):
    """Performs deconvolution of the image by dividing by SNR-weighted
    PSF in the Fourier space. Similar to Wiener transform excep the noise
    level is retained in the deconvolved image.
    
    TODO: right now, all input arrays are square, should deal with non-square images
    Args:
        img (np.ndarray): NxN image array 
        psf (np.ndarray): NxN normalized PSF
        sky_sigma (float): estimate of the sky standard deviation
        convolve_nyquist (bool): if True, convolve the final image with a 2px PSF to reduce artifacts
    Returns:
        img_deconv (np.ndarray): NxN deconvolved image
    """

    # Transform the image and the PSF
    img_fft = fft.fft2(img, norm='ortho')
    psf_fft = fft.fft2(fft.ifftshift(psf), norm='backward')

    # Calculate the noise array & transform
    # noise = np.sqrt(img + sky_sigma**2)
    noise = np.random.normal(loc=0, scale=np.abs(noise), size=img.shape)
    noise_fft = np.abs(fft.fft2(noise, norm='ortho'))
    # Smooth the noise map
    filtsize = int(noise_fft.shape[0]*0.1)
    noise_fft = savgol_filter(noise_fft, axis=0, window_length=filtsize, polyorder=1)
    noise_fft = savgol_filter(noise_fft, axis=1, window_length=filtsize, polyorder=1)


    # Get the SNR
    snr = _fit_snr(img_fft, noise_fft, psf_fft)

    # If True, convolve with a narrow 2px PSF
    if convolve_nyquist:
        nyquist_psf = Gaussian2DKernel(2, x_size=img.shape[1], y_size=img.shape[0])
        nyquist_fft = fft.fft2(fft.ifftshift(nyquist_psf), norm='backward')
    else:
        nyquist_fft = 1

    # Deconvolve
    H = (nyquist_fft + 1/snr) / (psf_fft + 1/snr)
    img_corr = img_fft * H

    # Do an inverse transform
    img_deconv = np.real(fft.ifft2(img_corr, norm='ortho'))

    return img_deconv