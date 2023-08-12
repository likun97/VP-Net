# -*- coding: utf-8 -*-
"""
License: GNU-3.0
Code Reference:https://github.com/wasaCheney/IQA_pansharpening_python
"""

import numpy as np
from scipy import ndimage
import cv2

 
from scipy.signal import convolve2d


def partial_sums(x, kernel_size=8):
    """Calculate partial sums of array in boxes (kernel_size x kernel_size).
    This corresponds to:
    scipy.signal.convolve2d(x, np.ones((kernel_size, kernel_size)), mode='valid')
    >>> partial_sums(np.arange(12).reshape(3, 4), 2)
    array([[10, 14, 18],
           [26, 30, 34]])
    """
    assert len(x.shape) >= 2 and x.shape[0] >= kernel_size and x.shape[1] >= kernel_size
    sums = x.cumsum(axis=0).cumsum(axis=1)
    sums = np.pad(sums, 1)[:-1, :-1]
    return (
        sums[kernel_size:, kernel_size:]
        + sums[:-kernel_size, :-kernel_size]
        - sums[:-kernel_size, kernel_size:]
        - sums[kernel_size:, :-kernel_size]
    )


def universal_image_quality_index(x, y, kernel_size=8):
    """Compute the Universal Image Quality Index (UIQI) of x and y."""

    N = kernel_size ** 2

    x = x.astype(np.float)
    y = y.astype(np.float)
    e = np.finfo(np.float).eps

    # sums and auxiliary expressions based on sums
    S_x = partial_sums(x, kernel_size)
    S_y = partial_sums(y, kernel_size)
    PS_xy = S_x * S_y
    SSS_xy = S_x*S_x + S_y*S_y

    # sums of squares and product
    S_xx = partial_sums(x*x, kernel_size)
    S_yy = partial_sums(y*y, kernel_size)
    S_xy = partial_sums(x*y, kernel_size)

    num = 4 * PS_xy * (N * S_xy - PS_xy)
    den = (N*(S_xx + S_yy) - SSS_xy) / (SSS_xy + e)

    Q_s = (num) / (den + e)

    return np.mean(Q_s)


def universal_image_quality_index_conv(x, y, kernelsize=8):
    """Compute the Universal Image Quality Index (UIQI) of x and y.
    Not normalized with epsilon, and using scipy.signal.convolve2d."""

    N = kernelsize ** 2
    kernel = np.ones((kernelsize, kernelsize))

    x = x.astype(np.float)
    y = y.astype(np.float)

    # sums and auxiliary expressions based on sums
    S_x = convolve2d(x, kernel, mode='valid')
    S_y = convolve2d(y, kernel, mode='valid')
    PS_xy = S_x * S_y
    SSS_xy = S_x*S_x + S_y*S_y

    # sums of squares and product
    S_xx = convolve2d(x*x, kernel, mode='valid')
    S_yy = convolve2d(y*y, kernel, mode='valid')
    S_xy = convolve2d(x*y, kernel, mode='valid')

    Q_s = 4 * PS_xy * (N * S_xy - PS_xy) / (N*(S_xx + S_yy) - SSS_xy) / SSS_xy

    return np.mean(Q_s)

# -------------------------------------------------------------------
def rmse(img1, img2, dynamic_range=255):
  
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
        
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    
    print('RMSE')
    print((np.square(img1_ - img2_)).shape)             #  (400, 400, 4)

          
    return np.sqrt(np.mean( np.square(img1_ - img2_) )  )

# ttps://github.com/Berhinj/Pansharpening/blob/10cef8395d0f4a30b663148c24486e59562df656/Pansharpening/quality.py
# 改写成下面 上下的RMSE结果一样 OK 

def RMSEs(A, B):
    """ 
    Arguments:  A: np.ndarray             Stack of 1D bands
                B: np.ndarray             Stack of 1D bands
    Returns:    RMSEs: np.array           1D np array with one RMSE value per bands
    """
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    
    A = A.reshape(A.shape[0]*A.shape[1],A.shape[2])
    B = B.reshape(B.shape[0]*B.shape[1],B.shape[2])
    
    SE = (B - A)**2
    
    s = np.mean(SE) ** 0.5
    
    return s


def RASE(A, B):

    """  
    Arguments:
        A: np.ndarray   Stack of 1D bands
        B: np.ndarray   Stack of 1D bands
        RMSE: float     Root Mean Square Error
           
    Returns:            RASE: float
    """
    
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    
    A = A.reshape(A.shape[0]*A.shape[1],A.shape[2])
    B = B.reshape(B.shape[0]*B.shape[1],B.shape[2])
    
    SE = (B - A)**2
    
    RMSEs = np.mean(SE) ** 0.5

    Ms = np.mean(A, axis=0)
    
    R = np.mean(100*np.sqrt(RMSEs**2/A.shape[1])/Ms)
    
    return R

import math  


# -------------------------------------------------------------------
#  仿照matlab版本的代码  差别太大 ? 
def RASE_(img1, img2):
   
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    [m,n,p]=img2.shape
    print(m,n,p)
 
    C1= np.square(img1[:,:,0]- - img2[:,:,0])
    C2= np.square(img1[:,:,1]- - img2[:,:,1])
    C3= np.square(img1[:,:,2]- - img2[:,:,2])
    C4= np.square(img1[:,:,3]- - img2[:,:,3])
    print(C1.shape)
    
    C1 = C1.reshape(m*n)
    C2 = C2.reshape(m*n)
    C3 = C3.reshape(m*n)
    C4 = C4.reshape(m*n)
    print(C1.shape)
    
    C1 = np.sum(C1)/(m*n)
    C2 = np.sum(C2)/(m*n)
    C3 = np.sum(C3)/(m*n)
    C4 = np.sum(C4)/(m*n)
    print(C1)
    # C4 = C4.reshape(m*n)
    # C4 = np.sum(C4)/(m*n)

    C = C1+C2+C3+C4
 
    mean = np.mean(np.mean(np.mean(img1, axis=0), axis=0), axis=0)
    N=    math.sqrt((C/4))  *100/mean
    return N


def sam(img1, img2):
    """SAM for 3D image, shape (H, W, C); uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    assert img1.ndim == 3 and img1.shape[2] > 1, "image n_channels should be greater than 1"
    
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    
    inner_product = (img1_ * img2_).sum(axis=2)
    
    img1_spectral_norm = np.sqrt((img1_**2).sum(axis=2))
    img2_spectral_norm = np.sqrt((img2_**2).sum(axis=2))
    
    # numerical stability
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + np.finfo(np.float64).eps)).clip(min=0, max=1)
    return np.mean(np.arccos(cos_theta))*180/np.pi


def psnr(img1, img2, dynamic_range=255):
    """PSNR metric, img uint8 if 225; uint16 if 2047"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    mse = np.mean((img1_ - img2_)**2)
    if mse <= 1e-10:
        return np.inf
    return 20 * np.log10(dynamic_range / (np.sqrt(mse) + np.finfo(np.float64).eps))


def scc(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not  img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        #print(img1_[..., i].reshape[1, -1].shape)
        #test = np.corrcoef(img1_[..., i].reshape[1, -1], img2_[..., i].rehshape(1, -1))
        #print(type(test))
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')


def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size**2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size/2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu2_sq
#    print(mu1_mu2.shape)
    #print(sigma2_sq.shape)
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    
#    print(np.min(sigma1_sq + sigma2_sq), np.min(mu1_sq + mu2_sq))
    
    idx = ((sigma1_sq + sigma2_sq) < 1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) < 1e-8)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) >1e-8) * ((mu1_sq + mu2_sq) >1e-8)
    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
#    print(np.mean(qindex_map))
    
#    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
#    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
#    # sigma !=0 and mu == 0
#    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
#    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
#    # sigma != 0 and mu != 0
#    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
#    qindex_map[idx] =((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
#        (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    
    return np.mean(qindex_map)


def qindex(img1, img2, block_size=8):
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def _ssim(img1, img2, dynamic_range=255):
    """SSIM for 2D (one-band) image, shape (H, W); uint8 if 225; uint16 if 2047"""
    C1 = (0.01 * dynamic_range)**2
    C2 = (0.03 * dynamic_range)**2
    
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)  # kernel size 11
    window = np.outer(kernel, kernel.transpose())
    
    mu1 = cv2.filter2D(img1_, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2_, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1_**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[5:-5, 5:-5] - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, dynamic_range=255):
    """SSIM for 2D (H, W) or 3D (H, W, C) image; uint8 if 225; uint16 if 2047"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _ssim(img1, img2, dynamic_range)
    elif img1.ndim == 3:
        ssims = [_ssim(img1[..., i], img2[..., i], dynamic_range) for i in range(img1.shape[2])]
        return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def ergas(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mse = np.mean((img_fake_ - img_real_)**2)
        return 100 / scale * np.sqrt(mse / (mean_real**2 + np.finfo(np.float64).eps))
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        mses = ((img_fake_ - img_real_)**2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100 / scale * np.sqrt((mses / (means_real**2 + np.finfo(np.float64).eps)).mean())
    else:
        raise ValueError('Wrong input image dimensions.')
        
        
# https://github.com/Berhinj/Pansharpening/blob/10cef8395d0f4a30b663148c24486e59562df656/Pansharpening/quality.py
# 根据这个网站的RMSE改写成下面 上下 不一样
def ERGAS_(A, B):

    A = A.astype(np.float64)
    B = B.astype(np.float64)
    A = A.reshape(A.shape[0]*A.shape[1],A.shape[2])
    B = B.reshape(B.shape[0]*B.shape[1],B.shape[2])
    
    SE = (B - A)**2
    RMSEs = np.mean(SE) ** 0.5
    f_ratio = 1 
    # Mean
    Ms = np.average(B, 0)
    # Relative Dimensionless Global Error in Synthesis
    ERGAS = 100 * f_ratio**2 * (np.sum((RMSEs/Ms)**2)/A.shape[1])**0.5
    return ERGAS



# -------------------------------------------------------------------
# observation model
def gaussian2d(N, std):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2)
    t1, t2 = np.meshgrid(t, t)
    std = np.double(std)
    w = np.exp(-0.5 * (t1 / std)**2) * np.exp(-0.5 * (t2 / std)**2) 
    return w


def kaiser2d(N, beta):
    t = np.arange(-(N - 1) // 2, (N + 2) // 2) / np.double(N - 1)
    t1, t2 = np.meshgrid(t, t)
    t12 = np.sqrt(t1 * t1 + t2 * t2)
    w1 = np.kaiser(N, beta)
    w = np.interp(t12, t, w1)
    w[t12 > t[-1]] = 0
    w[t12 < t[0]] = 0
    return w


def fir_filter_wind(Hd, w):
    """
    compute fir (finite impulse response) filter with window method
    Hd: desired freqeuncy response (2D)
    w: window (2D)
    """
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd, 2)), 2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h, 2)
    h = h * w
    h = h / np.sum(h)
    return h


def GNyq2win(GNyq, scale=4, N=41):
    """Generate a 2D convolutional window from a given GNyq
    GNyq: Nyquist frequency
    scale: spatial size of PAN / spatial size of MS
    """
    #fir filter with window method
    fcut = 1 / scale
    alpha = np.sqrt(((N - 1) * (fcut / 2))**2 / (-2 * np.log(GNyq)))
    H = gaussian2d(N, alpha)
    Hd = H / np.max(H)
    w = kaiser2d(N, 0.5)
    h = fir_filter_wind(Hd, w)
    return np.real(h)


def mtf_resize(img, satellite='QuickBird', scale=4):
    # satellite GNyq
    scale = int(scale)
    if satellite == 'QuickBird':
        GNyq = [0.34, 0.32, 0.30, 0.22]  # Band Order: B,G,R,NIR
        GNyqPan = 0.15
    elif satellite == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]  # Band Order: B,G,R,NIR
        GNyqPan = 0.17
    else:
        raise NotImplementedError('satellite: QuickBird or IKONOS')
    # lowpass
    img_ = img.squeeze()
    img_ = img_.astype(np.float64)
    if img_.ndim == 2:  # Pan
        H, W = img_.shape
        lowpass = GNyq2win(GNyqPan, scale, N=41)
    elif img_.ndim == 3:  # MS
        H, W, _ = img.shape
        lowpass = [GNyq2win(gnyq, scale, N=41) for gnyq in GNyq]
        lowpass = np.stack(lowpass, axis=-1)
    img_ = ndimage.filters.correlate(img_, lowpass, mode='nearest')
    # downsampling
    output_size = (H // scale, W // scale)
    img_ = cv2.resize(img_, dsize=output_size, interpolation=cv2.INTER_NEAREST)
    return img_


# -------------------------------------------------------------------
# No reference IQA
def D_lambda(img_fake, img_lm, block_size=32, p=1):
    """  Spectral distortion
                             img_fake,   generated HRMS
                             img_lm  ,   LRMS          """
                             
    assert img_fake.ndim == img_lm.ndim == 3, 'Images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'

    Q_fake = []
    Q_lm = []
    for i in range(C_f):
        for j in range(i+1, C_f):
            # for fake
            band1 = img_fake[..., i]
            band2 = img_fake[..., j]
            Q_fake.append(_qindex(band1, band2, block_size=block_size))
            # for real
            band1 = img_lm[..., i]
            band2 = img_lm[..., j]
            Q_lm.append(_qindex(band1, band2, block_size=block_size))
    Q_fake = np.array(Q_fake)
    Q_lm = np.array(Q_lm)
    D_lambda_index = (np.abs(Q_fake - Q_lm) ** p).mean()
    return D_lambda_index ** (1/p)


def D_s(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, q=1):
    """Spatial distortion
    img_fake, generated HRMS
    img_lm, LRMS
    pan, HRPan"""
    # fake and lm
    assert img_fake.ndim == img_lm.ndim == 3, 'MS images must be 3D!'
    H_f, W_f, C_f = img_fake.shape
    H_r, W_r, C_r = img_lm.shape
    assert H_f // H_r == W_f // W_r == scale, 'Spatial resolution should be compatible with scale'
    assert C_f == C_r, 'Fake and lm should have the same number of bands!'
    # fake and pan
    assert pan.ndim == 3, 'Panchromatic image must be 3D!'
    H_p, W_p, C_p = pan.shape
    assert C_p == 1, 'size of 3rd dim of Panchromatic image must be 1'
    assert H_f == H_p and W_f == W_p, "Pan's and fake's spatial resolution should be the same"
    # get LRPan, 2D
    pan_lr = mtf_resize(pan, satellite=satellite, scale=scale)
    #print(pan_lr.shape)
    # D_s
    Q_hr = []
    Q_lr = []
    for i in range(C_f):
        # for HR fake
        band1 = img_fake[..., i]
        band2 = pan[..., 0] # the input PAN is 3D with size=1 along 3rd dim
        #print(band1.shape)
        #print(band2.shape)
        Q_hr.append(_qindex(band1, band2, block_size=block_size))
        band1 = img_lm[..., i]
        band2 = pan_lr  # this is 2D
        #print(band1.shape)
        #print(band2.shape)
        Q_lr.append(_qindex(band1, band2, block_size=block_size))
    Q_hr = np.array(Q_hr)
    Q_lr = np.array(Q_lr)
    D_s_index = (np.abs(Q_hr - Q_lr) ** q).mean()
    return D_s_index ** (1/q)

def qnr(img_fake, img_lm, pan, satellite='QuickBird', scale=4, block_size=32, p=1, q=1, alpha=1, beta=1):
    """QNR - No reference IQA"""
    D_lambda_idx = D_lambda(img_fake, img_lm, block_size, p)
    D_s_idx = D_s(img_fake, img_lm, pan, satellite, scale, block_size, q)
    QNR_idx = (1 - D_lambda_idx) ** alpha * (1 - D_s_idx) ** beta
    return QNR_idx


def ref_evaluate(pred, gt):
    #reference metrics
    c_psnr = psnr(pred, gt)
    c_ssim = ssim(pred, gt)
    c_sam = sam(pred, gt)
    c_ergas = ergas(pred, gt)
    
    c_scc = scc(pred, gt)
    c_q = qindex(pred, gt)
    c_rmse = rmse(pred, gt)
    
    # c_rmse = RMSEs(pred, gt)
    # c_rase = RASE(pred, gt)
    # c_uiqi = universal_image_quality_index_conv(pred, gt)
    # c_uiqi =  universal_image_quality_index(pred, gt)

    return [c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q, c_rmse]

def no_ref_evaluate(pred, pan, hs):
    #no reference metrics
    c_D_lambda = D_lambda(pred, hs)
    c_D_s = D_s(pred, hs, pan)
    c_qnr = qnr(pred, hs, pan)
    
    return [c_D_lambda, c_D_s, c_qnr]





